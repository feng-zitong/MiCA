import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)
from utils.box_ops import compute_box_iou_accuracy, box_cxcywh_to_xyxy


def get_adapter_scaling_factors(model):
    """获取所有adapter的scaling_factor/gate参数统计信息"""
    scaling_info = {}
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查并联视觉adapter (DINOV2 blocks)
        if hasattr(module, 'parallel_v2_adapter') and hasattr(module.parallel_v2_adapter, 'gate'):
            gate = module.parallel_v2_adapter.gate
            if hasattr(module.parallel_v2_adapter, 'learnable_scale') and module.parallel_v2_adapter.learnable_scale:
                # learnable scale: 使用sigmoid
                scale_values = torch.sigmoid(gate).detach()
            else:
                # fixed scale: 直接使用gate值
                scale_values = gate.detach()
            
            # 提取层号
            layer_info = name.split('.')[-1] if 'blocks' in name else name
            scaling_info[f'vision_adapter_layer_{layer_info}'] = {
                'mean': scale_values.mean().item(),
                'std': scale_values.std().item(),
                'min': scale_values.min().item(),
                'max': scale_values.max().item()
            }
        
        # 检查并联文本adapter (CLIP ResidualAttentionBlocks)
        if name.endswith('parallel_v2_adapter') and hasattr(module, 'gate'):
            gate = module.gate
            if hasattr(module, 'learnable_scale') and module.learnable_scale:
                # learnable scale: 使用sigmoid
                scale_values = torch.sigmoid(gate).detach()
            else:
                # fixed scale: 直接使用gate值
                scale_values = gate.detach()
            
            # 提取层号
            parts = name.split('.')
            layer_info = None
            for part in parts:
                if part.isdigit():
                    layer_info = part
                    break
            if layer_info is None:
                layer_info = 'unknown'
            
            scaling_info[f'text_adapter_layer_{layer_info}'] = {
                'mean': scale_values.mean().item(),
                'std': scale_values.std().item(),
                'min': scale_values.min().item(),
                'max': scale_values.max().item()
            }
    
    return scaling_info


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    """通用训练函数，支持RIS和REC任务"""
    task = getattr(args, 'task', 'ris')
    
    if task == 'rec':
        return train_rec(train_loader, model, optimizer, scheduler, scaler, epoch, args)
    else:
        return train_ris(train_loader, model, optimizer, scheduler, scaler, epoch, args)


def train_ris(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    """RIS任务训练函数 (原有逻辑)"""
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')

    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        # forward
        with amp.autocast():
            pred, target, loss = model(image, text, target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()
        for name, param in model.named_parameters():
            if param.requires_grad :
                if param.grad is None:
                    print(f"{name} has no gradient.")  
        # metric
        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
    
    # 在每个epoch结束时打印scaling factors
    if dist.get_rank() == 0:  # 只在主进程打印
        scaling_info = get_adapter_scaling_factors(model)
        if scaling_info:
            logger.info(f"=== Epoch {epoch} Adapter Scaling Factors ===")
            for adapter_name, stats in scaling_info.items():
                logger.info(f"{adapter_name}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
                           f"Min={stats['min']:.4f}, Max={stats['max']:.4f}")
            logger.info("=" * 50)


def train_rec(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    """REC任务训练函数"""
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    acc_meter = AverageMeter('Acc@50', ':2.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, acc_meter],
        prefix="Training REC: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # REC任务数据: (image, text, mask, bbox)
        if len(batch_data) == 4:
            image, text, mask, target_box = batch_data
        else:
            # 兼容旧格式
            image, text, mask = batch_data
            target_box = None
        
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        if target_box is not None:
            target_box = target_box.cuda(non_blocking=True)

        # forward
        with amp.autocast():
            pred_box, target_box, loss = model(image, text, target_box=target_box)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()
        
        # metric: Acc@0.5
        acc, _ = compute_box_iou_accuracy(pred_box, target_box, threshold=0.5)
        
        dist.all_reduce(loss.detach())
        dist.all_reduce(acc)
        loss = loss / dist.get_world_size()
        acc = acc / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        acc_meter.update(acc.item() * 100, image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
    
    # 在每个epoch结束时打印scaling factors
    if dist.get_rank() == 0:
        scaling_info = get_adapter_scaling_factors(model)
        if scaling_info:
            logger.info(f"=== Epoch {epoch} REC Adapter Scaling Factors ===")
            for adapter_name, stats in scaling_info.items():
                logger.info(f"{adapter_name}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
                           f"Min={stats['min']:.4f}, Max={stats['max']:.4f}")
            logger.info("=" * 50)



@torch.no_grad()
def validate(val_loader, model, epoch, args):
    """通用验证函数，支持RIS和REC任务"""
    task = getattr(args, 'task', 'ris')
    
    if task == 'rec':
        return validate_rec(val_loader, model, epoch, args)
    else:
        return validate_ris(val_loader, model, epoch, args)


@torch.no_grad()
def validate_ris(val_loader, model, epoch, args):
    """RIS任务验证函数 (原有逻辑)"""
    iou_list = []
    model.eval()
    time.sleep(2)
    for imgs, texts, param in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        # inference
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        # process one batch
        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'],
                                                 param['inverse'],
                                                 param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > 0.35)
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}'.format(
        epoch, args.epochs, 100. * iou.item())
    logger.info(head + temp)
    
    # 在验证结束后也打印scaling factors
    scaling_info = get_adapter_scaling_factors(model)
    if scaling_info:
        logger.info(f"=== Epoch {epoch} Validation - Adapter Scaling Factors ===")
        for adapter_name, stats in scaling_info.items():
            logger.info(f"{adapter_name}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
                       f"Min={stats['min']:.4f}, Max={stats['max']:.4f}")
        logger.info("=" * 60)
    
    return iou.item(), prec


@torch.no_grad()
def validate_rec(val_loader, model, epoch, args):
    """REC任务验证函数"""
    iou_list = []
    model.eval()
    time.sleep(2)
    
    for imgs, texts, param in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        
        # inference - 预测bbox
        pred_boxes = model(imgs, texts)  # [B, 4] in [cx, cy, w, h] format
        
        # 将预测转换为原始图像坐标下的bbox
        for pred_box, ori_size, gt_bbox in zip(
            pred_boxes, param['ori_size'], param['gt_bbox']
        ):
            h, w = np.array(ori_size)
            pred_box = pred_box.cpu().numpy()  # [cx, cy, w, h] normalized
            gt_bbox = np.array(gt_bbox)  # [x, y, w, h] original scale
            
            # 将预测的归一化bbox转换为原始尺度
            # 注意：预测是在input_size上的，需要转换到原始图像尺度
            inp_size = args.input_size
            
            # 先转换为xyxy格式 (在input_size上)
            pred_cx, pred_cy, pred_w, pred_h = pred_box
            pred_x1 = (pred_cx - pred_w / 2) * inp_size
            pred_y1 = (pred_cy - pred_h / 2) * inp_size
            pred_x2 = (pred_cx + pred_w / 2) * inp_size
            pred_y2 = (pred_cy + pred_h / 2) * inp_size
            
            # 考虑letterbox变换：计算缩放比例和偏移
            scale = min(inp_size / h, inp_size / w)
            new_h, new_w = h * scale, w * scale
            bias_x, bias_y = (inp_size - new_w) / 2., (inp_size - new_h) / 2.
            
            # 去除偏移并反缩放
            pred_x1_ori = (pred_x1 - bias_x) / scale
            pred_y1_ori = (pred_y1 - bias_y) / scale
            pred_x2_ori = (pred_x2 - bias_x) / scale
            pred_y2_ori = (pred_y2 - bias_y) / scale
            
            # 裁剪到合理范围
            pred_x1_ori = np.clip(pred_x1_ori, 0, w)
            pred_y1_ori = np.clip(pred_y1_ori, 0, h)
            pred_x2_ori = np.clip(pred_x2_ori, 0, w)
            pred_y2_ori = np.clip(pred_y2_ori, 0, h)
            
            # GT bbox: [x, y, w, h] -> [x1, y1, x2, y2]
            gt_x1, gt_y1 = gt_bbox[0], gt_bbox[1]
            gt_x2, gt_y2 = gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]
            
            # 计算IoU
            inter_x1 = max(pred_x1_ori, gt_x1)
            inter_y1 = max(pred_y1_ori, gt_y1)
            inter_x2 = min(pred_x2_ori, gt_x2)
            inter_y2 = min(pred_y2_ori, gt_y2)
            
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            
            pred_area = (pred_x2_ori - pred_x1_ori) * (pred_y2_ori - pred_y1_ori)
            gt_area = gt_bbox[2] * gt_bbox[3]
            
            union_area = pred_area + gt_area - inter_area
            iou = inter_area / (union_area + 1e-6)
            iou_list.append(iou)
    
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    
    # 计算不同阈值下的精度
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    
    mean_iou = iou_list.mean()
    acc_50 = (iou_list > 0.5).float().mean()
    
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    
    head = 'REC Evaluation: Epoch=[{}/{}]  Acc@50={:.2f}  mIoU={:.2f}'.format(
        epoch, args.epochs, 100. * acc_50.item(), 100. * mean_iou.item())
    logger.info(head + temp)
    
    # 返回 Acc@50 作为主要指标 (用于保存best model)
    return acc_50.item(), prec


@torch.no_grad()
def inference(test_loader, model, args):
    """通用推理函数，支持RIS和REC任务"""
    task = getattr(args, 'task', 'ris')
    
    if task == 'rec':
        return inference_rec(test_loader, model, args)
    else:
        return inference_ris(test_loader, model, args)


@torch.no_grad()
def inference_rec(test_loader, model, args):
    """REC任务推理函数"""
    iou_list = []
    tbar = tqdm(test_loader, desc='REC Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        
        # 读取mask计算GT bbox
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to read mask: {param['mask_dir'][0]}")
            continue
        mask = mask / 255.
        
        # 从mask计算GT bbox [x, y, w, h]
        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            continue
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        gt_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        
        h, w = param['ori_size'].numpy()[0]
        
        # multiple sentences
        for sent in param['sents']:
            text = tokenize(sent, args.word_len, True)
            text = text.cuda(non_blocking=True)
            
            # inference - 预测bbox
            pred_box = model(img, text)  # [1, 4] in [cx, cy, w, h] normalized
            pred_box = pred_box[0].cpu().numpy()
            
            # 将预测的归一化bbox转换为原始尺度
            inp_size = args.input_size
            
            # 先转换为xyxy格式 (在input_size上)
            pred_cx, pred_cy, pred_w, pred_h = pred_box
            pred_x1 = (pred_cx - pred_w / 2) * inp_size
            pred_y1 = (pred_cy - pred_h / 2) * inp_size
            pred_x2 = (pred_cx + pred_w / 2) * inp_size
            pred_y2 = (pred_cy + pred_h / 2) * inp_size
            
            # 考虑letterbox变换
            scale = min(inp_size / h, inp_size / w)
            new_h, new_w = h * scale, w * scale
            bias_x, bias_y = (inp_size - new_w) / 2., (inp_size - new_h) / 2.
            
            # 去除偏移并反缩放
            pred_x1_ori = (pred_x1 - bias_x) / scale
            pred_y1_ori = (pred_y1 - bias_y) / scale
            pred_x2_ori = (pred_x2 - bias_x) / scale
            pred_y2_ori = (pred_y2 - bias_y) / scale
            
            # 裁剪到合理范围
            pred_x1_ori = np.clip(pred_x1_ori, 0, w)
            pred_y1_ori = np.clip(pred_y1_ori, 0, h)
            pred_x2_ori = np.clip(pred_x2_ori, 0, w)
            pred_y2_ori = np.clip(pred_y2_ori, 0, h)
            
            # GT bbox: [x, y, w, h] -> [x1, y1, x2, y2]
            gt_x1, gt_y1 = gt_bbox[0], gt_bbox[1]
            gt_x2, gt_y2 = gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]
            
            # 计算IoU
            inter_x1 = max(pred_x1_ori, gt_x1)
            inter_y1 = max(pred_y1_ori, gt_y1)
            inter_x2 = min(pred_x2_ori, gt_x2)
            inter_y2 = min(pred_y2_ori, gt_y2)
            
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            
            pred_area = (pred_x2_ori - pred_x1_ori) * (pred_y2_ori - pred_y1_ori)
            gt_area = gt_bbox[2] * gt_bbox[3]
            
            union_area = pred_area + gt_area - inter_area
            iou = inter_area / (union_area + 1e-6)
            iou_list.append(iou)
    
    logger.info('=> REC Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    
    # 计算不同阈值下的精度
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    
    mean_iou = iou_list.mean()
    acc_50 = (iou_list > 0.5).float().mean()
    
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
    
    logger.info('Acc@50={:.2f}'.format(100. * acc_50.item()))
    logger.info('Mean IoU={:.2f}'.format(100. * mean_iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100. * v))
    
    return acc_50.item(), prec


@torch.no_grad()
def inference_ris(test_loader, model, args):
    """RIS任务推理函数 (原有逻辑)"""
    # evaluation variables from lavt
    cum_I, cum_U = 0, 0 
    iou_list = []
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to read mask: {param['mask_dir'][0]}")
            continue
        # dump image & mask
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=mask)
        
        mask = mask / 255.
        # multiple sentences
        for sent in param['sents']:
            text = tokenize(sent, args.word_len, True)
            text = text.cuda(non_blocking=True)
            # inference
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > 0.35)
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            I = np.sum(inter)
            U = np.sum(union)
            iou = I / (U + 1e-6)
            iou_list.append(iou)
            # dump prediction
            if args.visualize:
                pred = np.array(pred*255, dtype=np.uint8)
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name),
                            img=pred)
            cum_I += I
            cum_U += U 

                
    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('Mean IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    logger.info('Overall IoU = %.2f\n' % (cum_I * 100. / cum_U))
    return iou.item(), prec
