"""
Box Operations for REC (Referring Expression Comprehension) task
包含 bbox 格式转换、GIoU Loss 等工具函数
"""

import torch
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes):
    """
    将 [cx, cy, w, h] 格式转换为 [x1, y1, x2, y2] 格式
    Args:
        boxes: Tensor of shape (..., 4)
    Returns:
        Tensor of shape (..., 4)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes):
    """
    将 [x1, y1, x2, y2] 格式转换为 [cx, cy, w, h] 格式
    Args:
        boxes: Tensor of shape (..., 4)
    Returns:
        Tensor of shape (..., 4)
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1, boxes2):
    """
    计算两组boxes之间的IoU
    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        boxes2: Tensor of shape (M, 4) in [x1, y1, x2, y2] format
    Returns:
        iou: Tensor of shape (N, M)
        union: Tensor of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算交集
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    计算Generalized IoU (GIoU)
    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        boxes2: Tensor of shape (N, 4) in [x1, y1, x2, y2] format (必须与boxes1一一对应)
    Returns:
        giou: Tensor of shape (N,)
    """
    assert boxes1.shape == boxes2.shape, "boxes1 and boxes2 must have the same shape"
    
    # 计算交集
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    # 计算各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    
    # 计算最小外接矩形
    lt_enclosing = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, 0] * wh_enclosing[:, 1]
    
    # GIoU = IoU - (C - Union) / C
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)
    
    return giou


def giou_loss(pred_boxes, target_boxes):
    """
    计算 GIoU Loss
    Args:
        pred_boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        target_boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
    Returns:
        loss: scalar tensor
    """
    giou = generalized_box_iou(pred_boxes, target_boxes)
    loss = 1 - giou
    return loss.mean()


def l1_loss(pred_boxes, target_boxes):
    """
    计算 L1 Loss for boxes
    Args:
        pred_boxes: Tensor of shape (N, 4)
        target_boxes: Tensor of shape (N, 4)
    Returns:
        loss: scalar tensor
    """
    return F.l1_loss(pred_boxes, target_boxes, reduction='mean')


def box_loss(pred_boxes, target_boxes, l1_weight=5.0, giou_weight=2.0):
    """
    组合 L1 Loss 和 GIoU Loss
    Args:
        pred_boxes: Tensor of shape (N, 4) in [cx, cy, w, h] format (归一化)
        target_boxes: Tensor of shape (N, 4) in [cx, cy, w, h] format (归一化)
        l1_weight: L1 loss 权重
        giou_weight: GIoU loss 权重
    Returns:
        total_loss: scalar tensor
        loss_dict: dict with individual losses
    """
    # L1 loss on [cx, cy, w, h]
    loss_l1 = l1_loss(pred_boxes, target_boxes)
    
    # GIoU loss (需要先转换为xyxy格式)
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    loss_giou = giou_loss(pred_xyxy, target_xyxy)
    
    total_loss = l1_weight * loss_l1 + giou_weight * loss_giou
    
    loss_dict = {
        'loss_l1': loss_l1,
        'loss_giou': loss_giou,
        'loss_total': total_loss
    }
    
    return total_loss, loss_dict


def compute_box_iou_accuracy(pred_boxes, target_boxes, threshold=0.5):
    """
    计算 Acc@threshold (IoU > threshold 的比例)
    Args:
        pred_boxes: Tensor of shape (N, 4) in [cx, cy, w, h] format (归一化)
        target_boxes: Tensor of shape (N, 4) in [cx, cy, w, h] format (归一化)
        threshold: IoU threshold
    Returns:
        accuracy: scalar tensor
        iou_values: Tensor of shape (N,)
    """
    # 转换为 xyxy 格式
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    
    # 计算IoU (对角线元素)
    giou = generalized_box_iou(pred_xyxy, target_xyxy)
    # 这里其实是GIoU，但我们需要IoU来计算accuracy
    # 重新计算标准IoU
    lt = torch.max(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    area1 = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area2 = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    
    union = area1 + area2 - inter
    iou_values = inter / (union + 1e-6)
    
    # 计算accuracy
    accuracy = (iou_values > threshold).float().mean()
    
    return accuracy, iou_values
