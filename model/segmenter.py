import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model.clip import build_model
from .layers import Neck, Decoder, Projector
from .fusion import Fusion
from .dinov2.models.vision_transformer import vit_base,vit_large
from utils.box_ops import box_loss, box_cxcywh_to_xyxy

class DETRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 任务类型: 'ris' 或 'rec'
        self.task = getattr(cfg, 'task', 'ris')
        
        # ======================= 新版Adapter系统配置验证 =======================
        self._validate_adapter_config(cfg)
        
        # ======================= Text Encoder =======================
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        
        # 判断使用新版还是旧版adapter系统
        use_legacy = getattr(cfg, 'use_legacy_adapter', False)
        add_adapter_layer = getattr(cfg, 'txtual_adapter_layer', []) if use_legacy else []
        
        self.txt_backbone = build_model(
            state_dict=clip_model.state_dict(), 
            txt_length=cfg.word_len, 
            new_resolution=cfg.input_size, 
            add_adapter_layer=add_adapter_layer,  # 旧版兼容
            txt_adapter_dim=getattr(cfg, 'txt_adapter_dim', 64),
            config=cfg  # 新版系统通过config参数传递所有配置
        ).float()
        
        # Fusion模块
        self.fusion = Fusion(d_model=cfg.ladder_dim, nhead=cfg.nhead,
                           dino_layers=cfg.dino_layers, output_dinov2=cfg.output_dinov2)
    
        # Fix Text Backbone - 只有adapter参数可训练
        for param_name, param in self.txt_backbone.named_parameters():
            if 'adapter' not in param_name : 
                param.requires_grad = False       
   
        # ======================= Vision Encoder =======================
        state_dict = torch.load(cfg.dino_pretrain) 
        
        # 判断使用新版还是旧版adapter系统
        use_legacy = getattr(cfg, 'use_legacy_adapter', False)
        visual_add_adapter_layer = getattr(cfg, 'visual_adapter_layer', []) if use_legacy else []
        
        if cfg.dino_name == 'dino-base':
            self.dinov2 = vit_base(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=visual_add_adapter_layer,  # 旧版兼容
                visual_adapter_dim=getattr(cfg, 'visual_adapter_dim', 128),
                config=cfg,  # 新版系统通过config参数传递所有配置
            )
        else:
            self.dinov2 = vit_large(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=visual_add_adapter_layer,  # 旧版兼容
                visual_adapter_dim=getattr(cfg, 'visual_adapter_dim', 128),
                config=cfg,  # 新版系统通过config参数传递所有配置
            )
        self.dinov2.load_state_dict(state_dict, strict=False)

        # Fix Vision Backbone - 只有adapter参数可训练
        for param_name, param in self.dinov2.named_parameters():
            if 'adapter' not in param_name:
                param.requires_grad = False
        
        # ======================= Multi-Modal Decoder =======================
        self.neck = Neck(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = Decoder(num_layers=cfg.num_layers,
                              d_model=cfg.vis_dim,
                              nhead=cfg.num_head,
                              dim_ffn=cfg.dim_ffn,
                              dropout=cfg.dropout,
                              return_intermediate=cfg.intermediate)

        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        
        # ======================= REC任务的Box Head =======================
        if self.task == 'rec':
            # Box预测头: 输入是融合特征，输出是 [cx, cy, w, h]
            self.box_head = nn.Sequential(
                nn.Linear(cfg.vis_dim, cfg.vis_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(cfg.vis_dim, cfg.vis_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(cfg.vis_dim // 2, 4),
                nn.Sigmoid()  # 归一化到0-1
            )
        
        # ======================= 打印Adapter配置信息 =======================
        self._print_adapter_info(cfg)

    def _validate_adapter_config(self, cfg):
        """验证新版adapter配置的完整性，并在需要时启用旧版兼容"""
        
        # 检查基本开关
        use_multiscale = getattr(cfg, 'use_multiscale_lite_adapter', False)
        use_parallel_v2 = getattr(cfg, 'use_parallel_v2_adapter', False)
        use_parallel_v2_text = getattr(cfg, 'use_parallel_v2_text_adapter', False)
        
        # 检查是否有旧版配置
        old_visual_layers = getattr(cfg, 'visual_adapter_layer', [])
        old_text_layers = getattr(cfg, 'txtual_adapter_layer', [])
        has_old_config = bool(old_visual_layers or old_text_layers)
        
        # 如果新版都未启用，但有旧版配置，则自动启用旧版兼容模式
        if not any([use_multiscale, use_parallel_v2, use_parallel_v2_text]) and has_old_config:
            print("\n" + "="*60)
            print("🔄 检测到旧版Adapter配置，自动启用旧版兼容模式")
            print("="*60)
            
            # 激活旧版配置标志
            cfg.use_legacy_adapter = True
            
            if old_visual_layers:
                print(f"   📌 视觉Adapter: 层 {old_visual_layers}")
                print(f"      - 维度: {getattr(cfg, 'visual_adapter_dim', 128)}")
            
            if old_text_layers:
                print(f"   📌 文本Adapter: 层 {old_text_layers}")
                print(f"      - 维度: {getattr(cfg, 'txt_adapter_dim', 64)}")
            
            print("="*60 + "\n")
            return
        
        if not any([use_multiscale, use_parallel_v2, use_parallel_v2_text]):
            print("⚠️ 警告：所有新版adapter都未启用，且无旧版配置，模型将只使用预训练权重")
        
        # 检查串联adapter配置
        if use_multiscale:
            visual_config = getattr(cfg, 'visual_multiscale_lite_adapter', {})
            text_config = getattr(cfg, 'text_multiscale_lite_adapter', {})
            
            if not visual_config.get('enabled', False) and not text_config.get('enabled', False):
                print("⚠️ 警告：串联adapter已启用但视觉和文本配置都未enabled")
        
        # 检查并联adapter配置
        if use_parallel_v2:
            visual_v2_config = getattr(cfg, 'parallel_v2_visual_adapter', {})
            if not visual_v2_config.get('enabled', False):
                print("⚠️ 警告：视觉并联adapter已启用但配置未enabled")
                
        if use_parallel_v2_text:
            text_v2_config = getattr(cfg, 'parallel_v2_text_adapter', {})
            if not text_v2_config.get('enabled', False):
                print("⚠️ 警告：文本并联adapter已启用但配置未enabled")

    def _print_adapter_info(self, cfg):
        """打印adapter配置信息，用于调试"""
        
        print("\n" + "="*60)
        print("🔧 MiCA Adapter 配置信息")
        print("="*60)
        
        # 串联Adapter信息
        use_multiscale = getattr(cfg, 'use_multiscale_lite_adapter', False)
        print(f"📌 串联Adapter (Multi-Scale Lite): {'✅ 启用' if use_multiscale else '❌ 禁用'}")
        
        if use_multiscale:
            visual_config = getattr(cfg, 'visual_multiscale_lite_adapter', {})
            text_config = getattr(cfg, 'text_multiscale_lite_adapter', {})
            
            if visual_config.get('enabled', False):
                layers = visual_config.get('layers', [])
                print(f"   🎯 视觉串联: 层 {layers} (共{len(layers)}层)")
                print(f"      - 瓶颈维度: {visual_config.get('dim', 128)}")
                print(f"      - 卷积核: {visual_config.get('kernel_sizes', [3,5,7])}")
                
            if text_config.get('enabled', False):
                layers = text_config.get('layers', [])
                print(f"   🎯 文本串联: 层 {layers} (共{len(layers)}层)")
                print(f"      - 瓶颈维度: {text_config.get('dim', 64)}")
                print(f"      - 卷积核: {text_config.get('kernel_sizes', [3,5,7])}")
        
        # 并联Adapter信息
        use_parallel_v2 = getattr(cfg, 'use_parallel_v2_adapter', False)
        use_parallel_v2_text = getattr(cfg, 'use_parallel_v2_text_adapter', False)
        
        print(f"📌 并联Adapter (V2): {'✅ 启用' if any([use_parallel_v2, use_parallel_v2_text]) else '❌ 禁用'}")
        
        if use_parallel_v2:
            visual_v2_config = getattr(cfg, 'parallel_v2_visual_adapter', {})
            if visual_v2_config.get('enabled', False):
                layers = visual_v2_config.get('layers', [])
                print(f"   🎯 视觉并联: 层 {layers} (共{len(layers)}层)")
                print(f"      - 瓶颈维度: {visual_v2_config.get('dim', 128)}")
                print(f"      - FiLM调制: {'✅' if visual_v2_config.get('use_film', True) else '❌'}")
                print(f"      - 跨模态注意力头数: {visual_v2_config.get('num_heads', 8)}")
        
        if use_parallel_v2_text:
            text_v2_config = getattr(cfg, 'parallel_v2_text_adapter', {})
            if text_v2_config.get('enabled', False):
                layers = text_v2_config.get('layers', [])
                print(f"   🎯 文本并联: 层 {layers} (共{len(layers)}层)")
                print(f"      - 瓶颈维度: {text_v2_config.get('dim', 64)}")
                print(f"      - 注意力头数: {text_v2_config.get('num_heads', 4)}")
        
        # 旧版兼容性检查
        use_legacy = getattr(cfg, 'use_legacy_adapter', False)
        old_visual_layers = getattr(cfg, 'visual_adapter_layer', [])
        old_text_layers = getattr(cfg, 'txtual_adapter_layer', [])
        
        if use_legacy:
            print(f"📌 旧版Adapter模式: ✅ 已启用")
            if old_visual_layers:
                print(f"   🎯 视觉Adapter: 层 {old_visual_layers} (共{len(old_visual_layers)}层)")
                print(f"      - 维度: {getattr(cfg, 'visual_adapter_dim', 128)}")
            if old_text_layers:
                print(f"   🎯 文本Adapter: 层 {old_text_layers} (共{len(old_text_layers)}层)")
                print(f"      - 维度: {getattr(cfg, 'txt_adapter_dim', 64)}")
        elif old_visual_layers or old_text_layers:
            print(f"⚠️  旧版参数检测:")
            if old_visual_layers:
                print(f"   - visual_adapter_layer: {old_visual_layers} (已忽略)")
            if old_text_layers:
                print(f"   - txtual_adapter_layer: {old_text_layers} (已忽略)")
            print("   原因：新版Adapter已启用，旧版配置被覆盖")
            print("   建议：移除配置文件中的旧版参数，完全使用新版系统")
        
        print("="*60 + "\n")

    def forward(self, img, word, mask=None, target_box=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w (RIS任务)
            target_box: b, 4 (REC任务, [cx, cy, w, h] 归一化)
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis, word, state = self.fusion(img, word, self.txt_backbone, self.dinov2)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # ======================= 任务分支 =======================
        if self.task == 'rec':
            # REC任务: 预测bounding box
            # 全局平均池化得到特征向量
            fq_pooled = fq.mean(dim=[2, 3])  # b, c
            pred_box = self.box_head(fq_pooled)  # b, 4 [cx, cy, w, h]
            
            if self.training:
                assert target_box is not None, "REC训练需要target_box"
                # 计算box loss
                loss, loss_dict = box_loss(pred_box, target_box, l1_weight=5.0, giou_weight=2.0)
                
                # 检查并处理nan loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Warning: NaN/Inf loss detected in REC, using fallback loss")
                    loss = torch.tensor(0.5, device=pred_box.device, requires_grad=True)
                
                return pred_box.detach(), target_box, loss
            else:
                return pred_box.detach()
        else:
            # RIS任务: 预测segmentation mask (原有逻辑)
            # b, 1, 104, 104
            pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:],
                                         mode='nearest').detach()
                # 数值稳定性处理：clamp预测值防止BCE loss产生nan
                pred_clamped = torch.clamp(pred, min=-50, max=50)
                loss = F.binary_cross_entropy_with_logits(pred_clamped, mask) 
                
                # 检查并处理nan loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Warning: NaN/Inf loss detected, using fallback loss")
                    # 使用一个安全的fallback loss
                    loss = torch.tensor(0.5, device=pred.device, requires_grad=True)
                
                return pred.detach(), mask, loss
            else:
                return pred.detach()
