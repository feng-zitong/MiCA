"""
Multi-Scale Conv + Lite-Attention Adapter Implementation for MiCA
实现多尺度卷积 + 轻量注意力的高效adapter方案
结构: DownProj -> Multi-Scale DW-Conv -> PW-Conv -> Lite-MHSA -> UpProj -> Gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math


class MultiScaleDepthwiseConv(nn.Module):
    """
    多尺度深度可分离卷积模块
    使用不同kernel size的深度卷积捕获多尺度空间特征
    增强了数值稳定性
    """
    
    def __init__(self, dim: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        
        # 为每个kernel size创建深度卷积
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim, 
                kernel_size=k, 
                padding=k//2, 
                groups=dim,  # 深度卷积
                bias=False
            ) for k in kernel_sizes
        ])
        
        # 点卷积融合多尺度特征
        self.pw_conv = nn.Conv1d(dim * len(kernel_sizes), dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6)  # 增大eps提高数值稳定性
        
        # 初始化卷积权重以提高稳定性
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重以提高数值稳定性"""
        for conv in self.dw_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pw_conv.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d] -> 需要转换为 [B, d, L] 进行卷积
        """
        B, L, d = x.shape
        
        # 转换为卷积格式
        x_conv = x.transpose(1, 2)  # [B, d, L]
        
        # 应用多尺度深度卷积
        multi_scale_features = []
        for dw_conv in self.dw_convs:
            feat = dw_conv(x_conv)  # [B, d, L]
            # 防止卷积输出过大
            feat = torch.clamp(feat, min=-1e4, max=1e4)
            multi_scale_features.append(feat)
        
        # 拼接多尺度特征
        concat_feat = torch.cat(multi_scale_features, dim=1)  # [B, d*num_scales, L]
        
        # 点卷积融合
        fused_feat = self.pw_conv(concat_feat)  # [B, d, L]
        
        # 转换回原格式并归一化
        output = fused_feat.transpose(1, 2)  # [B, L, d]
        output = self.norm(output + x)  # 残差连接
        
        # 处理可能的nan值
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        
        return output


class LiteMultiHeadAttention(nn.Module):
    """
    轻量多头注意力模块
    在瓶颈维度进行注意力计算，大幅减少参数量和计算量
    增加了数值稳定性处理防止nan
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影（在瓶颈维度）
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # 轻量归一化
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, dim] 输入特征
        Returns:
            output: [B, L, dim] 注意力输出
        """
        B, L, d = x.shape
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数 - 添加数值稳定性clamp
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, L, L]
        attn = torch.clamp(attn, min=-50000, max=50000)  # 防止溢出
        attn = F.softmax(attn, dim=-1)
        # 处理可能的nan值
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, L, d)  # [B, L, dim]
        out = self.proj(out)
        
        # 残差连接和归一化
        return self.norm(out + x)


class MultiScaleLiteAdapter(nn.Module):
    """
    Multi-Scale Conv + Lite-Attention Adapter
    结合多尺度卷积的局部特征建模和轻量注意力的全局交互
    """
    
    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        kernel_sizes: list = [3, 5, 7],
        num_heads: int = 4,
        scaling_factor: Union[float, str] = 0.1,
        learnable_scale: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.learnable_scale = learnable_scale
        
        # 下投影层
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        
        # 多尺度深度卷积
        self.multi_scale_conv = MultiScaleDepthwiseConv(
            dim=bottleneck_dim, 
            kernel_sizes=kernel_sizes
        )
        
        # 轻量多头注意力
        self.lite_attention = LiteMultiHeadAttention(
            dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 上投影层
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 通道级门控机制
        if learnable_scale or scaling_factor == "learnable_scalar":
            self.gate = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer('gate', torch.full((d_model,), float(scaling_factor)))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使用较小的初始化值确保训练稳定性"""
        # Xavier初始化投影层
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.02)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 增加数值稳定性处理
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
        Returns:
            adapter输出 [batch_size, seq_len, d_model]
        """
        # 下投影到瓶颈维度
        h = self.down_proj(x)  # [B, L, bottleneck_dim]
        
        # 添加激活函数并限制范围防止数值爆炸
        h = F.relu(h, inplace=False)
        h = torch.clamp(h, min=0, max=1e4)
        
        # 多尺度卷积建模局部特征
        h = self.multi_scale_conv(h)  # [B, L, bottleneck_dim]
        
        # 轻量注意力建模全局交互
        h = self.lite_attention(h)  # [B, L, bottleneck_dim]
        
        # Dropout
        h = self.dropout(h)
        
        # 上投影回原维度
        h = self.up_proj(h)  # [B, L, d_model]
        
        # 处理可能的nan值
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 应用通道级门控
        if self.learnable_scale:
            return torch.sigmoid(self.gate) * h
        else:
            return self.gate * h


class MultiScaleLiteAdapterLayer(nn.Module):
    """
    Multi-Scale Lite Adapter层，包含残差连接逻辑
    """
    
    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        kernel_sizes: list = [3, 5, 7],
        num_heads: int = 4,
        scaling_factor: Union[float, str] = 0.1,
        learnable_scale: bool = False,
        residual_position: str = "before",  # "before" or "after"
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.residual_position = residual_position
        assert residual_position in ["before", "after"], \
            f"residual_position must be 'before' or 'after', got {residual_position}"
        
        self.adapter = MultiScaleLiteAdapter(
            d_model=d_model,
            bottleneck_dim=bottleneck_dim,
            kernel_sizes=kernel_sizes,
            num_heads=num_heads,
            scaling_factor=scaling_factor,
            learnable_scale=learnable_scale,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，包含残差连接
        Args:
            x: 输入特征
            residual: 残差连接的输入（用于after模式）
        Returns:
            输出特征
        """
        if self.residual_position == "before":
            # 模式：output = x + adapter(x)
            return x + self.adapter(x)
        else:
            # 模式：output = scale * (x + adapter(x))
            if residual is None:
                residual = x
            adapter_out = self.adapter(x)
            if hasattr(self.adapter, 'gate'):
                return self.adapter.gate * (residual + adapter_out)
            else:
                return residual + adapter_out


class MultiLayerMultiScaleLiteAdapter(nn.Module):
    """
    多层Multi-Scale Lite Adapter管理器
    """
    
    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        layers: list,
        kernel_sizes: list = [3, 5, 7],
        num_heads: int = 4,
        scaling_factor: Union[float, str] = 0.1,
        learnable_scale: bool = False,
        residual_position: str = "before",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = layers
        self.adapters = nn.ModuleDict()
        
        # 为每个指定层创建adapter
        for layer_idx in layers:
            self.adapters[str(layer_idx)] = MultiScaleLiteAdapterLayer(
                d_model=d_model,
                bottleneck_dim=bottleneck_dim,
                kernel_sizes=kernel_sizes,
                num_heads=num_heads,
                scaling_factor=scaling_factor,
                learnable_scale=learnable_scale,
                residual_position=residual_position,
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor, layer_idx: int, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用指定层的adapter
        Args:
            x: 输入特征
            layer_idx: 当前层索引
            residual: 残差连接输入
        Returns:
            处理后的特征
        """
        if layer_idx in self.layers and str(layer_idx) in self.adapters:
            return self.adapters[str(layer_idx)](x, residual)
        else:
            return x


def create_visual_multiscale_lite_adapter(config, d_model=None) -> Optional[MultiLayerMultiScaleLiteAdapter]:
    """
    根据配置创建视觉Multi-Scale Lite Adapter
    """
    if not getattr(config, 'use_multiscale_lite_adapter', False):
        return None
    
    visual_config = getattr(config, 'visual_multiscale_lite_adapter', {})
    if not visual_config.get('enabled', False):
        return None
    
    # 如果传入了d_model参数，使用传入的值，否则使用配置中的值
    model_dim = d_model if d_model is not None else getattr(config, 'vis_dim', 512)
    
    return MultiLayerMultiScaleLiteAdapter(
        d_model=model_dim,
        bottleneck_dim=visual_config.get('dim', 128),
        layers=visual_config.get('layers', []),
        kernel_sizes=visual_config.get('kernel_sizes', [3, 5, 7]),
        num_heads=visual_config.get('num_heads', 4),
        scaling_factor=visual_config.get('scaling_factor', 0.1),
        learnable_scale=visual_config.get('learnable_scale', False),
        residual_position=visual_config.get('residual_position', 'before'),
        dropout=visual_config.get('dropout', 0.1)
    )


def create_text_multiscale_lite_adapter(config) -> Optional[MultiLayerMultiScaleLiteAdapter]:
    """
    根据配置创建文本Multi-Scale Lite Adapter
    """
    if not getattr(config, 'use_multiscale_lite_adapter', False):
        return None
    
    text_config = getattr(config, 'text_multiscale_lite_adapter', {})
    if not text_config.get('enabled', False):
        return None
    
    return MultiLayerMultiScaleLiteAdapter(
        d_model=config.word_dim,
        bottleneck_dim=text_config.get('dim', 64),
        layers=text_config.get('layers', []),
        kernel_sizes=text_config.get('kernel_sizes', [3, 5, 7]),
        num_heads=text_config.get('num_heads', 4),
        scaling_factor=text_config.get('scaling_factor', 0.1),
        learnable_scale=text_config.get('learnable_scale', False),
        residual_position=text_config.get('residual_position', 'before'),
        dropout=text_config.get('dropout', 0.1)
    )


# 兼容性函数，保持向后兼容
def create_visual_serial_adapter(config, d_model=None):
    """向后兼容的函数，重定向到新的Multi-Scale Lite Adapter"""
    # 检查是否启用串联adapter
    if not getattr(config, 'use_multiscale_lite_adapter', False):
        return None
    return create_visual_multiscale_lite_adapter(config, d_model)

def create_text_serial_adapter(config):
    """向后兼容的函数，重定向到新的Multi-Scale Lite Adapter"""
    # 检查是否启用串联adapter
    if not getattr(config, 'use_multiscale_lite_adapter', False):
        return None
    return create_text_multiscale_lite_adapter(config)


# 简化的SerialAdapter类，用于对比和回退
class SerialAdapter(nn.Module):
    """
    简化的串联Adapter实现（用于对比）
    结构：下投影 -> GELU -> 上投影 -> 缩放
    """
    
    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        scaling_factor: Union[float, str] = 0.1,
        learnable_scale: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.learnable_scale = learnable_scale
        
        # 下投影层
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        
        # 非线性激活
        self.activation = nn.GELU()
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 上投影层
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        
        # 缩放因子
        if learnable_scale or scaling_factor == "learnable_scalar":
            self.scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('scale', torch.tensor(float(scaling_factor)))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.02)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return self.scale * h 