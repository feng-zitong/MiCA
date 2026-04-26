from typing import Tuple, Union, List, Any
import math
import numpy as np
import torch
import torch.nn.functional as F
import math
from torch import nn, Tensor

class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class TextAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        conv_block = BasicConv1d
        self.dense_branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.dense_branch2 = nn.Sequential(
            conv_block(in_channels + ch1x1, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.dense_branch3 = nn.Sequential(
            conv_block(in_channels + ch1x1 + ch3x3, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )
        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.D_fc1(x)
        B, P, D = x0.shape

        x0 = F.relu(x0, inplace=True)

        xs = x0[:, 1:, :].permute(0, 2, 1)  
        dense_branch1 = self.dense_branch1(xs)
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))
        outputs = [dense_branch1, dense_branch2, dense_branch3]
        outputs = torch.cat(outputs, dim=1).permute(0, 2, 1) 

        clstoken = x0[:, 0:1, :]
        outputs = torch.cat([clstoken, outputs], dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)

        if self.skip_connect:
            outputs += x
        return outputs
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class CrossModalAttention(nn.Module):
    def __init__(self, image_dim, text_dim, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        self.back_proj = nn.Linear(embed_dim, image_dim)
        
        # 初始化权重以提高稳定性
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.image_proj.weight, gain=0.02)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.xavier_uniform_(self.text_proj.weight, gain=0.02)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_uniform_(self.back_proj.weight, gain=0.02)
        nn.init.zeros_(self.back_proj.bias)
        
    def forward(self, image_features, text_features, attention_mask=None):

        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)
        
        # 添加数值稳定性处理
        image_features = torch.clamp(image_features, min=-1e4, max=1e4)
        text_features = torch.clamp(text_features, min=-1e4, max=1e4)

        query = image_features.permute(1, 0, 2)
        key = text_features.permute(1, 0, 2)
        value = text_features.permute(1, 0, 2)
        
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attention_mask)
        
        attn_output = self.back_proj(attn_output)
        
        # 处理可能的nan值
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
        
        return attn_output.permute(1, 0, 2)


class DenseAligner(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        skip_connect=False,
        embed_dim=128,
        num_heads=8,
        text_dim=512,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect
        conv_block = BasicConv2d
        self.dense_branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.dense_branch2 = nn.Sequential(
            conv_block(in_channels+ch1x1, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.dense_branch3 = nn.Sequential(
            conv_block(in_channels+ch1x1+ch3x3, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

        self.cross = CrossModalAttention(in_channels, text_dim, embed_dim, num_heads)

        self._initialize_weights()

    def _initialize_weights(self):
   
        for module in self.modules():
            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  
            elif isinstance(module, nn.Linear):
  
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)        

    def forward(self, x: Tensor, text_features,split_token=5) -> List[Tensor]:

        x0 = self.D_fc1(x)
        B,P,D = x0.shape
        W = H = int(math.sqrt(P-1))

        x0 = F.relu(x0, inplace=True)
        
        xs = x0[:,split_token:,:]
        # B, W, H, D / 8, 32, 32, 384 -> B, D, W, H / B, 384, 32, 32
        xs = xs.reshape(B,W,H,D).permute(0,3,1,2)
        # B, 192, 32, 32
        dense_branch1 = self.dense_branch1(xs)
        # B, 96, 32, 32
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        # B, 96, 32, 32
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))
        outputs = [dense_branch1, dense_branch2, dense_branch3]
        # B, 384, 32, 32
        outputs = torch.cat(outputs,dim=1) + xs
        # B, 384, 32, 32 -> B, 384, 1024 -> B, 1024, 384
        outputs = outputs.reshape(B,D,W*H).permute(0,2,1)
        # text fusion
        outputs = self.cross(outputs, text_features)

        clstoken =  x0[:,0:split_token,:]
        outputs = torch.cat([clstoken,outputs],dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)
        if self.skip_connect:
            outputs+=x
        return outputs


# ================ 新增的并联Adapter模块 ================

# 从serial_adapter.py导入需要的组件
from .serial_adapter import MultiScaleDepthwiseConv, LiteMultiHeadAttention


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)
    特征线性调制模块 - 使用文本CLS token生成gamma和beta参数对视觉特征进行通道级调制
    """
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        # 生成gamma和beta参数的线性层
        self.film = nn.Linear(cond_dim, feat_dim * 2)
        
        # 初始化策略：gamma=1, beta=0 (保持特征不变的初始状态)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # 将gamma部分的bias初始化为1，保证初始时不改变特征
        self.film.bias.data[:feat_dim] = 1.0

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        FiLM调制前向传播
        Args:
            feat: [B, C, H, W] 视觉特征图
            cond: [B, cond_dim] 条件特征 (文本CLS token)
        Returns:
            [B, C, H, W] 调制后的特征图
        """
        # 生成调制参数
        modulation = self.film(cond)  # [B, feat_dim * 2]
        gamma, beta = modulation.chunk(2, dim=-1)  # [B, feat_dim] × 2
        
        # 扩展维度以匹配特征图的空间维度
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        
        # 应用FiLM调制: gamma * feat + beta
        return gamma * feat + beta


class ParallelVisionAdapter(nn.Module):
    """
    新版并联视觉Adapter - 使用多尺度卷积+FiLM调制+Cross-Modal注意力+learnable_scale
    结构: Down-Proj → Multi-Scale DW-Conv → FiLM(text_cls) → Cross-Modal Attention(text_tokens) → Up-Proj → learnable_scale
    """
    
    def __init__(
        self,
        fc_in_channels: int,
        bottleneck_dim: int,
        kernel_sizes: list = [3, 5, 7],
        embed_dim: int = 128,
        num_heads: int = 8,
        text_dim: int = 512,
        scaling_factor: Union[float, str] = 0.1,
        learnable_scale: bool = True,
        dropout: float = 0.1,
        skip_connect: bool = False,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        
        self.skip_connect = skip_connect
        self.learnable_scale = learnable_scale
        self.use_film = use_film
        
        # 下投影层
        self.D_fc1 = nn.Linear(fc_in_channels, bottleneck_dim)
        
        # 多尺度深度卷积（用于2D特征）
        self.multi_scale_conv = MultiScaleDepthwiseConv2D(
            dim=bottleneck_dim, 
            kernel_sizes=kernel_sizes
        )
        
        # FiLM调制模块 - 使用文本CLS token进行特征调制
        if use_film:
            self.film = FiLM(cond_dim=text_dim, feat_dim=bottleneck_dim)
        
        # 跨模态注意力 - 使用文本tokens进行语义对齐
        self.cross_attention = CrossModalAttention(
            bottleneck_dim, text_dim, embed_dim, num_heads, dropout
        )
        
        # 上投影层
        self.D_fc2 = nn.Linear(bottleneck_dim, fc_in_channels)
        
        # learnable scale门控
        if learnable_scale or scaling_factor == "learnable_scalar":
            self.gate = nn.Parameter(torch.zeros(fc_in_channels))
        else:
            self.register_buffer('gate', torch.full((fc_in_channels,), float(scaling_factor)))
        
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor, text_features, split_token: int = 5) -> Tensor:
        """
        前向传播 - 实现文本特征分离使用策略
        Args:
            x: [B, P, D] 输入视觉特征
            text_features: [B, L, text_dim] 文本特征
            split_token: 分割token的位置
        """
        # 分离文本特征：CLS用于FiLM，tokens用于Cross-Attention
        text_cls = text_features[:, 0]      # [B, text_dim] - 文本CLS token
        text_tokens = text_features[:, 1:]  # [B, L-1, text_dim] - 文本剩余tokens
        
        # 下投影到瓶颈维度
        x0 = self.D_fc1(x)  # [B, P, bottleneck_dim]
        B, P, D = x0.shape
        
        x0 = F.relu(x0, inplace=True)
        
        # 分离cls token和spatial tokens
        clstoken = x0[:, :split_token, :]  # [B, split_token, D]
        xs = x0[:, split_token:, :]  # [B, spatial_tokens, D]
        
        spatial_tokens = xs.shape[1]  # 实际的spatial token数量
        
        # 计算最接近的完美平方数维度
        W = H = int(math.sqrt(spatial_tokens))
        if W * H < spatial_tokens:
            W = H = W + 1
        
        # 如果需要padding到完美平方数
        if W * H > spatial_tokens:
            padding_size = W * H - spatial_tokens
            padding = torch.zeros(B, padding_size, D, device=xs.device, dtype=xs.dtype)
            xs = torch.cat([xs, padding], dim=1)
        
        # 重塑为2D特征图进行卷积：[B, W*H, D] -> [B, D, W, H]
        xs = xs.reshape(B, W, H, D).permute(0, 3, 1, 2)
        
        # 应用多尺度卷积
        xs = self.multi_scale_conv(xs)  # [B, D, W, H]
        
        # 应用FiLM调制 - 使用文本CLS token
        if self.use_film:
            xs = self.film(xs, text_cls)  # [B, D, W, H]
        
        # 重塑回序列格式：[B, D, W, H] -> [B, W*H, D]
        xs = xs.permute(0, 2, 3, 1).reshape(B, W*H, D)
        
        # 跨模态注意力 - 使用文本tokens（不包括CLS）
        xs = self.cross_attention(xs, text_tokens)  # [B, W*H, D]
        
        # 如果之前有padding，现在去掉padding
        if W * H > spatial_tokens:
            xs = xs[:, :spatial_tokens, :]  # 去掉padding的部分
        
        # 重新拼接cls token
        outputs = torch.cat([clstoken, xs], dim=1)  # [B, P, D]
        
        # 残差连接
        outputs += x0
        
        # 上投影
        outputs = self.D_fc2(outputs)  # [B, P, fc_in_channels]
        
        # 应用learnable scale
        if self.learnable_scale:
            outputs = torch.sigmoid(self.gate) * outputs
        else:
            outputs = self.gate * outputs
        
        # skip connection
        if self.skip_connect:
            outputs += x
            
        return outputs


class ParallelTextAdapter(nn.Module):
    """
    新版并联文本Adapter - 使用多尺度卷积+Lite注意力+learnable_scale
    结构: Down-Proj → Multi-Scale DW-Conv → Lite-MHSA → Up-Proj → learnable_scale
    """
    
    def __init__(
        self,
        fc_in_channels: int,
        bottleneck_dim: int,
        kernel_sizes: list = [3, 5, 7],
        num_heads: int = 4,
        scaling_factor: Union[float, str] = 0.1,
        learnable_scale: bool = True,
        dropout: float = 0.1,
        skip_connect: bool = False,
    ) -> None:
        super().__init__()
        
        self.skip_connect = skip_connect
        self.learnable_scale = learnable_scale
        
        # 下投影层
        self.D_fc1 = nn.Linear(fc_in_channels, bottleneck_dim)
        
        # 多尺度深度卷积（用于1D序列）
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
        self.D_fc2 = nn.Linear(bottleneck_dim, fc_in_channels)
        
        # learnable scale门控
        if learnable_scale or scaling_factor == "learnable_scalar":
            self.gate = nn.Parameter(torch.zeros(fc_in_channels))
        else:
            self.register_buffer('gate', torch.full((fc_in_channels,), float(scaling_factor)))
        
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        Args:
            x: [B, L, D] 输入文本特征
        """
        # 下投影到瓶颈维度
        x0 = self.D_fc1(x)  # [B, L, bottleneck_dim]
        B, L, D = x0.shape
        
        # 使用clamp防止激活后数值过大
        x0 = F.relu(x0, inplace=False)
        x0 = torch.clamp(x0, min=0, max=1e4)
        
        # 分离cls token和sequence tokens
        clstoken = x0[:, 0:1, :]  # [B, 1, D]
        xs = x0[:, 1:, :]  # [B, L-1, D]
        
        # 应用多尺度卷积
        xs = self.multi_scale_conv(xs)  # [B, L-1, D]
        
        # 重新拼接cls token
        outputs = torch.cat([clstoken, xs], dim=1)  # [B, L, D]
        
        # 轻量注意力
        outputs = self.lite_attention(outputs)  # [B, L, D]
        
        # 残差连接
        outputs = outputs + x0
        
        # 上投影
        outputs = self.D_fc2(outputs)  # [B, L, fc_in_channels]
        
        # 应用learnable scale
        if self.learnable_scale:
            outputs = torch.sigmoid(self.gate) * outputs
        else:
            outputs = self.gate * outputs
        
        # 安全处理nan值
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # skip connection
        if self.skip_connect:
            outputs = outputs + x
            
        return outputs


class MultiScaleDepthwiseConv2D(nn.Module):
    """
    2D版本的多尺度深度可分离卷积模块
    用于视觉特征的空间建模
    """
    
    def __init__(self, dim: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        
        # 为每个kernel size创建深度卷积
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(
                dim, dim, 
                kernel_size=k, 
                padding=k//2, 
                groups=dim,  # 深度卷积
                bias=False
            ) for k in kernel_sizes
        ])
        
        # 点卷积融合多尺度特征
        self.pw_conv = nn.Conv2d(dim * len(kernel_sizes), dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W] 2D特征图
        """
        # 应用多尺度深度卷积
        multi_scale_features = []
        for dw_conv in self.dw_convs:
            feat = dw_conv(x)  # [B, D, H, W]
            multi_scale_features.append(feat)
        
        # 拼接多尺度特征
        concat_feat = torch.cat(multi_scale_features, dim=1)  # [B, D*num_scales, H, W]
        
        # 点卷积融合
        fused_feat = self.pw_conv(concat_feat)  # [B, D, H, W]
        
        # 归一化和残差连接
        output = self.norm(fused_feat + x)
        
        return output