"""
扩散模型
轻量级Transformer用于噪声预测
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from embedding import SinusoidalTimeEmbedding


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = q.size(0)
        
        # 线性变换并分头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 合并头
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(out)


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # 前馈 + 残差
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class NoisePredictor(nn.Module):
    """噪声预测网络
    
    输入: 加噪后的嵌入 x_t 和时间步 t
    输出: 预测的噪声
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 时间步嵌入
        self.time_embedding = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_t: [batch, seq_len, d_model] 加噪后的嵌入
        t: [batch] 时间步
        mask: [batch, seq_len] 可选的注意力mask
        
        返回: [batch, seq_len, d_model] 预测的噪声
        """
        batch_size, seq_len, _ = x_t.shape
        
        # 时间步嵌入
        t_emb = self.time_embedding(t)  # [batch, d_model]
        t_emb = self.time_mlp(t_emb)    # [batch, d_model]
        
        # 添加时间信息到每个位置
        x = x_t + t_emb.unsqueeze(1)
        
        # Transformer处理
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出
        x = self.output_norm(x)
        noise_pred = self.output_proj(x)
        
        return noise_pred


class DualNoisePredictor(nn.Module):
    """双语言噪声预测器
    
    共享核心网络，语言特定的输入/输出投影
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 时间步嵌入（共享）
        self.time_embedding = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # 语言特定的输入投影
        self.zh_input_proj = nn.Linear(d_model, d_model)
        self.en_input_proj = nn.Linear(d_model, d_model)
        
        # 共享Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 语言特定的输出投影
        self.zh_output_proj = nn.Linear(d_model, d_model)
        self.en_output_proj = nn.Linear(d_model, d_model)
        
        self.output_norm = nn.LayerNorm(d_model)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        lang: str = "zh",
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_t: [batch, seq_len, d_model]
        t: [batch]
        lang: "zh" 或 "en"
        """
        # 时间步嵌入
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # 语言特定输入投影
        if lang == "zh":
            x = self.zh_input_proj(x_t)
        else:
            x = self.en_input_proj(x_t)
        
        # 添加时间信息
        x = x + t_emb.unsqueeze(1)
        
        # 共享Transformer
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出归一化
        x = self.output_norm(x)
        
        # 语言特定输出投影
        if lang == "zh":
            noise_pred = self.zh_output_proj(x)
        else:
            noise_pred = self.en_output_proj(x)
        
        return noise_pred


def create_model(config) -> DualNoisePredictor:
    """创建模型"""
    model = DualNoisePredictor(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_len=config.model.max_len,
        dropout=config.model.dropout,
    )
    return model
