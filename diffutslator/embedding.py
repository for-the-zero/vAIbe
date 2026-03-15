"""
嵌入层
语言特定的嵌入，包含位置编码和长度编码
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SinusoidalTimeEmbedding(nn.Module):
    """时间步的正弦嵌入（用于扩散）"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [batch] 时间步，范围 [0, T]
        返回: [batch, d_model]
        """
        # 归一化到 [0, 1]
        t = t.float().unsqueeze(-1)  # [batch, 1]
        
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)  # [batch, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class LanguageEmbedding(nn.Module):
    """语言特定的嵌入层"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 长度嵌入（用于变长序列）
        self.length_embedding = nn.Embedding(max_len + 1, d_model)
        
        # 缩放
        self.scale = math.sqrt(d_model)
        
        # 初始化
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.length_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        token_ids: [batch, seq_len]
        lengths: [batch] 可选，序列实际长度
        返回: [batch, seq_len, d_model]
        """
        # Token嵌入
        x = self.token_embedding(token_ids) * self.scale
        
        # 位置编码
        x = self.position_encoding(x)
        
        # 长度嵌入
        if lengths is not None:
            # 将长度信息广播到每个位置
            len_emb = self.length_embedding(lengths)  # [batch, d_model]
            x = x + len_emb.unsqueeze(1)  # 广播到序列长度
        
        return x
    
    def embed_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """生成纯噪声嵌入
        
        shape: (batch, seq_len, d_model)
        """
        return torch.randn(shape, device=device)


class DualLanguageEmbedding(nn.Module):
    """双语嵌入层，管理中英文嵌入"""
    
    def __init__(
        self,
        vocab_size_zh: int,
        vocab_size_en: int,
        d_model: int,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        self.zh_embedding = LanguageEmbedding(vocab_size_zh, d_model, max_len, dropout)
        self.en_embedding = LanguageEmbedding(vocab_size_en, d_model, max_len, dropout)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        lang: str,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        lang: 'zh' 或 'en'
        """
        if lang == 'zh':
            return self.zh_embedding(token_ids, lengths)
        else:
            return self.en_embedding(token_ids, lengths)
    
    def embed_tokens(
        self,
        zh_ids: Optional[torch.Tensor] = None,
        en_ids: Optional[torch.Tensor] = None,
        zh_lens: Optional[torch.Tensor] = None,
        en_lens: Optional[torch.Tensor] = None,
    ) -> tuple:
        """同时嵌入中英文"""
        zh_emb = None
        en_emb = None
        
        if zh_ids is not None:
            zh_emb = self.zh_embedding(zh_ids, zh_lens)
        if en_ids is not None:
            en_emb = self.en_embedding(en_ids, en_lens)
        
        return zh_emb, en_emb


class OutputProjection(nn.Module):
    """输出投影层，将隐藏状态投影回词表空间"""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        返回: [batch, seq_len, vocab_size] logits
        """
        return self.projection(x)


class DualOutputProjection(nn.Module):
    """双语输出投影层"""
    
    def __init__(self, d_model: int, vocab_size_zh: int, vocab_size_en: int):
        super().__init__()
        
        self.zh_projection = OutputProjection(d_model, vocab_size_zh)
        self.en_projection = OutputProjection(d_model, vocab_size_en)
    
    def forward(self, x: torch.Tensor, lang: str) -> torch.Tensor:
        if lang == 'zh':
            return self.zh_projection(x)
        else:
            return self.en_projection(x)
