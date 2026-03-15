"""
语言切换器
判断当前噪声状态更接近哪种语言
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LanguageSwitcher(nn.Module):
    """语言切换分类器
    
    输入: 噪声状态 x_t [batch, seq_len, d_model]
    输出: 语言概率 [batch, 2] -> [中文概率, 英文概率]
    """
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2类：中文/英文
        )
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x_t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_t: [batch, seq_len, d_model]
        mask: [batch, seq_len] 可选的mask
        
        返回: [batch, 2] logits (中文, 英文)
        """
        # 应用mask
        if mask is not None:
            x_t = x_t * mask.unsqueeze(-1)
        
        # 全局池化: [batch, seq_len, d_model] -> [batch, d_model, seq_len] -> [batch, d_model, 1]
        x = x_t.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)  # [batch, d_model]
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, x_t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[str, float]:
        """预测语言
        
        返回:
            lang: "zh" 或 "en"
            confidence: 置信度 [0, 1]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_t, mask)
            probs = F.softmax(logits, dim=-1)
            
            # 取第一个样本（假设batch=1）
            zh_prob = probs[0, 0].item()
            en_prob = probs[0, 1].item()
            
            if zh_prob > en_prob:
                return "zh", zh_prob
            else:
                return "en", en_prob
    
    def get_probabilities(self, x_t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取中文和英文的概率
        
        返回:
            zh_probs: [batch] 中文概率
            en_probs: [batch] 英文概率
        """
        logits = self.forward(x_t, mask)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 0], probs[:, 1]


class AdaptiveSwitcher(nn.Module):
    """自适应语言切换器
    
    根据扩散时间步动态调整切换策略
    - 早期（高噪声）：更激进的切换
    - 后期（低噪声）：更保守的切换
    """
    
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        switch_threshold: float = 0.6,  # 切换阈值
    ):
        super().__init__()
        
        self.switch_threshold = switch_threshold
        
        # 基础切换器
        self.base_switcher = LanguageSwitcher(d_model, hidden_dim, dropout)
        
        # 时间调制
        self.time_modulation = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_t: [batch, seq_len, d_model]
        t: [batch] 时间步，用于调制
        """
        # 基础预测
        logits = self.base_switcher(x_t, mask)
        
        # 时间调制（可选）
        if t is not None:
            # 归一化时间
            t_norm = t.float().unsqueeze(-1) / 1000.0  # [batch, 1]
            modulation = self.time_modulation(t_norm)  # [batch, 2]
            logits = logits * modulation
        
        return logits
    
    def should_switch(
        self,
        x_t: torch.Tensor,
        current_lang: str,
        t: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, str, float]:
        """判断是否应该切换语言
        
        返回:
            should_switch: 是否切换
            new_lang: 新语言
            confidence: 置信度
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_t, t, mask)
            probs = F.softmax(logits, dim=-1)
            
            zh_prob = probs[0, 0].item()
            en_prob = probs[0, 1].item()
            
            # 判断
            predicted_lang = "zh" if zh_prob > en_prob else "en"
            confidence = max(zh_prob, en_prob)
            
            # 是否切换
            should_switch = (
                predicted_lang != current_lang and
                confidence > self.switch_threshold
            )
            
            return should_switch, predicted_lang, confidence


def create_switcher(config) -> LanguageSwitcher:
    """创建语言切换器"""
    return LanguageSwitcher(
        d_model=config.model.d_model,
        hidden_dim=config.model.d_model // 2,
        dropout=config.model.dropout,
    )
