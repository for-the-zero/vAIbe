"""
Transformer模型用于2048游戏决策
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class Game2048Transformer(nn.Module):
    """
    小型Transformer模型用于2048游戏
    
    输入: 
        - 棋盘状态 (batch, 4, 4) 或 (batch, 16) 
        - 可选: 分数特征 (batch, 2)
    
    输出:
        - policy: (batch, 4) 动作概率
        - value: (batch, 1) 状态价值
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 棋盘位置embedding
        # 每个格子: 0表示空，1-15表示log2(value)
        self.position_embedding = nn.Embedding(16, d_model)
        
        # 空间位置编码（4x4棋盘的行列位置）
        self.row_embedding = nn.Embedding(4, d_model // 2)
        self.col_embedding = nn.Embedding(4, d_model // 2)
        
        # 分数特征embedding
        self.score_embedding = nn.Linear(2, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 策略头（输出4个动作的概率）
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4)
        )
        
        # 价值头（输出状态价值）
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self, 
        board: torch.Tensor, 
        scores: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            board: (batch, 4, 4) 或 (batch, 16) 棋盘状态，值为log2(value)
            scores: (batch, 2) 可选的分数特征 [累积分数, 局面分数]
            mask: (batch, 4) 可选的动作mask，True表示有效动作
            
        Returns:
            policy: (batch, 4) 动作logits
            value: (batch, 1) 状态价值
        """
        batch_size = board.shape[0]
        
        # 展平棋盘
        if board.dim() == 3:
            board_flat = board.view(batch_size, -1)  # (batch, 16)
        else:
            board_flat = board
        
        # 将棋盘值转换为embedding索引（clamp到有效范围）
        board_indices = torch.clamp(board_flat.long(), 0, 15)
        
        # 位置embedding
        pos_embeddings = self.position_embedding(board_indices)  # (batch, 16, d_model)
        
        # 添加空间位置编码
        row_indices = torch.arange(4, device=board.device).repeat(4)
        col_indices = torch.arange(4, device=board.device).repeat_interleave(4)
        row_emb = self.row_embedding(row_indices)  # (16, d_model//2)
        col_emb = self.col_embedding(col_indices)  # (16, d_model//2)
        spatial_emb = torch.cat([row_emb, col_emb], dim=-1)  # (16, d_model)
        
        # 合并embedding
        x = pos_embeddings + spatial_emb.unsqueeze(0)  # (batch, 16, d_model)
        
        # 如果提供分数特征，作为第17个token
        if scores is not None:
            score_emb = self.score_embedding(scores).unsqueeze(1)  # (batch, 1, d_model)
            x = torch.cat([x, score_emb], dim=1)  # (batch, 17, d_model)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, 17 or 16, d_model)
        
        # 全局池化
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 输出头
        policy_logits = self.policy_head(x)  # (batch, 4)
        value = self.value_head(x)  # (batch, 1)
        
        # 应用动作mask
        if mask is not None:
            # 无效动作设为很小的值
            policy_logits = policy_logits.masked_fill(~mask, -1e9)
        
        return policy_logits, value
    
    def get_action(
        self, 
        board: torch.Tensor, 
        scores: torch.Tensor = None,
        mask: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        选择动作
        
        Args:
            board: (1, 4, 4) 或 (4, 4) 棋盘状态
            scores: (1, 2) 或 (2,) 分数特征
            mask: (1, 4) 或 (4,) 动作mask
            deterministic: 是否确定性选择
            
        Returns:
            action: 选择的动作
            log_prob: 动作的log概率
            value: 状态价值
        """
        # 确保维度正确
        if board.dim() == 2:
            board = board.unsqueeze(0)
        if scores is not None and scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.forward(board, scores, mask)
            probs = F.softmax(policy_logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                # 从概率分布采样
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            log_prob = F.log_softmax(policy_logits, dim=-1)[0, action].item()
        
        return action, log_prob, value.item()
    
    def evaluate_actions(
        self,
        board: torch.Tensor,
        actions: torch.Tensor,
        scores: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于训练）
        
        Args:
            board: (batch, 4, 4) 棋盘状态
            actions: (batch,) 采取的动作
            scores: (batch, 2) 分数特征
            mask: (batch, 4) 动作mask
            
        Returns:
            log_probs: (batch,) 动作log概率
            values: (batch, 1) 状态价值
            entropy: (batch,) 策略熵
        """
        policy_logits, values = self.forward(board, scores, mask)
        
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # 选择动作的log概率
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算熵
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_probs, values, entropy


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Game2048CNN(nn.Module):
    """
    CNN版本的2048模型（作为备选）
    更简单，可能更快
    """
    
    def __init__(self, channels: int = 64):
        super().__init__()
        
        # 输入: (batch, 1, 4, 4)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # 分数处理
        self.score_fc = nn.Linear(2, channels)
        
        # 输出头
        self.policy_head = nn.Sequential(
            nn.Linear(channels * 16 + channels, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(channels * 16 + channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(
        self, 
        board: torch.Tensor, 
        scores: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 添加channel维度
        x = board.unsqueeze(1)  # (batch, 1, 4, 4)
        
        # CNN特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch, channels*16)
        
        # 合并分数
        if scores is not None:
            score_feat = F.relu(self.score_fc(scores))
            x = torch.cat([x, score_feat], dim=-1)
        else:
            x = torch.cat([x, torch.zeros(x.size(0), 64, device=x.device)], dim=-1)
        
        # 输出
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)
        
        return policy_logits, value


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cpu")
    
    # Transformer模型
    model = Game2048Transformer().to(device)
    print(f"Transformer参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 4
    board = torch.randint(0, 12, (batch_size, 4, 4), dtype=torch.float32).to(device)
    scores = torch.rand(batch_size, 2).to(device)
    mask = torch.ones(batch_size, 4, dtype=torch.bool).to(device)
    
    policy_logits, value = model(board, scores, mask)
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # 测试动作选择
    action, log_prob, val = model.get_action(board[0], scores[0], mask[0])
    print(f"Action: {action}, Log prob: {log_prob:.4f}, Value: {val:.4f}")
    
    # CNN模型
    cnn_model = Game2048CNN().to(device)
    print(f"\nCNN参数量: {count_parameters(cnn_model):,}")
