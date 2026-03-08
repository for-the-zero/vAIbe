"""
PPO训练器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Transition:
    """状态转移数据"""
    state: np.ndarray      # 棋盘状态 (4, 4)
    scores: np.ndarray     # 分数特征 (2,)
    action: int            # 采取的动作
    reward: float          # 奖励
    next_state: np.ndarray # 下一状态
    next_scores: np.ndarray # 下一分数
    done: bool             # 是否结束
    log_prob: float        # 动作的log概率
    value: float           # 状态价值
    valid_actions: np.ndarray  # 有效动作mask


class RolloutBuffer:
    """存储轨迹数据的缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0
    
    def push(self, transition: Transition) -> None:
        """添加一个转移"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def push_batch(self, transitions: List[Transition]) -> None:
        """批量添加转移"""
        for t in transitions:
            self.push(t)
    
    def get_all(self) -> List[Transition]:
        """获取所有数据"""
        return self.buffer.copy()
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer = []
        self.position = 0
    
    def __len__(self) -> int:
        return len(self.buffer)


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(
        self,
        model,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 训练统计
        self.stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'total_loss': deque(maxlen=100)
        }
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: 奖励序列 (T,)
            values: 价值序列 (T,)
            dones: 结束标志序列 (T,)
            next_value: 最后状态的下一个价值
            
        Returns:
            returns: 回报 (T,)
            advantages: 优势 (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        last_gae = 0
        last_return = next_value
        
        for t in reversed(range(T)):
            if dones[t]:
                next_value_t = 0
                last_gae = 0
            else:
                next_value_t = values[t + 1] if t + 1 < T else next_value
            
            delta = rewards[t] + self.gamma * next_value_t - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            
            last_return = rewards[t] + self.gamma * (1 - dones[t]) * last_return
            returns[t] = last_return
        
        return returns, advantages
    
    def update(self, buffer: RolloutBuffer) -> dict:
        """
        使用PPO更新模型
        
        Args:
            buffer: 存储轨迹数据的缓冲区
            
        Returns:
            训练统计信息
        """
        if len(buffer) < self.batch_size:
            return {}
        
        # 获取所有数据
        transitions = buffer.get_all()
        
        # 转换为数组
        states = np.array([t.state for t in transitions])
        scores = np.array([t.scores for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        dones = np.array([t.done for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions])
        old_values = np.array([t.value for t in transitions])
        valid_actions = np.array([t.valid_actions for t in transitions])
        
        # 计算优势和回报
        returns, advantages = self.compute_gae(rewards, old_values, dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states_t = torch.FloatTensor(states).to(self.device)
        scores_t = torch.FloatTensor(scores).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        valid_actions_t = torch.BoolTensor(valid_actions).to(self.device)
        
        # PPO更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        dataset_size = len(transitions)
        indices = np.arange(dataset_size)
        
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states_t[batch_indices]
                batch_scores = scores_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_valid = valid_actions_t[batch_indices]
                
                # 前向传播
                log_probs, values, entropy = self.model.evaluate_actions(
                    batch_states, batch_actions, batch_scores, batch_valid
                )
                
                # 策略损失 (PPO Clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 总损失
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy.mean()
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # 记录统计
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        for key, value in stats.items():
            self.stats[key].append(value)
        
        return stats
    
    def get_recent_stats(self) -> dict:
        """获取最近的训练统计"""
        return {key: np.mean(values) for key, values in self.stats.items() if values}


class TrainingStats:
    """训练统计记录器"""
    
    def __init__(self):
        self.games_played = 0
        self.total_steps = 0
        self.scores = []           # 每局累积分数
        self.situational_scores = []  # 每局平均局面分数
        self.max_tiles = []        # 每局最大砖块
        self.game_lengths = []     # 每局步数
        
        # 历史记录用于绘图
        self.score_history = []
        self.situational_history = []
        self.max_tile_history = []
        self.steps_history = []
        
        # 最佳记录
        self.best_score = 0
        self.best_max_tile = 0
    
    def record_game(
        self, 
        score: int, 
        situational_score: float,
        max_tile: int, 
        steps: int
    ) -> None:
        """记录一局游戏"""
        self.games_played += 1
        self.total_steps += steps
        
        self.scores.append(score)
        self.situational_scores.append(situational_score)
        self.max_tiles.append(max_tile)
        self.game_lengths.append(steps)
        
        self.score_history.append(score)
        self.situational_history.append(situational_score)
        self.max_tile_history.append(max_tile)
        self.steps_history.append(steps)
        
        if score > self.best_score:
            self.best_score = score
        if max_tile > self.best_max_tile:
            self.best_max_tile = max_tile
    
    def get_avg_stats(self, window: int = 100) -> dict:
        """获取平均统计"""
        def avg(lst):
            if not lst:
                return 0
            recent = lst[-window:]
            return sum(recent) / len(recent)
        
        return {
            'games_played': self.games_played,
            'total_steps': self.total_steps,
            'avg_score': avg(self.scores),
            'avg_situational': avg(self.situational_scores),
            'avg_max_tile': avg(self.max_tiles),
            'avg_game_length': avg(self.game_lengths),
            'best_score': self.best_score,
            'best_max_tile': self.best_max_tile,
            'recent_scores': self.scores[-10:] if self.scores else [],
            'recent_max_tiles': self.max_tiles[-10:] if self.max_tiles else []
        }


if __name__ == "__main__":
    from model import Game2048Transformer
    
    # 测试PPO训练器
    device = torch.device("cpu")
    model = Game2048Transformer().to(device)
    trainer = PPOTrainer(model, device=device)
    
    # 创建测试数据
    buffer = RolloutBuffer(capacity=1000)
    
    for _ in range(100):
        t = Transition(
            state=np.random.randn(4, 4).astype(np.float32),
            scores=np.random.rand(2).astype(np.float32),
            action=np.random.randint(0, 4),
            reward=np.random.randn(),
            next_state=np.random.randn(4, 4).astype(np.float32),
            next_scores=np.random.rand(2).astype(np.float32),
            done=np.random.rand() < 0.1,
            log_prob=np.random.randn(),
            value=np.random.randn(),
            valid_actions=np.ones(4, dtype=bool)
        )
        buffer.push(t)
    
    # 测试更新
    stats = trainer.update(buffer)
    print(f"Training stats: {stats}")
    
    # 测试统计
    training_stats = TrainingStats()
    for i in range(10):
        training_stats.record_game(
            score=1000 * (i + 1),
            situational_score=50.0 + i * 5,
            max_tile=2 ** (i + 5),
            steps=100 + i * 10
        )
    
    print(f"Average stats: {training_stats.get_avg_stats()}")
