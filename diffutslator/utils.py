"""
工具函数
"""

import time
import math
from typing import Optional
from datetime import datetime


class Timer:
    """计时器，用于统计训练速度"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        self.count = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time:
            self.elapsed += time.time() - self.start_time
            self.count += 1
            self.start_time = None
    
    def reset(self):
        self.elapsed = 0
        self.count = 0
        self.start_time = None
    
    @property
    def avg_time(self) -> float:
        if self.count == 0:
            return 0
        return self.elapsed / self.count
    
    @property
    def speed(self) -> float:
        if self.elapsed == 0:
            return 0
        return self.count / self.elapsed


class ProgressTracker:
    """训练进度追踪器"""
    
    def __init__(self, total_steps: int, desc: str = "Training"):
        self.total_steps = total_steps
        self.desc = desc
        self.current_step = 0
        self.start_time = time.time()
        self.loss_history = []
    
    @property
    def elapsed(self) -> float:
        """已用时间"""
        return time.time() - self.start_time
    
    @property
    def count(self) -> int:
        """已处理步数"""
        return self.current_step
    
    def update(self, step: int, loss: Optional[float] = None):
        self.current_step = step
        if loss is not None:
            self.loss_history.append(loss)
    
    def format_progress(self, current_loss: Optional[float] = None) -> str:
        """格式化进度显示"""
        elapsed = time.time() - self.start_time
        progress = self.current_step / self.total_steps
        
        # 预计剩余时间
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = self._format_time(eta)
        else:
            eta_str = "--:--:--"
        
        # 速度
        speed = self.current_step / elapsed if elapsed > 0 else 0
        
        # 进度条
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        # 损失
        loss_str = f"loss={current_loss:.4f}" if current_loss is not None else ""
        
        return f"{self.desc}: |{bar}| {self.current_step}/{self.total_steps} [{self._format_time(elapsed)}<{eta_str}, {speed:.2f}it/s] {loss_str}"
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 0:
            return "--:--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def cosine_similarity(a, b):
    """计算余弦相似度"""
    import torch
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def count_parameters(model) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """格式化数字，添加千分位"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def get_timestamp() -> str:
    """获取时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    """确保目录存在"""
    import os
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, epoch: int, step: int, loss: float, path: str):
    """保存检查点"""
    import torch
    torch.save({
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path: str):
    """加载检查点"""
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
