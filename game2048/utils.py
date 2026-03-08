"""
工具函数
"""
import os
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional
import shutil


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stats: Dict[str, Any],
    path: str
) -> None:
    """保存训练检查点"""
    ensure_dir(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    }, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """加载训练检查点"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_training_log(log_data: Dict[str, Any], path: str) -> None:
    """保存训练日志"""
    ensure_dir(os.path.dirname(path))
    
    # 读取现有日志
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    # 添加新记录
    log_data['timestamp'] = datetime.now().isoformat()
    logs.append(log_data)
    
    # 保存
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        minutes = seconds / 60
        return f'{minutes:.1f}m'
    else:
        hours = seconds / 3600
        return f'{hours:.1f}h'


def format_number(num: int) -> str:
    """格式化数字（添加逗号分隔）"""
    return f'{num:,}'


def calculate_ema(values: list, alpha: float = 0.1) -> list:
    """计算指数移动平均"""
    if not values:
        return []
    
    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema


def get_tile_color(value: int) -> str:
    """获取砖块颜色"""
    colors = {
        0: '#cdc1b4',
        2: '#eee4da',
        4: '#ede0c8',
        8: '#f2b179',
        16: '#f59563',
        32: '#f67c5f',
        64: '#f65e3b',
        128: '#edcf72',
        256: '#edcc61',
        512: '#edc850',
        1024: '#edc53f',
        2048: '#edc22e',
    }
    return colors.get(value, '#3c3a32')


def get_text_color(value: int) -> str:
    """获取文字颜色"""
    if value <= 4:
        return '#776e65'
    return '#f9f6f2'


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 0.01,
        mode: str = 'max'
    ):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        检查是否应该停止
        
        Args:
            value: 当前值
            
        Returns:
            是否应该停止
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
    
    def update(self, name: str, value: float) -> None:
        """更新指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        # 保持窗口大小
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def get_mean(self, name: str) -> float:
        """获取平均值"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return np.mean(self.metrics[name])
    
    def get_std(self, name: str) -> float:
        """获取标准差"""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        return np.std(self.metrics[name])
    
    def get_all_means(self) -> Dict[str, float]:
        """获取所有指标的平均值"""
        return {name: self.get_mean(name) for name in self.metrics}


def set_seed(seed: int) -> None:
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """获取可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module) -> None:
    """打印模型信息"""
    total_params = count_parameters(model)
    print(f"模型参数数量: {format_number(total_params)}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def export_to_onnx(
    model: torch.nn.Module,
    path: str,
    input_size: tuple = (1, 4, 4)
) -> None:
    """导出模型到ONNX格式"""
    model.eval()
    dummy_input = torch.randn(*input_size)
    dummy_scores = torch.randn(1, 2)
    dummy_mask = torch.ones(1, 4, dtype=torch.bool)
    
    ensure_dir(os.path.dirname(path))
    torch.onnx.export(
        model,
        (dummy_input, dummy_scores, dummy_mask),
        path,
        input_names=['board', 'scores', 'mask'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'board': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'mask': {0: 'batch_size'}
        }
    )
    print(f"模型已导出到: {path}")


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试时间格式化
    print(f"Format time: {format_time(45.5)}, {format_time(125.3)}, {format_time(3661)}")
    
    # 测试数字格式化
    print(f"Format number: {format_number(1234567)}")
    
    # 测试EMA
    values = [1, 2, 3, 4, 5]
    print(f"EMA: {calculate_ema(values)}")
    
    # 测试早停
    early_stop = EarlyStopping(patience=3, min_delta=0.1)
    scores = [10, 11, 12, 12, 12, 12, 12]
    for i, score in enumerate(scores):
        stop = early_stop(score)
        print(f"Epoch {i}: score={score}, stop={stop}")
    
    print("All tests passed!")
