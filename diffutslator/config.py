"""
Diffutslator 配置文件
所有超参数集中管理
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ModelConfig:
    """模型配置"""
    d_model: int = 256           # 嵌入维度
    n_heads: int = 4             # 注意力头数
    n_layers: int = 4            # Transformer层数
    d_ff: int = 512              # 前馈网络维度
    max_len: int = 128           # 最大序列长度
    dropout: float = 0.1         # Dropout率
    
    # 词表
    vocab_size_zh: int = 8000    # 中文词表大小
    vocab_size_en: int = 8000    # 英文词表大小
    
    # 特殊token
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    mask_token: str = "<mask>"


@dataclass
class DiffusionConfig:
    """扩散过程配置"""
    timesteps: int = 1000        # 训练时的扩散步数
    ddim_steps: int = 50         # DDIM推理步数
    
    # 噪声调度 - 线性
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # 长度变化
    length_noise_scale: float = 0.3  # 扩散时长度变化的噪声程度


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64         # 批量大小（CPU擅长大批量）
    gradient_accumulation: int = 1  # 梯度累积步数
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    epochs: int = 10
    save_every: int = 1          # 每多少epoch保存一次
    eval_every: int = 100        # 每多少步评估一次
    
    # 快速验证模式
    quick_mode: bool = False
    quick_samples: int = 1000
    
    # 检查点
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None  # 恢复训练的检查点路径


@dataclass
class DataConfig:
    """数据配置"""
    # 数据集路径
    tatoeba_path: str = "../_dataset/tatoeba.tsv"
    cveto_zh_path: str = "../_dataset/cveto/train.zh"
    cveto_en_path: str = "../_dataset/cveto/train.en"
    
    # 数据处理
    max_samples: Optional[int] = None  # 最大样本数（None=全部）
    min_len: int = 2             # 最小句子长度
    max_len: int = 128           # 最大句子长度
    
    # 缓存
    use_cache: bool = True       # 是否缓存预处理后的数据
    cache_dir: str = ".cache"


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 项目根目录
    project_dir: str = ""
    
    def __post_init__(self):
        # 设置项目根目录
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 更新相对路径为绝对路径
        if not os.path.isabs(self.data.tatoeba_path):
            self.data.tatoeba_path = os.path.join(self.project_dir, self.data.tatoeba_path)
        if not os.path.isabs(self.data.cveto_zh_path):
            self.data.cveto_zh_path = os.path.join(self.project_dir, self.data.cveto_zh_path)
        if not os.path.isabs(self.data.cveto_en_path):
            self.data.cveto_en_path = os.path.join(self.project_dir, self.data.cveto_en_path)
        
        # 创建必要目录
        os.makedirs(os.path.join(self.project_dir, self.training.checkpoint_dir), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, self.data.cache_dir), exist_ok=True)
    
    @classmethod
    def quick(cls) -> "Config":
        """快速验证模式配置"""
        config = cls()
        config.training.quick_mode = True
        config.training.quick_samples = 1000
        config.training.epochs = 5
        config.training.batch_size = 32  # CPU擅长大批量
        config.data.max_samples = 1000
        return config


# 默认配置实例
default_config = Config()
