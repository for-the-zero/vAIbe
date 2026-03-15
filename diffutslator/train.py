"""
训练脚本
支持快速验证和完整训练，可暂停和恢复
"""

import os
import sys
import signal
import argparse
import time
from typing import Optional
from datetime import datetime

import torch

# 设置PyTorch使用所有CPU核心
torch.set_num_threads(os.cpu_count())
# 启用OpenMP并行
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from config import Config
from tokenizer import Tokenizer, train_tokenizers
from dataset import load_all_data, create_dataloaders
from embedding import DualLanguageEmbedding, DualOutputProjection
from model import create_model
from diffusion import get_diffusion, NoiseScheduler
from switcher import create_switcher
from utils import ProgressTracker, count_parameters, format_number, save_checkpoint, load_checkpoint


class Trainer:
    """训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cpu")  # CPU训练
        
        # 初始化组件
        self._init_components()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.should_stop = False
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_components(self):
        """初始化所有组件"""
        print("初始化训练组件...")
        
        # 加载或训练分词器
        tokenizer_path = os.path.join(self.config.project_dir, self.config.data.cache_dir)
        zh_tokenizer_path = os.path.join(tokenizer_path, "tokenizer_zh.json")
        en_tokenizer_path = os.path.join(tokenizer_path, "tokenizer_en.json")
        
        if os.path.exists(zh_tokenizer_path) and os.path.exists(en_tokenizer_path):
            print("  加载已有分词器...")
            self.zh_tokenizer = Tokenizer.load(zh_tokenizer_path)
            self.en_tokenizer = Tokenizer.load(en_tokenizer_path)
        else:
            print("  训练分词器...")
            # 先加载数据用于训练分词器
            train_pairs, _, _ = load_all_data(self.config)
            zh_texts = [p.zh for p in train_pairs]
            en_texts = [p.en for p in train_pairs]
            self.zh_tokenizer, self.en_tokenizer = train_tokenizers(
                self.config, zh_texts, en_texts
            )
            self.zh_tokenizer.save(zh_tokenizer_path)
            self.en_tokenizer.save(en_tokenizer_path)
        
        # 数据集
        print("  加载数据集...")
        train_pairs, val_pairs, test_pairs = load_all_data(self.config)
        self.train_loader, self.val_loader = create_dataloaders(
            train_pairs, val_pairs,
            self.zh_tokenizer, self.en_tokenizer,
            self.config
        )
        
        # 嵌入层
        print("  初始化嵌入层...")
        self.embedding = DualLanguageEmbedding(
            vocab_size_zh=self.zh_tokenizer.vocab_size_actual,
            vocab_size_en=self.en_tokenizer.vocab_size_actual,
            d_model=self.config.model.d_model,
            max_len=self.config.model.max_len,
            dropout=self.config.model.dropout,
        )
        
        # 输出投影
        self.output_proj = DualOutputProjection(
            d_model=self.config.model.d_model,
            vocab_size_zh=self.zh_tokenizer.vocab_size_actual,
            vocab_size_en=self.en_tokenizer.vocab_size_actual,
        )
        
        # 噪声预测模型
        print("  初始化模型...")
        self.model = create_model(self.config)
        
        # 语言切换器
        self.switcher = create_switcher(self.config)
        
        # 扩散过程
        self.diffusion, self.ddim_sampler = get_diffusion(self.config)
        self.scheduler = self.diffusion.scheduler.to(self.device)
        
        # 优化器
        all_params = (
            list(self.embedding.parameters()) +
            list(self.output_proj.parameters()) +
            list(self.model.parameters()) +
            list(self.switcher.parameters())
        )
        self.optimizer = optim.AdamW(
            all_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # 学习率调度器
        total_steps = len(self.train_loader) * self.config.training.epochs
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 打印模型信息
        total_params = sum(count_parameters(m) for m in [self.embedding, self.output_proj, self.model, self.switcher])
        print(f"  总参数量: {format_number(total_params)}")
    
    def _signal_handler(self, signum, frame):
        """信号处理：保存模型并退出"""
        print("\n\n收到中断信号，保存检查点...")
        self._save_checkpoint("interrupted")
        self.should_stop = True
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config.project_dir, self.config.training.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        path = os.path.join(checkpoint_dir, f"{name}.pt")
        
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'embedding': self.embedding.state_dict(),
            'output_proj': self.output_proj.state_dict(),
            'model': self.model.state_dict(),
            'switcher': self.switcher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'config': self.config,
        }
        
        torch.save(state, path)
        print(f"  检查点已保存: {path}")
    
    def _load_checkpoint(self, path: str):
        """加载检查点"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        self.current_epoch = state['epoch']
        self.global_step = state['global_step']
        self.best_loss = state['best_loss']
        
        self.embedding.load_state_dict(state['embedding'])
        self.output_proj.load_state_dict(state['output_proj'])
        self.model.load_state_dict(state['model'])
        self.switcher.load_state_dict(state['switcher'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        
        print(f"  从检查点恢复: epoch={self.current_epoch}, step={self.global_step}")
    
    def train_step(self, batch: dict) -> dict:
        """单步训练"""
        # 获取数据
        zh_ids = batch['zh_ids'].to(self.device)
        en_ids = batch['en_ids'].to(self.device)
        zh_lens = batch['zh_lens'].to(self.device)
        en_lens = batch['en_lens'].to(self.device)
        
        batch_size = zh_ids.size(0)
        
        # 嵌入
        zh_emb = self.embedding(zh_ids, 'zh', zh_lens)
        en_emb = self.embedding(en_ids, 'en', en_lens)
        
        # 随机时间步
        t_zh = torch.randint(0, self.config.diffusion.timesteps, (batch_size,), device=self.device)
        t_en = torch.randint(0, self.config.diffusion.timesteps, (batch_size,), device=self.device)
        
        # 前向扩散
        zh_noisy, zh_noise = self.diffusion.q_sample(zh_emb, t_zh)
        en_noisy, en_noise = self.diffusion.q_sample(en_emb, t_en)
        
        # 预测噪声
        zh_noise_pred = self.model(zh_noisy, t_zh, lang='zh')
        en_noise_pred = self.model(en_noisy, t_en, lang='en')
        
        # 噪声预测损失
        loss_noise_zh = self.mse_loss(zh_noise_pred, zh_noise)
        loss_noise_en = self.mse_loss(en_noise_pred, en_noise)
        
        # 语言切换损失
        # 标签: 0=中文, 1=英文
        zh_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        en_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        zh_switch_logits = self.switcher(zh_noisy)
        en_switch_logits = self.switcher(en_noisy)
        
        loss_switch = (
            self.ce_loss(zh_switch_logits, zh_labels) +
            self.ce_loss(en_switch_logits, en_labels)
        ) / 2
        
        # 总损失
        loss = loss_noise_zh + loss_noise_en + 0.1 * loss_switch
        
        # 反向传播（梯度累积）
        loss = loss / self.config.training.gradient_accumulation
        loss.backward()
        
        return {
            'loss': loss.item() * self.config.training.gradient_accumulation,
            'loss_noise_zh': loss_noise_zh.item(),
            'loss_noise_en': loss_noise_en.item(),
            'loss_switch': loss_switch.item(),
        }
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        self.embedding.train()
        self.output_proj.train()
        self.switcher.train()
        
        total_loss = 0
        num_batches = len(self.train_loader)
        
        tracker = ProgressTracker(
            total_steps=num_batches,
            desc=f"Epoch {epoch}/{self.config.training.epochs}"
        )
        
        batch_size = self.config.training.batch_size
        
        for batch_idx, batch in enumerate(self.train_loader):
            if self.should_stop:
                break
            
            # 训练步骤
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            
            # 梯度累积
            if (batch_idx + 1) % self.config.training.gradient_accumulation == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.embedding.parameters()) +
                    list(self.output_proj.parameters()) +
                    list(self.model.parameters()) +
                    list(self.switcher.parameters()),
                    1.0
                )
                
                # 更新参数
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # 更新进度
            tracker.update(batch_idx + 1, metrics['loss'])
            
            # 每个batch都打印进度（实时反馈）
            samples_speed = tracker.count * batch_size / tracker.elapsed if tracker.elapsed > 0 else 0
            progress_str = tracker.format_progress(metrics['loss'])
            progress_str = progress_str.replace("it/s", f"samples/s")
            print(f"\r{progress_str} ({samples_speed:.0f} samples/s)", end="", flush=True)
        
        print()  # 换行
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        self.embedding.eval()
        self.output_proj.eval()
        self.switcher.eval()
        
        total_loss = 0
        num_batches = min(len(self.val_loader), 50)  # 限制验证步数
        
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= num_batches:
                break
            
            zh_ids = batch['zh_ids'].to(self.device)
            en_ids = batch['en_ids'].to(self.device)
            zh_lens = batch['zh_lens'].to(self.device)
            en_lens = batch['en_lens'].to(self.device)
            
            batch_size = zh_ids.size(0)
            
            # 嵌入
            zh_emb = self.embedding(zh_ids, 'zh', zh_lens)
            en_emb = self.embedding(en_ids, 'en', en_lens)
            
            # 随机时间步
            t = torch.randint(0, self.config.diffusion.timesteps, (batch_size,), device=self.device)
            
            # 前向扩散
            zh_noisy, zh_noise = self.diffusion.q_sample(zh_emb, t)
            en_noisy, en_noise = self.diffusion.q_sample(en_emb, t)
            
            # 预测噪声
            zh_noise_pred = self.model(zh_noisy, t, lang='zh')
            en_noise_pred = self.model(en_noisy, t, lang='en')
            
            # 损失
            loss = self.mse_loss(zh_noise_pred, zh_noise) + self.mse_loss(en_noise_pred, en_noise)
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self):
        """完整训练"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch + 1, self.config.training.epochs + 1):
            if self.should_stop:
                break
            
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 打印结果
            print(f"\nEpoch {epoch} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            
            # 保存检查点
            if epoch % self.config.training.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch}")
            
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint("best")
                print("  新的最佳模型!")
        
        # 训练完成
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"训练完成! 总用时: {elapsed/60:.1f} 分钟")
        print(f"最佳验证损失: {self.best_loss:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Diffutslator 训练脚本")
    
    # 模式
    parser.add_argument("--quick", action="store_true", help="快速验证模式")
    parser.add_argument("--full", action="store_true", help="完整训练模式")
    
    # 参数覆盖
    parser.add_argument("--samples", type=int, default=None, help="使用的数据量")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="批量大小")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.quick:
        config = Config.quick()
        print("模式: 快速验证")
    else:
        config = Config()
        print("模式: 完整训练")
    
    # 覆盖参数
    if args.samples:
        config.data.max_samples = args.samples
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.resume:
        config.training.resume = args.resume
    
    # 打印配置
    print(f"\n配置:")
    print(f"  数据量: {config.data.max_samples or '全部'}")
    print(f"  批量大小: {config.training.batch_size}")
    print(f"  梯度累积: {config.training.gradient_accumulation}")
    print(f"  有效批量: {config.training.batch_size * config.training.gradient_accumulation}")
    print(f"  训练轮数: {config.training.epochs}")
    print(f"  学习率: {config.training.learning_rate}")
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 恢复训练
    if config.training.resume:
        trainer._load_checkpoint(config.training.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
