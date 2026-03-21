"""
扩散核心
实现前向扩散和反向扩散，支持跨语言渐变扩散
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable


class NoiseScheduler:
    """噪声调度器"""
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.timesteps = timesteps
        
        # 计算beta
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            # Cosine schedule
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        else:
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # 计算alpha
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 前向扩散系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 反向扩散系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def to(self, device: torch.device) -> "NoiseScheduler":
        """移动到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class CrossLingualDiffusion:
    """跨语言渐变扩散过程
    
    核心思想：从源语言嵌入渐变到目标语言嵌入，而不是加噪到纯噪声
    - t=0: 源语言嵌入
    - t=T: 目标语言嵌入
    - 中间: 插值 + 噪声
    
    注意：处理变长序列时，使用零填充对齐
    """
    
    def __init__(self, scheduler: NoiseScheduler, interpolation_strength: float = 0.8):
        self.scheduler = scheduler
        self.timesteps = scheduler.timesteps
        self.interpolation_strength = interpolation_strength  # 语言插值强度 (0-1)
    
    def get_interpolation_factor(self, t: torch.Tensor) -> torch.Tensor:
        """获取时间步 t 对应的语言插值因子
        
        t=0 -> factor=0 (源语言)
        t=T -> factor=1 (目标语言)
        """
        # 使用 sigmoid 平滑过渡
        normalized_t = t.float() / self.timesteps
        # 使用 smoothstep 实现更平滑的过渡
        factor = normalized_t * normalized_t * (3 - 2 * normalized_t)
        return factor * self.interpolation_strength
    
    def _align_sequences(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """对齐两个序列到相同长度
        
        Args:
            x_source: [batch, seq_len_s, d_model]
            x_target: [batch, seq_len_t, d_model]
        
        Returns:
            x_source_aligned: 对齐后的源序列
            x_target_aligned: 对齐后的目标序列
            target_len: 目标长度
        """
        batch_size, seq_len_s, d_model = x_source.shape
        _, seq_len_t, _ = x_target.shape
        
        # 选择较长的长度作为目标
        target_len = max(seq_len_s, seq_len_t)
        
        # 如果长度相同，直接返回
        if seq_len_s == seq_len_t:
            return x_source, x_target, target_len
        
        # 对源序列进行填充或截断
        if seq_len_s < target_len:
            # 填充
            padding = torch.zeros(batch_size, target_len - seq_len_s, d_model, 
                                  device=x_source.device, dtype=x_source.dtype)
            x_source_aligned = torch.cat([x_source, padding], dim=1)
        else:
            # 截断
            x_source_aligned = x_source[:, :target_len, :]
        
        # 对目标序列进行填充或截断
        if seq_len_t < target_len:
            # 填充
            padding = torch.zeros(batch_size, target_len - seq_len_t, d_model,
                                  device=x_target.device, dtype=x_target.dtype)
            x_target_aligned = torch.cat([x_target, padding], dim=1)
        else:
            # 截断
            x_target_aligned = x_target[:, :target_len, :]
        
        return x_source_aligned, x_target_aligned, target_len
    
    def q_sample(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """跨语言前向扩散：从源语言渐变到目标语言
        
        Args:
            x_source: 源语言嵌入 [batch, seq_len, d_model]
            x_target: 目标语言嵌入 [batch, seq_len, d_model]
            t: 时间步 [batch]
            noise: 可选噪声
        
        Returns:
            x_t: 渐变后的嵌入
            noise: 使用的噪声
        """
        # 对齐序列长度
        x_source_aligned, x_target_aligned, target_len = self._align_sequences(x_source, x_target)
        
        if noise is None:
            noise = torch.randn_like(x_source_aligned)
        
        # 获取插值因子
        interp_factor = self.get_interpolation_factor(t).view(-1, 1, 1)
        
        # 语言插值
        x_interp = (1 - interp_factor) * x_source_aligned + interp_factor * x_target_aligned
        
        # 添加噪声
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x_t = sqrt_alpha * x_interp + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def q_sample_single_lang(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单语言前向扩散（用于训练稳定性）
        
        Args:
            x_0: 初始嵌入 [batch, seq_len, d_model]
            t: 时间步 [batch]
            noise: 可选噪声
        
        Returns:
            x_t: 加噪后的嵌入
            noise: 使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 获取系数
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        
        # 扩展维度以匹配序列
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1)
        
        # 加噪
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """反向扩散：从x_t采样x_{t-1}
        
        Args:
            x_t: 当前噪声状态 [batch, seq_len, d_model]
            t: 当前时间步 [batch]
            predicted_noise: 预测的噪声
        
        Returns:
            x_{t-1}
        """
        # 获取系数
        sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        beta = self.scheduler.betas[t]
        
        # 扩展维度
        sqrt_recip_alpha = sqrt_recip_alpha.view(-1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1)
        beta = beta.view(-1, 1, 1)
        
        # 计算均值
        mean = sqrt_recip_alpha * (x_t - beta * predicted_noise / sqrt_one_minus_alpha)
        
        # 添加噪声（除了t=0）
        if t[0] > 0:
            posterior_var = self.scheduler.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(posterior_var) * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """从噪声状态预测原始嵌入
        
        Args:
            x_t: 当前噪声状态
            t: 时间步
            predicted_noise: 预测的噪声
        
        Returns:
            预测的原始嵌入
        """
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x_0 = (x_t - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
        return x_0


class DiffusionProcess:
    """标准扩散过程（向后兼容）"""
    
    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler
        self.timesteps = scheduler.timesteps
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向扩散：从x_0采样x_t
        
        Args:
            x_0: 初始嵌入 [batch, seq_len, d_model]
            t: 时间步 [batch]
            noise: 可选噪声
        
        Returns:
            x_t: 加噪后的嵌入
            noise: 使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 获取系数
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        
        # 扩展维度以匹配序列
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1)
        
        # 加噪
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """反向扩散：从x_t采样x_{t-1}
        
        Args:
            x_t: 当前噪声状态 [batch, seq_len, d_model]
            t: 当前时间步 [batch]
            predicted_noise: 预测的噪声
        
        Returns:
            x_{t-1}
        """
        # 获取系数
        sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        beta = self.scheduler.betas[t]
        
        # 扩展维度
        sqrt_recip_alpha = sqrt_recip_alpha.view(-1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1)
        beta = beta.view(-1, 1, 1)
        
        # 计算均值
        mean = sqrt_recip_alpha * (x_t - beta * predicted_noise / sqrt_one_minus_alpha)
        
        # 添加噪声（除了t=0）
        if t[0] > 0:
            posterior_var = self.scheduler.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(posterior_var) * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    def q_sample_full(
        self,
        x_0: torch.Tensor,
        target_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """完整前向扩散到纯噪声
        
        Args:
            x_0: 初始嵌入
            target_len: 目标长度（用于变长序列）
        
        Returns:
            x_T: 纯噪声
            noises: 所有时间步的噪声
            t: 最终时间步
        """
        batch_size = x_0.size(0)
        t = torch.full((batch_size,), self.timesteps - 1, dtype=torch.long, device=x_0.device)
        
        noise = torch.randn_like(x_0)
        x_T, _ = self.q_sample(x_0, t, noise)
        
        return x_T, noise, t


class DDIMSampler:
    """DDIM采样器，加速推理"""
    
    def __init__(self, scheduler: NoiseScheduler, ddim_steps: int = 50):
        self.scheduler = scheduler
        self.timesteps = scheduler.timesteps
        self.ddim_steps = ddim_steps
        
        # 计算DDIM时间步
        self.ddim_timesteps = self._get_ddim_timesteps()
    
    def _get_ddim_timesteps(self) -> List[int]:
        """获取DDIM采样使用的时间步"""
        c = self.timesteps // self.ddim_steps
        ddim_timesteps = [i * c for i in range(self.ddim_steps)]
        ddim_timesteps = list(reversed(ddim_timesteps))
        return ddim_timesteps
    
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        predicted_noise: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM单步采样
        
        Args:
            x_t: 当前状态
            t: 当前时间步
            t_prev: 前一时间步
            predicted_noise: 预测的噪声
            eta: 随机性参数 (0=deterministic, 1=DDPM)
        
        Returns:
            x_{t-1}
        """
        device = x_t.device
        
        # 获取alpha
        alpha_t = self.scheduler.alphas_cumprod[t]
        alpha_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
        
        # 预测x_0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        sqrt_alpha_t = sqrt_alpha_t.view(1, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, 1, 1)
        
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
        # 计算方差
        sigma = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        
        # 计算方向指向x_t
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev - sigma ** 2)
        sqrt_one_minus_alpha_t_prev = sqrt_one_minus_alpha_t_prev.view(1, 1, 1)
        
        # 计算均值
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev).view(1, 1, 1)
        mean = sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * predicted_noise
        
        # 添加噪声
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = mean + sigma.view(1, 1, 1) * noise
        else:
            x_t_prev = mean
        
        return x_t_prev
    
    def sample(
        self,
        x_T: torch.Tensor,
        predict_noise_fn: Callable,
        callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """完整DDIM采样
        
        Args:
            x_T: 纯噪声
            predict_noise_fn: 噪声预测函数 (x_t, t) -> noise
            callback: 回调函数，用于可视化
        
        Returns:
            x_0
        """
        x_t = x_T
        
        for i, t in enumerate(self.ddim_timesteps[:-1]):
            t_prev = self.ddim_timesteps[i + 1]
            
            # 预测噪声
            t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device)
            predicted_noise = predict_noise_fn(x_t, t_tensor)
            
            # DDIM步骤
            x_t = self.ddim_step(x_t, t, t_prev, predicted_noise, eta=0.0)
            
            # 回调
            if callback:
                callback(t, x_t)
        
        return x_t


def get_diffusion(config) -> Tuple[DiffusionProcess, DDIMSampler]:
    """创建扩散过程和采样器"""
    scheduler = NoiseScheduler(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )
    
    diffusion = DiffusionProcess(scheduler)
    ddim_sampler = DDIMSampler(scheduler, ddim_steps=config.diffusion.ddim_steps)
    
    return diffusion, ddim_sampler


def get_cross_lingual_diffusion(config) -> Tuple[CrossLingualDiffusion, DDIMSampler]:
    """创建跨语言扩散过程和采样器"""
    scheduler = NoiseScheduler(
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )
    
    diffusion = CrossLingualDiffusion(
        scheduler,
        interpolation_strength=getattr(config.diffusion, 'interpolation_strength', 0.8)
    )
    ddim_sampler = DDIMSampler(scheduler, ddim_steps=config.diffusion.ddim_steps)
    
    return diffusion, ddim_sampler
