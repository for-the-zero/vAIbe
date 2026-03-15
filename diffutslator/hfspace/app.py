"""
Diffutslator Hugging Face Space 应用
基于扩散模型的机器翻译演示
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gradio as gr
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import json


# ==================== 配置（与config.py保持一致，用于加载检查点）====================
@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_len: int = 128
    dropout: float = 0.1
    vocab_size_zh: int = 8000
    vocab_size_en: int = 8000
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    mask_token: str = "<mask>"


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    ddim_steps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    length_noise_scale: float = 0.3


@dataclass
class TrainingConfig:
    batch_size: int = 64
    gradient_accumulation: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    epochs: int = 10
    save_every: int = 1
    eval_every: int = 100
    quick_mode: bool = False
    quick_samples: int = 1000
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None


@dataclass
class DataConfig:
    tatoeba_path: str = ""
    cveto_zh_path: str = ""
    cveto_en_path: str = ""
    max_samples: Optional[int] = None
    min_len: int = 2
    max_len: int = 128
    use_cache: bool = True
    cache_dir: str = ".cache"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    project_dir: str = ""


# 创建一个假的config模块，用于加载检查点时反序列化
class _FakeConfigModule:
    Config = Config
    ModelConfig = ModelConfig
    DiffusionConfig = DiffusionConfig
    TrainingConfig = TrainingConfig
    DataConfig = DataConfig


# 将假模块注入sys.modules
sys.modules['config'] = _FakeConfigModule()


# ==================== 分词器 ====================
import re

class Tokenizer:
    """BPE分词器（与tokenizer.py兼容）"""
    
    def __init__(self, vocab_size: int = 8000, lang: str = "zh"):
        self.vocab_size = vocab_size
        self.lang = lang
        
        # 特殊token
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token, self.mask_token]
        
        # 词表
        self.token_to_id: dict = {}
        self.id_to_token: dict = {}
        
        # BPE合并规则
        self.merges: list = []
        self.bpe_ranks: dict = {}
    
    @property
    def vocab_size_actual(self) -> int:
        return len(self.token_to_id)
    
    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]
    
    @property
    def sos_id(self) -> int:
        return self.token_to_id[self.sos_token]
    
    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]
    
    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]
    
    def _is_chinese(self, char: str) -> bool:
        return '\u4e00' <= char <= '\u9fff'
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """预分词"""
        if self.lang == "zh":
            tokens = []
            current = ""
            for char in text:
                if self._is_chinese(char):
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(char)
                elif char.isalnum():
                    current += char.lower()
                else:
                    if current:
                        tokens.append(current)
                        current = ""
                    if char.strip():
                        tokens.append(char)
            if current:
                tokens.append(current)
            return tokens
        else:
            text = text.lower()
            tokens = re.findall(r"\w+|[^\w\s]", text)
            return tokens
    
    def _get_pairs(self, word: tuple) -> set:
        """获取词中的所有相邻字符对"""
        pairs = set()
        prev = word[0]
        for char in word[1:]:
            pairs.add((prev, char))
            prev = char
        return pairs
    
    def _apply_bpe(self, token: str) -> List[str]:
        """对单个token应用BPE"""
        if not token:
            return []
        
        word = tuple(token) + ('</w>',)
        
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # 找到rank最高的pair
            min_pair = None
            min_rank = float('inf')
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            
            if min_pair is None or min_rank == float('inf'):
                break
            
            # 合并
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == min_pair[0] and word[i + 1] == min_pair[1]:
                    new_word.append(min_pair[0] + min_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        
        return [t for t in word if t != '</w>']
    
    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """编码文本为token id序列"""
        tokens = self._pre_tokenize(text)
        
        ids = []
        if add_sos:
            ids.append(self.sos_id)
        
        for token in tokens:
            bpe_tokens = self._apply_bpe(token)
            for t in bpe_tokens:
                ids.append(self.token_to_id.get(t, self.unk_id))
        
        if add_eos:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """解码token id序列为文本"""
        tokens = []
        for id in ids:
            token = self.id_to_token.get(id, self.unk_token)
            if skip_special and token in self.special_tokens:
                continue
            token = token.replace('</w>', '')
            if token:
                tokens.append(token)
        
        if self.lang == "en":
            text = ' '.join(tokens)
            text = re.sub(r'\s+([.,!?;:\'\"])', r'\1', text)
            text = re.sub(r'([.,!?;:])([a-zA-Z])', r'\1 \2', text)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = ''.join(tokens)
        
        return text
    
    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """加载分词器"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"], lang=data["lang"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.bpe_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}
        tokenizer.special_tokens = data["special_tokens"]
        
        return tokenizer


# ==================== 模型组件 ====================
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SinusoidalTimeEmbedding(nn.Module):
    """时间步的正弦嵌入（用于扩散）"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(-1)
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class LanguageEmbedding(nn.Module):
    """语言特定的嵌入层"""
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.length_embedding = nn.Embedding(max_len + 1, d_model)
        self.scale = math.sqrt(d_model)
    
    def forward(self, token_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_embedding(token_ids) * self.scale
        x = self.position_encoding(x)
        if lengths is not None:
            len_emb = self.length_embedding(lengths)
            x = x + len_emb.unsqueeze(1)
        return x


class DualLanguageEmbedding(nn.Module):
    """双语嵌入层"""
    
    def __init__(self, vocab_size_zh: int, vocab_size_en: int, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.zh_embedding = LanguageEmbedding(vocab_size_zh, d_model, max_len, dropout)
        self.en_embedding = LanguageEmbedding(vocab_size_en, d_model, max_len, dropout)
    
    def forward(self, token_ids: torch.Tensor, lang: str, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lang == 'zh':
            return self.zh_embedding(token_ids, lengths)
        else:
            return self.en_embedding(token_ids, lengths)


class OutputProjection(nn.Module):
    """输出投影层"""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


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
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DualNoisePredictor(nn.Module):
    """双语言噪声预测器"""
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 4, d_ff: int = 512, max_len: int = 128, dropout: float = 0.1):
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
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, lang: str = "zh", mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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


class LanguageSwitcher(nn.Module):
    """语言切换分类器"""
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, x_t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x_t = x_t * mask.unsqueeze(-1)
        x = x_t.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x)
        return logits
    
    def predict(self, x_t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[str, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_t, mask)
            probs = F.softmax(logits, dim=-1)
            zh_prob = probs[0, 0].item()
            en_prob = probs[0, 1].item()
            if zh_prob > en_prob:
                return "zh", zh_prob
            else:
                return "en", en_prob


# ==================== 扩散过程 ====================
class Diffusion:
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.timesteps = config.timesteps
        
        # Beta schedule (linear)
        betas = torch.linspace(config.beta_start, config.beta_end, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        x_t = sqrt_alpha.view(-1, 1, 1) * x_0 + sqrt_one_minus_alpha.view(-1, 1, 1) * noise
        return x_t, noise
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        beta = self.betas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = 1.0 / torch.sqrt(self.alphas[t])
        
        # 去噪
        x_0_pred = sqrt_recip_alpha.view(-1, 1, 1) * (x_t - sqrt_one_minus_alpha.view(-1, 1, 1) * predicted_noise)
        
        # 添加噪声（除了最后一步）
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_0_pred + torch.sqrt(beta).view(-1, 1, 1) * noise
        else:
            x_prev = x_0_pred
        
        return x_prev


class DDIMSampler:
    def __init__(self, diffusion: Diffusion, ddim_steps: int = 50):
        self.diffusion = diffusion
        self.ddim_steps = ddim_steps
        
        # 选择均匀分布的时间步，从高到低（从噪声到干净）
        c = self.diffusion.timesteps // ddim_steps
        ddim_timesteps = [i * c for i in range(ddim_steps)]
        self.ddim_timesteps = torch.tensor(list(reversed(ddim_timesteps)))
    
    def ddim_step(self, x_t: torch.Tensor, t: int, t_prev: int, 
                  predicted_noise: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """DDIM单步"""
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # 预测 x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # 方差
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        
        # DDIM更新
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise
        
        if t_prev >= 0:
            noise = torch.randn_like(x_t)
            x_prev = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise
        else:
            x_prev = x_0_pred
        
        return x_prev


# ==================== 翻译器 ====================
class Translator:
    def __init__(self, model_dir: str = "."):
        self.device = torch.device("cpu")
        
        # 配置
        self.model_config = ModelConfig()
        self.diffusion_config = DiffusionConfig()
        
        # 加载分词器
        self.zh_tokenizer = Tokenizer.load(os.path.join(model_dir, "tokenizer_zh.json"))
        self.en_tokenizer = Tokenizer.load(os.path.join(model_dir, "tokenizer_en.json"))
        
        # 初始化模型
        self.embedding = DualLanguageEmbedding(
            vocab_size_zh=self.zh_tokenizer.vocab_size_actual,
            vocab_size_en=self.en_tokenizer.vocab_size_actual,
            d_model=self.model_config.d_model,
            max_len=self.model_config.max_len,
            dropout=0.0,
        )
        
        self.output_proj = DualOutputProjection(
            d_model=self.model_config.d_model,
            vocab_size_zh=self.zh_tokenizer.vocab_size_actual,
            vocab_size_en=self.en_tokenizer.vocab_size_actual,
        )
        
        self.model = DualNoisePredictor(
            d_model=self.model_config.d_model,
            n_heads=self.model_config.n_heads,
            n_layers=self.model_config.n_layers,
            d_ff=self.model_config.d_ff,
            max_len=self.model_config.max_len,
            dropout=0.0,
        )
        self.switcher = LanguageSwitcher(
            d_model=self.model_config.d_model,
            hidden_dim=self.model_config.d_model // 2,
            dropout=0.0,
        )
        
        self.diffusion = Diffusion(self.diffusion_config)
        
        # 加载权重
        self._load_checkpoint(os.path.join(model_dir, "best.pt"))
    
    def _load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.embedding.load_state_dict(state['embedding'])
        self.output_proj.load_state_dict(state['output_proj'])
        self.model.load_state_dict(state['model'])
        self.switcher.load_state_dict(state['switcher'])
        print(f"已加载模型: {path}")
    
    def _encode(self, text: str, lang: str) -> torch.Tensor:
        if lang == "zh":
            ids = self.zh_tokenizer.encode(text, add_sos=True, add_eos=True)
        else:
            ids = self.en_tokenizer.encode(text, add_sos=True, add_eos=True)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    def _decode(self, ids: torch.Tensor, lang: str) -> str:
        ids = ids[0].tolist()
        if lang == "zh":
            return self.zh_tokenizer.decode(ids, skip_special=True)
        else:
            return self.en_tokenizer.decode(ids, skip_special=True)
    
    def _embed_to_tokens(self, x: torch.Tensor, lang: str) -> torch.Tensor:
        logits = self.output_proj(x, lang)
        return logits.argmax(dim=-1)
    
    @torch.no_grad()
    def translate(
        self,
        text: str,
        source_lang: str,
        ddim_steps: int = 50,
        show_process: bool = False,
    ) -> Tuple[str, List[str]]:
        """翻译文本，返回结果和中间过程"""
        self.model.eval()
        self.embedding.eval()
        self.output_proj.eval()
        self.switcher.eval()
        
        target_lang = "en" if source_lang == "zh" else "zh"
        
        # 更新DDIM步数
        self.diffusion_config.ddim_steps = ddim_steps
        ddim_sampler = DDIMSampler(self.diffusion, ddim_steps)
        
        # 编码源语言
        source_ids = self._encode(text, source_lang)
        source_len = torch.tensor([source_ids.size(1)])
        
        # 嵌入源语言
        source_emb = self.embedding(source_ids, source_lang, source_len)
        
        # 前向扩散到纯噪声
        batch_size = source_emb.size(0)
        t_full = torch.full((batch_size,), self.diffusion_config.timesteps - 1, dtype=torch.long)
        noise = torch.randn_like(source_emb)
        x_t, _ = self.diffusion.q_sample(source_emb, t_full, noise)
        
        # DDIM反向扩散
        timesteps = ddim_sampler.ddim_timesteps
        total_steps = len(timesteps)
        switch_point = total_steps // 2
        
        process_steps = []
        
        for i, t in enumerate(timesteps[:-1]):
            t_prev = timesteps[i + 1]
            
            # 语言切换
            if i < switch_point:
                current_lang = source_lang
            else:
                current_lang = target_lang
            
            # 预测噪声
            t_tensor = torch.full((x_t.size(0),), t.item(), dtype=torch.long)
            predicted_noise = self.model(x_t, t_tensor, lang=current_lang)
            
            # 记录过程
            if show_process and i % max(1, total_steps // 10) == 0:
                current_ids = self._embed_to_tokens(x_t, current_lang)
                current_text = self._decode(current_ids, current_lang)
                process_steps.append(f"Step {t.item()}: {current_text[:50]}")
            
            # DDIM步骤
            x_t = ddim_sampler.ddim_step(x_t, t.item(), t_prev.item(), predicted_noise, eta=0.0)
        
        # 最终解码
        final_ids = self._embed_to_tokens(x_t, target_lang)
        result = self._decode(final_ids, target_lang)
        
        return result, process_steps


# ==================== Gradio 应用 ====================
def create_app():
    # 加载模型
    print("正在加载模型...")
    # 使用脚本所在目录作为模型目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    translator = Translator(model_dir=script_dir)
    print("模型加载完成!")
    
    def translate_text(text: str, language: str, ddim_steps: int, show_process: bool):
        if not text.strip():
            return "", []
        
        # 自动检测或手动选择
        if language == "自动检测":
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                source_lang = "zh"
            else:
                source_lang = "en"
        else:
            source_lang = "zh" if language == "中文 → 英文" else "en"
        
        try:
            result, process = translator.translate(
                text, source_lang, ddim_steps, show_process
            )
            process_text = "\n".join(process) if process else "（过程未显示）"
            return result, process_text
        except Exception as e:
            return f"翻译出错: {str(e)}", ""
    
    # 创建界面
    with gr.Blocks(
        title="Diffutslator",
        theme=gr.themes.Soft(),
        css="""
        .output-box { min-height: 100px; }
        .process-box { font-family: monospace; font-size: 12px; }
        """
    ) as app:
        gr.Markdown(
            """
            # Diffutslator 扩散翻译器
            
            基于扩散模型的机器翻译系统，可视化翻译过程中的语言渐变。
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="输入文本",
                    placeholder="输入要翻译的中文或英文...",
                    lines=5,
                )
                
                with gr.Row():
                    language = gr.Dropdown(
                        choices=["自动检测", "中文 → 英文", "英文 → 中文"],
                        value="自动检测",
                        label="翻译方向",
                    )
                    
                    ddim_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="DDIM步数",
                        info="步数越多质量越高，速度越慢",
                    )
                
                show_process = gr.Checkbox(
                    value=False,
                    label="显示扩散过程",
                    info="显示翻译中间步骤（会增加推理时间）",
                )
                
                translate_btn = gr.Button("翻译", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="翻译结果",
                    lines=5,
                    interactive=False,
                    elem_classes=["output-box"],
                )
                
                process_text = gr.Textbox(
                    label="扩散过程",
                    lines=5,
                    interactive=False,
                    visible=False,
                    elem_classes=["process-box"],
                )
        
        # 示例
        gr.Examples(
            examples=[
                ["你好，世界！", "自动检测"],
                ["Hello, how are you today?", "自动检测"],
                ["机器学习正在改变世界。", "中文 → 英文"],
                ["The quick brown fox jumps over the lazy dog.", "英文 → 中文"],
            ],
            inputs=[input_text, language],
        )
        
        # 事件处理
        def toggle_process(show):
            return gr.Textbox(visible=show)
        
        show_process.change(
            fn=toggle_process,
            inputs=[show_process],
            outputs=[process_text],
        )
        
        translate_btn.click(
            fn=translate_text,
            inputs=[input_text, language, ddim_steps, show_process],
            outputs=[output_text, process_text],
        )
        
        # 回车提交
        input_text.submit(
            fn=translate_text,
            inputs=[input_text, language, ddim_steps, show_process],
            outputs=[output_text, process_text],
        )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
