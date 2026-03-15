"""
数据集加载
支持tatoeba和cveto数据集
"""

import os
import sys
import random
import psutil
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer


def check_memory():
    """检查可用内存"""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    return available_gb


@dataclass
class TranslationPair:
    """翻译句对"""
    zh: str
    en: str


class TranslationDataset(Dataset):
    """翻译数据集 - 流式处理，内存友好"""
    
    def __init__(
        self,
        pairs: List[TranslationPair],
        zh_tokenizer: Tokenizer,
        en_tokenizer: Tokenizer,
        max_len: int = 128,
        cache_tokenized: bool = True,
    ):
        self.pairs = pairs
        self.zh_tokenizer = zh_tokenizer
        self.en_tokenizer = en_tokenizer
        self.max_len = max_len
        
        # 小缓存，只缓存最近访问的数据
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._cache_size = min(5000, len(pairs) // 10)  # 缓存10%或最多5000条
        
        print(f"  数据集: {len(pairs)} 条 (流式处理)")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 检查缓存
        if idx in self._cache:
            return self._cache[idx]
        
        # 处理数据
        pair = self.pairs[idx]
        zh_ids = self.zh_tokenizer.encode(pair.zh, add_sos=True, add_eos=True)[:self.max_len]
        en_ids = self.en_tokenizer.encode(pair.en, add_sos=True, add_eos=True)[:self.max_len]
        
        result = {
            'zh_ids': torch.tensor(zh_ids, dtype=torch.long),
            'en_ids': torch.tensor(en_ids, dtype=torch.long),
            'zh_len': len(zh_ids),
            'en_len': len(en_ids),
            'zh_text': pair.zh,
            'en_text': pair.en,
        }
        
        # 添加到缓存
        if len(self._cache) < self._cache_size:
            self._cache[idx] = result
        
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """批处理函数，动态padding"""
    zh_ids_list = [item['zh_ids'] for item in batch]
    en_ids_list = [item['en_ids'] for item in batch]
    
    # 找最大长度
    max_zh_len = max(len(ids) for ids in zh_ids_list)
    max_en_len = max(len(ids) for ids in en_ids_list)
    
    # Padding
    zh_padded = torch.zeros(len(batch), max_zh_len, dtype=torch.long)
    en_padded = torch.zeros(len(batch), max_en_len, dtype=torch.long)
    
    zh_lens = []
    en_lens = []
    
    for i, (zh_ids, en_ids) in enumerate(zip(zh_ids_list, en_ids_list)):
        zh_padded[i, :len(zh_ids)] = zh_ids
        en_padded[i, :len(en_ids)] = en_ids
        zh_lens.append(len(zh_ids))
        en_lens.append(len(en_ids))
    
    return {
        'zh_ids': zh_padded,
        'en_ids': en_padded,
        'zh_lens': torch.tensor(zh_lens, dtype=torch.long),
        'en_lens': torch.tensor(en_lens, dtype=torch.long),
        'zh_texts': [item['zh_text'] for item in batch],
        'en_texts': [item['en_text'] for item in batch],
    }


def load_tatoeba(path: str, max_samples: Optional[int] = None) -> List[TranslationPair]:
    """加载tatoeba数据集
    
    格式: 编号\t中文\t编号\t英文
    """
    pairs = []
    seen = set()
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            
            zh = parts[1].strip()
            en = parts[3].strip()
            
            # 去重
            key = (zh, en)
            if key in seen:
                continue
            seen.add(key)
            
            pairs.append(TranslationPair(zh=zh, en=en))
            
            if max_samples and len(pairs) >= max_samples:
                break
    
    return pairs


def load_cveto(zh_path: str, en_path: str, max_samples: Optional[int] = None) -> List[TranslationPair]:
    """加载cveto数据集
    
    两个文件，行号对应
    """
    pairs = []
    
    # 先统计总行数
    print("    统计文件行数...", end="", flush=True)
    with open(zh_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f" {total_lines:,} 行")
    
    print("    读取数据...", end="", flush=True)
    last_print = 0
    with open(zh_path, 'r', encoding='utf-8') as zh_f, \
         open(en_path, 'r', encoding='utf-8') as en_f:
        
        for i, (zh_line, en_line) in enumerate(zip(zh_f, en_f)):
            zh = zh_line.strip()
            en = en_line.strip()
            
            if zh and en:
                pairs.append(TranslationPair(zh=zh, en=en))
            
            # 每10万行打印一次进度
            if i - last_print >= 100000:
                print(f".{i//100000}", end="", flush=True)
                last_print = i
            
            if max_samples and len(pairs) >= max_samples:
                break
    
    print(f" 完成")
    return pairs


def load_all_data(config) -> Tuple[List[TranslationPair], List[TranslationPair], List[TranslationPair]]:
    """加载所有数据，返回训练集、验证集、测试集"""
    print("加载数据集...")
    
    # 加载tatoeba
    tatoeba_path = config.data.tatoeba_path
    if os.path.exists(tatoeba_path):
        print(f"  加载 tatoeba: {tatoeba_path}")
        tatoeba_pairs = load_tatoeba(tatoeba_path, max_samples=config.data.max_samples)
        print(f"    句对数: {len(tatoeba_pairs)}")
    else:
        tatoeba_pairs = []
        print(f"  警告: tatoeba路径不存在: {tatoeba_path}")
    
    # 合并所有数据
    all_pairs = tatoeba_pairs.copy()
    
    # 如果还需要更多数据，加载cveto
    if config.data.max_samples is None or len(all_pairs) < config.data.max_samples:
        cveto_zh_path = config.data.cveto_zh_path
        cveto_en_path = config.data.cveto_en_path
        
        if os.path.exists(cveto_zh_path) and os.path.exists(cveto_en_path):
            print(f"  加载 cveto...")
            remaining = None
            if config.data.max_samples:
                remaining = config.data.max_samples - len(all_pairs)
            
            cveto_pairs = load_cveto(cveto_zh_path, cveto_en_path, max_samples=remaining)
            print(f"    句对数: {len(cveto_pairs)}")
            all_pairs.extend(cveto_pairs)
    
    # 过滤长度
    print(f"  过滤数据...", end="", flush=True)
    filtered_pairs = []
    total = len(all_pairs)
    last_print = 0
    for i, pair in enumerate(all_pairs):
        zh_len = len(pair.zh)
        en_len = len(pair.en)
        if config.data.min_len <= zh_len <= config.data.max_len and \
           config.data.min_len <= en_len <= config.data.max_len:
            filtered_pairs.append(pair)
        
        # 每10万条打印进度
        if i - last_print >= 100000:
            progress = (i + 1) / total * 100
            print(f".{progress:.0f}%", end="", flush=True)
            last_print = i
    
    print(f" 完成")
    
    print(f"  过滤后句对数: {len(filtered_pairs)}")
    
    # 打乱并分割
    random.shuffle(filtered_pairs)
    n = len(filtered_pairs)
    
    # 80% 训练, 10% 验证, 10% 测试
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_pairs = filtered_pairs[:train_end]
    val_pairs = filtered_pairs[train_end:val_end]
    test_pairs = filtered_pairs[val_end:]
    
    print(f"  训练集: {len(train_pairs)}")
    print(f"  验证集: {len(val_pairs)}")
    print(f"  测试集: {len(test_pairs)}")
    
    return train_pairs, val_pairs, test_pairs


def create_dataloaders(
    train_pairs: List[TranslationPair],
    val_pairs: List[TranslationPair],
    zh_tokenizer: Tokenizer,
    en_tokenizer: Tokenizer,
    config,
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    train_dataset = TranslationDataset(
        train_pairs,
        zh_tokenizer,
        en_tokenizer,
        max_len=config.model.max_len,
    )
    
    val_dataset = TranslationDataset(
        val_pairs,
        zh_tokenizer,
        en_tokenizer,
        max_len=config.model.max_len,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # CPU环境不用多进程
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    
    return train_loader, val_loader
