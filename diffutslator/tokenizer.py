"""
分词器
支持中文字符级和BPE
"""

import os
import re
import json
import pickle
from typing import List, Dict, Optional, Tuple
from collections import Counter
from functools import lru_cache


class Tokenizer:
    """基础分词器"""
    
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
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE合并规则
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
    
    def _is_chinese(self, char: str) -> bool:
        """判断是否为中文字符"""
        return '\u4e00' <= char <= '\u9fff'
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """预分词"""
        if self.lang == "zh":
            # 中文：字符级 + 保留英文单词和数字
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
            # 英文：单词级
            text = text.lower()
            tokens = re.findall(r"\w+|[^\w\s]", text)
            return tokens
    
    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """获取词中的所有相邻字符对"""
        pairs = set()
        prev = word[0]
        for char in word[1:]:
            pairs.add((prev, char))
            prev = char
        return pairs
    
    def train_bpe(self, texts: List[str], num_merges: Optional[int] = None):
        """训练BPE"""
        if num_merges is None:
            num_merges = self.vocab_size - len(self.special_tokens) - 100
        
        # 统计词频
        print(f"    统计词频 ({len(texts)} 文本)...", end="", flush=True)
        word_freqs: Counter = Counter()
        for text in texts:
            for token in self._pre_tokenize(text):
                # 将token拆分为字符序列
                chars = tuple(token) + ('</w>',)
                word_freqs[chars] += 1
        print(f" {len(word_freqs)} 词")
        
        # BPE合并
        print(f"    BPE合并 ({num_merges} 轮)...", end="", flush=True)
        self.merges = []
        last_print = 0
        for i in range(num_merges):
            # 统计相邻字符对频率
            pairs: Counter = Counter()
            for word, freq in word_freqs.items():
                pairs_in_word = self._get_pairs(word)
                for pair in pairs_in_word:
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # 找最高频的pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # 合并所有词中的该pair
            new_word_freqs: Counter = Counter()
            bigram = re.escape(' '.join(best_pair))
            pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            
            for word, freq in word_freqs.items():
                new_word = ' '.join(word)
                new_word = pattern.sub(''.join(best_pair), new_word)
                new_word = tuple(new_word.split())
                new_word_freqs[new_word] += freq
            
            word_freqs = new_word_freqs
            
            # 每1000轮打印进度
            if i - last_print >= 100:
                print(f".{(i+1)//100}k", end="", flush=True)
                last_print = i
        
        print(f" 完成")
        
        # 构建词表
        self._build_vocab(word_freqs)
    
    def _build_vocab(self, word_freqs: Counter):
        """构建词表"""
        # 特殊token
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # 收集所有token
        vocab = set()
        for word in word_freqs.keys():
            for token in word:
                if token != '</w>':
                    vocab.add(token)
        
        # 添加合并后的token
        for pair in self.merges:
            vocab.add(''.join(pair))
        
        # 按频率排序并截断
        sorted_vocab = sorted(vocab)
        for i, token in enumerate(sorted_vocab[:self.vocab_size - len(self.special_tokens)]):
            idx = i + len(self.special_tokens)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
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
        
        # 移除</w>标记
        return [t.replace('</w>', '') for t in word if t.replace('</w>', '')]
    
    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """编码文本为token id序列"""
        # 缓存检查
        cache_key = (text, add_sos, add_eos)
        if hasattr(self, '_encode_cache') and cache_key in self._encode_cache:
            return self._encode_cache[cache_key]
        
        tokens = self._pre_tokenize(text)
        
        ids = []
        if add_sos:
            ids.append(self.token_to_id[self.sos_token])
        
        for token in tokens:
            bpe_tokens = self._apply_bpe(token)
            for t in bpe_tokens:
                ids.append(self.token_to_id.get(t, self.token_to_id[self.unk_token]))
        
        if add_eos:
            ids.append(self.token_to_id[self.eos_token])
        
        # 缓存结果（限制缓存大小）
        if not hasattr(self, '_encode_cache'):
            self._encode_cache = {}
        if len(self._encode_cache) < 100000:  # 最多缓存10万条
            self._encode_cache[cache_key] = ids
        
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """解码token id序列为文本"""
        tokens = []
        for id in ids:
            token = self.id_to_token.get(id, self.unk_token)
            if skip_special and token in self.special_tokens:
                continue
            # 移除BPE的</w>标记
            token = token.replace('</w>', '')
            if token:  # 跳过空token
                tokens.append(token)
        
        if self.lang == "en":
            # 英文：BPE子词之间用空格连接，然后清理多余空格
            text = ' '.join(tokens)
            # 标点符号前移除空格
            text = re.sub(r'\s+([.,!?;:\'\"])', r'\1', text)
            # 标点符号后添加空格（如果后面有字母）
            text = re.sub(r'([.,!?;:])([a-zA-Z])', r'\1 \2', text)
            # 清理多余空格
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            # 中文：直接拼接
            text = ''.join(tokens)
        
        return text
    
    @property
    def vocab_size_actual(self) -> int:
        """实际词表大小"""
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
    
    def save(self, path: str):
        """保存分词器"""
        data = {
            'vocab_size': self.vocab_size,
            'lang': self.lang,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'merges': self.merges,
            'special_tokens': self.special_tokens,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """加载分词器"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'], lang=data['lang'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.bpe_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}
        tokenizer.special_tokens = data['special_tokens']
        
        return tokenizer
    
    def __len__(self) -> int:
        return self.vocab_size_actual


def train_tokenizers(config, zh_texts: List[str], en_texts: List[str]) -> Tuple[Tokenizer, Tokenizer]:
    """训练中英文分词器"""
    print("训练中文分词器...")
    zh_tokenizer = Tokenizer(vocab_size=config.model.vocab_size_zh, lang="zh")
    zh_tokenizer.train_bpe(zh_texts)
    zh_tokenizer.bpe_ranks = {pair: i for i, pair in enumerate(zh_tokenizer.merges)}
    
    print("训练英文分词器...")
    en_tokenizer = Tokenizer(vocab_size=config.model.vocab_size_en, lang="en")
    en_tokenizer.train_bpe(en_texts)
    en_tokenizer.bpe_ranks = {pair: i for i, pair in enumerate(en_tokenizer.merges)}
    
    print(f"中文词表大小: {zh_tokenizer.vocab_size_actual}")
    print(f"英文词表大小: {en_tokenizer.vocab_size_actual}")
    
    return zh_tokenizer, en_tokenizer
