"""
导出模型为JSON格式，用于WebGPU推理
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

from config import Config
from tokenizer import Tokenizer
from embedding import DualLanguageEmbedding, DualOutputProjection
from model import create_model
from diffusion import get_diffusion


def tensor_to_list(t) -> list:
    """将tensor转换为list"""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().tolist()
    return t


def export_model(config: Config, checkpoint_path: str, output_dir: str):
    """导出模型为JSON格式"""
    
    print(f"加载检查点: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 加载分词器
    cache_dir = os.path.join(config.project_dir, config.data.cache_dir)
    zh_tokenizer = Tokenizer.load(os.path.join(cache_dir, "tokenizer_zh.json"))
    en_tokenizer = Tokenizer.load(os.path.join(cache_dir, "tokenizer_en.json"))
    
    # 创建模型
    embedding = DualLanguageEmbedding(
        vocab_size_zh=zh_tokenizer.vocab_size_actual,
        vocab_size_en=en_tokenizer.vocab_size_actual,
        d_model=config.model.d_model,
        max_len=config.model.max_len,
        dropout=0.0,
    )
    
    output_proj = DualOutputProjection(
        d_model=config.model.d_model,
        vocab_size_zh=zh_tokenizer.vocab_size_actual,
        vocab_size_en=en_tokenizer.vocab_size_actual,
    )
    
    model = create_model(config)
    
    # 加载权重
    embedding.load_state_dict(state['embedding'])
    output_proj.load_state_dict(state['output_proj'])
    model.load_state_dict(state['model'])
    
    embedding.eval()
    output_proj.eval()
    model.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出扩散参数
    diffusion, ddim_sampler = get_diffusion(config)
    scheduler = diffusion.scheduler
    
    diffusion_params = {
        'timesteps': config.diffusion.timesteps,
        'ddim_steps': config.diffusion.ddim_steps,
        'betas': tensor_to_list(scheduler.betas),
        'alphas': tensor_to_list(scheduler.alphas),
        'alphas_cumprod': tensor_to_list(scheduler.alphas_cumprod),
        'sqrt_alphas_cumprod': tensor_to_list(scheduler.sqrt_alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': tensor_to_list(scheduler.sqrt_one_minus_alphas_cumprod),
        'ddim_timesteps': ddim_sampler.ddim_timesteps,
    }
    
    with open(os.path.join(output_dir, 'diffusion_params.json'), 'w') as f:
        json.dump(diffusion_params, f)
    
    print("导出扩散参数完成")
    
    # 导出分词器
    zh_vocab = {
        'token_to_id': zh_tokenizer.token_to_id,
        'id_to_token': {str(k): v for k, v in zh_tokenizer.id_to_token.items()},
        'merges': zh_tokenizer.merges,
        'special_tokens': zh_tokenizer.special_tokens,
        'lang': 'zh',
    }
    
    en_vocab = {
        'token_to_id': en_tokenizer.token_to_id,
        'id_to_token': {str(k): v for k, v in en_tokenizer.id_to_token.items()},
        'merges': en_tokenizer.merges,
        'special_tokens': en_tokenizer.special_tokens,
        'lang': 'en',
    }
    
    with open(os.path.join(output_dir, 'tokenizer_zh.json'), 'w', encoding='utf-8') as f:
        json.dump(zh_vocab, f, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'tokenizer_en.json'), 'w', encoding='utf-8') as f:
        json.dump(en_vocab, f, ensure_ascii=False)
    
    print("导出分词器完成")
    
    # 导出嵌入层权重为JSON
    def extract_embedding_weights(lang_emb):
        """提取嵌入层权重"""
        return {
            'token_embedding': tensor_to_list(lang_emb.token_embedding.weight),
            'position_encoding': tensor_to_list(lang_emb.position_encoding.pe),
            'length_embedding': tensor_to_list(lang_emb.length_embedding.weight),
            'scale': lang_emb.scale,
        }
    
    embedding_weights = {
        'zh': extract_embedding_weights(embedding.zh_embedding),
        'en': extract_embedding_weights(embedding.en_embedding),
    }
    
    with open(os.path.join(output_dir, 'embedding.json'), 'w') as f:
        json.dump(embedding_weights, f)
    
    print("导出嵌入层完成")
    
    # 导出输出投影权重
    output_weights = {
        'zh_projection': tensor_to_list(output_proj.zh_projection.projection.weight),
        'en_projection': tensor_to_list(output_proj.en_projection.projection.weight),
    }
    
    with open(os.path.join(output_dir, 'output_proj.json'), 'w') as f:
        json.dump(output_weights, f)
    
    print("导出输出投影完成")
    
    # 导出噪声预测模型权重
    def extract_model_weights(model):
        """提取模型权重"""
        weights = {}
        
        # 时间嵌入
        weights['time_mlp'] = {
            '0.weight': tensor_to_list(model.time_mlp[0].weight),
            '0.bias': tensor_to_list(model.time_mlp[0].bias),
            '2.weight': tensor_to_list(model.time_mlp[2].weight),
            '2.bias': tensor_to_list(model.time_mlp[2].bias),
        }
        
        # 语言特定投影
        weights['zh_input_proj'] = {
            'weight': tensor_to_list(model.zh_input_proj.weight),
            'bias': tensor_to_list(model.zh_input_proj.bias),
        }
        weights['en_input_proj'] = {
            'weight': tensor_to_list(model.en_input_proj.weight),
            'bias': tensor_to_list(model.en_input_proj.bias),
        }
        weights['zh_output_proj'] = {
            'weight': tensor_to_list(model.zh_output_proj.weight),
            'bias': tensor_to_list(model.zh_output_proj.bias),
        }
        weights['en_output_proj'] = {
            'weight': tensor_to_list(model.en_output_proj.weight),
            'bias': tensor_to_list(model.en_output_proj.bias),
        }
        
        # 输出归一化
        weights['output_norm'] = {
            'weight': tensor_to_list(model.output_norm.weight),
            'bias': tensor_to_list(model.output_norm.bias),
        }
        
        # Transformer层
        weights['layers'] = []
        for i, layer in enumerate(model.layers):
            layer_weights = {
                # 自注意力
                'w_q.weight': tensor_to_list(layer.attn.w_q.weight),
                'w_q.bias': tensor_to_list(layer.attn.w_q.bias),
                'w_k.weight': tensor_to_list(layer.attn.w_k.weight),
                'w_k.bias': tensor_to_list(layer.attn.w_k.bias),
                'w_v.weight': tensor_to_list(layer.attn.w_v.weight),
                'w_v.bias': tensor_to_list(layer.attn.w_v.bias),
                'w_o.weight': tensor_to_list(layer.attn.w_o.weight),
                'w_o.bias': tensor_to_list(layer.attn.w_o.bias),
                # 前馈网络
                'w1.weight': tensor_to_list(layer.ff.w1.weight),
                'w1.bias': tensor_to_list(layer.ff.w1.bias),
                'w2.weight': tensor_to_list(layer.ff.w2.weight),
                'w2.bias': tensor_to_list(layer.ff.w2.bias),
                # LayerNorm
                'norm1.weight': tensor_to_list(layer.norm1.weight),
                'norm1.bias': tensor_to_list(layer.norm1.bias),
                'norm2.weight': tensor_to_list(layer.norm2.weight),
                'norm2.bias': tensor_to_list(layer.norm2.bias),
            }
            weights['layers'].append(layer_weights)
        
        return weights
    
    model_weights = extract_model_weights(model)
    
    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(model_weights, f)
    
    print("导出模型权重完成")
    
    # 导出配置
    config_dict = {
        'd_model': config.model.d_model,
        'n_heads': config.model.n_heads,
        'n_layers': config.model.n_layers,
        'd_ff': config.model.d_ff,
        'max_len': config.model.max_len,
        'vocab_size_zh': zh_tokenizer.vocab_size_actual,
        'vocab_size_en': en_tokenizer.vocab_size_actual,
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f)
    
    print(f"\n导出完成! 文件保存在: {output_dir}")
    print("文件列表:")
    for f in os.listdir(output_dir):
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出模型为JSON格式")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="检查点路径")
    parser.add_argument("--output", type=str, default="web/models", help="输出目录")
    
    args = parser.parse_args()
    
    config = Config()
    export_model(config, args.checkpoint, args.output)
