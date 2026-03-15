"""
导出模型为ONNX格式，用于WebGPU推理
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from typing import Dict, Any

from config import Config
from tokenizer import Tokenizer
from embedding import DualLanguageEmbedding, DualOutputProjection
from model import create_model
from diffusion import get_diffusion


def export_model(config: Config, checkpoint_path: str, output_dir: str):
    """导出模型为ONNX格式"""
    
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
        'betas': scheduler.betas.tolist(),
        'alphas': scheduler.alphas.tolist(),
        'alphas_cumprod': scheduler.alphas_cumprod.tolist(),
        'sqrt_alphas_cumprod': scheduler.sqrt_alphas_cumprod.tolist(),
        'sqrt_one_minus_alphas_cumprod': scheduler.sqrt_one_minus_alphas_cumprod.tolist(),
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
    
    # 导出嵌入层权重
    torch.save({
        'zh_embedding': embedding.zh_embedding.state_dict(),
        'en_embedding': embedding.en_embedding.state_dict(),
        'zh_projection': output_proj.zh_projection.state_dict(),
        'en_projection': output_proj.en_projection.state_dict(),
    }, os.path.join(output_dir, 'embedding.pt'))
    
    print("导出嵌入层完成")
    
    # 导出噪声预测模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    print("导出模型完成")
    
    # 尝试导出ONNX（可选）
    try:
        export_onnx(model, embedding, output_proj, config, output_dir)
        print("导出ONNX完成")
    except Exception as e:
        print(f"ONNX导出跳过: {e}")
    
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


def export_onnx(model, embedding, output_proj, config, output_dir):
    """导出ONNX格式"""
    
    batch_size = 1
    seq_len = 32
    d_model = config.model.d_model
    
    # 导出中文嵌入
    class ZHEmbeddingWrapper(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb
        
        def forward(self, token_ids):
            return self.emb(token_ids)
    
    zh_emb_wrapper = ZHEmbeddingWrapper(embedding.zh_embedding)
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
    
    torch.onnx.export(
        zh_emb_wrapper,
        dummy_input,
        os.path.join(output_dir, 'zh_embedding.onnx'),
        input_names=['token_ids'],
        output_names=['embedding'],
        dynamic_axes={'token_ids': {0: 'batch', 1: 'seq'}, 'embedding': {0: 'batch', 1: 'seq'}},
        opset_version=14,
    )
    
    # 导出英文嵌入
    class ENEmbeddingWrapper(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb
        
        def forward(self, token_ids):
            return self.emb(token_ids)
    
    en_emb_wrapper = ENEmbeddingWrapper(embedding.en_embedding)
    
    torch.onnx.export(
        en_emb_wrapper,
        dummy_input,
        os.path.join(output_dir, 'en_embedding.onnx'),
        input_names=['token_ids'],
        output_names=['embedding'],
        dynamic_axes={'token_ids': {0: 'batch', 1: 'seq'}, 'embedding': {0: 'batch', 1: 'seq'}},
        opset_version=14,
    )
    
    # 导出噪声预测模型
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x_t, t):
            return self.model(x_t, t, lang='zh')  # 默认中文
    
    model_wrapper = ModelWrapper(model)
    
    dummy_x = torch.randn(batch_size, seq_len, d_model)
    dummy_t = torch.zeros(batch_size, dtype=torch.long)
    
    torch.onnx.export(
        model_wrapper,
        (dummy_x, dummy_t),
        os.path.join(output_dir, 'model.onnx'),
        input_names=['x_t', 't'],
        output_names=['noise_pred'],
        dynamic_axes={'x_t': {0: 'batch', 1: 'seq'}, 't': {0: 'batch'}, 'noise_pred': {0: 'batch', 1: 'seq'}},
        opset_version=14,
    )
    
    # 导出输出投影
    class OutputWrapper(nn.Module):
        def __init__(self, proj):
            super().__init__()
            self.proj = proj
        
        def forward(self, x):
            return self.proj(x)
    
    zh_out_wrapper = OutputWrapper(output_proj.zh_projection)
    
    torch.onnx.export(
        zh_out_wrapper,
        dummy_x,
        os.path.join(output_dir, 'zh_output.onnx'),
        input_names=['hidden'],
        output_names=['logits'],
        dynamic_axes={'hidden': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch', 1: 'seq'}},
        opset_version=14,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出模型为ONNX格式")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="检查点路径")
    parser.add_argument("--output", type=str, default="web/models", help="输出目录")
    
    args = parser.parse_args()
    
    config = Config()
    export_model(config, args.checkpoint, args.output)
