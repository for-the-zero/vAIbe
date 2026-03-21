"""
推理脚本
可视化跨语言渐变扩散翻译过程
"""

import os
import argparse
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List

from config import Config
from tokenizer import Tokenizer
from embedding import DualLanguageEmbedding, DualOutputProjection
from model import create_model
from diffusion import get_cross_lingual_diffusion, CrossLingualDiffusion
from switcher import create_switcher


class Translator:
    """翻译器 - 跨语言渐变扩散"""
    
    def __init__(self, config: Config, checkpoint_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cpu")
        
        # 加载分词器
        cache_dir = os.path.join(config.project_dir, config.data.cache_dir)
        self.zh_tokenizer = Tokenizer.load(os.path.join(cache_dir, "tokenizer_zh.json"))
        self.en_tokenizer = Tokenizer.load(os.path.join(cache_dir, "tokenizer_en.json"))
        
        # 初始化模型组件
        self.embedding = DualLanguageEmbedding(
            vocab_size_zh=self.zh_tokenizer.vocab_size_actual,
            vocab_size_en=self.en_tokenizer.vocab_size_actual,
            d_model=config.model.d_model,
            max_len=config.model.max_len,
            dropout=0.0,  # 推理时不使用dropout
        )
        
        self.output_proj = DualOutputProjection(
            d_model=config.model.d_model,
            vocab_size_zh=self.zh_tokenizer.vocab_size_actual,
            vocab_size_en=self.en_tokenizer.vocab_size_actual,
        )
        
        self.model = create_model(config)
        self.switcher = create_switcher(config)
        
        self.cross_diffusion, self.ddim_sampler = get_cross_lingual_diffusion(config)
        
        # 加载权重
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, path: str):
        """加载检查点"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        self.embedding.load_state_dict(state['embedding'])
        self.output_proj.load_state_dict(state['output_proj'])
        self.model.load_state_dict(state['model'])
        self.switcher.load_state_dict(state['switcher'])
        
        print(f"已加载检查点: {path}")
    
    def _encode(self, text: str, lang: str) -> torch.Tensor:
        """编码文本"""
        if lang == "zh":
            ids = self.zh_tokenizer.encode(text, add_sos=True, add_eos=True)
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        else:
            ids = self.en_tokenizer.encode(text, add_sos=True, add_eos=True)
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    def _decode(self, ids: torch.Tensor, lang: str) -> str:
        """解码为文本"""
        ids = ids[0].tolist()
        if lang == "zh":
            return self.zh_tokenizer.decode(ids, skip_special=True)
        else:
            return self.en_tokenizer.decode(ids, skip_special=True)
    
    def _embed_to_tokens(self, x: torch.Tensor, lang: str) -> torch.Tensor:
        """从嵌入空间解码到token"""
        logits = self.output_proj(x, lang)
        ids = logits.argmax(dim=-1)
        return ids
    
    @torch.no_grad()
    def translate(
        self,
        text: str,
        source_lang: str,
        verbose: bool = True,
        ddim: bool = True,
    ) -> str:
        """翻译文本 - 跨语言渐变扩散
        
        Args:
            text: 输入文本
            source_lang: 源语言 "zh" 或 "en"
            verbose: 是否打印扩散过程
            ddim: 是否使用DDIM加速
        
        Returns:
            翻译结果
        """
        self.model.eval()
        self.embedding.eval()
        self.output_proj.eval()
        self.switcher.eval()
        
        target_lang = "en" if source_lang == "zh" else "zh"
        
        if verbose:
            print(f"\n翻译模式: {source_lang.upper()} → {target_lang.upper()}")
            print(f"输入: {text}")
            print(f"\n跨语言渐变扩散过程:")
        
        # 编码源语言
        source_ids = self._encode(text, source_lang)
        source_len = torch.tensor([source_ids.size(1)])
        
        # 嵌入源语言
        source_emb = self.embedding(source_ids, source_lang, source_len)
        
        # 生成目标语言的初始噪声（因为我们没有目标语言的真实嵌入）
        # 使用源语言嵌入作为起点，通过扩散过程渐变到目标语言
        target_emb = torch.randn_like(source_emb) * 0.5  # 轻微噪声初始化
        
        if verbose:
            print(f"  源语言嵌入 → 目标语言嵌入 渐变")
        
        # 使用DDIM进行跨语言反向扩散
        if ddim:
            result = self._cross_lingual_ddim_reverse(
                source_emb, source_lang, target_lang, verbose
            )
        else:
            result = self._cross_lingual_ddpm_reverse(
                source_emb, source_lang, target_lang, verbose
            )
        
        return result
    
    def _cross_lingual_ddim_reverse(
        self,
        source_emb: torch.Tensor,
        source_lang: str,
        target_lang: str,
        verbose: bool,
    ) -> str:
        """跨语言DDIM反向扩散
        
        从源语言嵌入开始，渐变到目标语言嵌入
        
        核心思路：
        1. 从 t=T 开始（此时嵌入接近源语言）
        2. 逐步去噪，同时应用跨语言插值
        3. 到 t=0 时，嵌入应该接近目标语言
        """
        ddim_steps = self.config.diffusion.ddim_steps
        timesteps = self.ddim_sampler.ddim_timesteps
        total_steps = len(timesteps)
        
        # 初始状态：添加噪声的源语言嵌入
        # 模拟前向扩散的最后一步
        batch_size = source_emb.size(0)
        t_start = torch.full((batch_size,), timesteps[0], dtype=torch.long)
        noise_start = torch.randn_like(source_emb)
        
        # 使用跨语言扩散的前向过程（但 target 用随机噪声模拟）
        # 这样可以保持与训练时一致的形式
        target_emb_fake = torch.randn_like(source_emb) * 0.3 + source_emb * 0.7
        x_t, _ = self.cross_diffusion.q_sample(source_emb, target_emb_fake, t_start, noise_start)
        
        for i, t in enumerate(timesteps[:-1]):
            t_prev = timesteps[i + 1]
            
            # 计算当前的渐变进度
            progress = i / total_steps
            
            # 根据进度决定使用哪种语言去噪
            # 前半段：源语言风格，后半段：目标语言风格
            if progress < 0.3:
                current_lang = source_lang
            else:
                current_lang = target_lang
            
            # 预测噪声
            t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long)
            predicted_noise = self.model(x_t, t_tensor, lang=current_lang)
            
            # 显示当前解码结果
            if verbose and i % max(1, total_steps // 10) == 0:
                # 显示当前语言的解码结果
                current_ids = self._embed_to_tokens(x_t, current_lang)
                current_text = self._decode(current_ids, current_lang)
                if len(current_text) > 50:
                    current_text = current_text[:50] + "..."
                
                print(f"  Step {t:4d} ({progress*100:5.1f}%) [{current_lang}] → {current_text}")
            
            # DDIM步骤
            x_t = self.ddim_sampler.ddim_step(x_t, t, t_prev, predicted_noise, eta=0.0)
        
        # 最终解码
        final_ids = self._embed_to_tokens(x_t, target_lang)
        result = self._decode(final_ids, target_lang)
        
        if verbose:
            print(f"\n输出: {result}")
        
        return result
    
    def _cross_lingual_ddpm_reverse(
        self,
        source_emb: torch.Tensor,
        source_lang: str,
        target_lang: str,
        verbose: bool,
    ) -> str:
        """跨语言DDPM反向扩散（标准方法，较慢）"""
        total_steps = self.config.diffusion.timesteps
        
        # 初始状态：添加噪声的源语言嵌入
        batch_size = source_emb.size(0)
        t_start = torch.full((batch_size,), total_steps - 1, dtype=torch.long)
        noise_start = torch.randn_like(source_emb)
        
        # 使用跨语言扩散的前向过程
        target_emb_fake = torch.randn_like(source_emb) * 0.3 + source_emb * 0.7
        x_t, _ = self.cross_diffusion.q_sample(source_emb, target_emb_fake, t_start, noise_start)
        
        for t in range(total_steps - 1, -1, -1):
            # 计算进度
            progress = 1 - t / total_steps
            
            # 根据时间步决定用哪种语言
            if progress < 0.3:
                current_lang = source_lang
            else:
                current_lang = target_lang
            
            t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(x_t, t_tensor, lang=current_lang)
            
            if verbose and t % (total_steps // 10) == 0:
                current_ids = self._embed_to_tokens(x_t, current_lang)
                current_text = self._decode(current_ids, current_lang)
                if len(current_text) > 50:
                    current_text = current_text[:50] + "..."
                print(f"  Step {t:4d} ({progress*100:5.1f}%) [{current_lang}] → {current_text}")
            
            # DDPM步骤
            x_t = self.cross_diffusion.p_sample(x_t, t_tensor, predicted_noise)
        
        # 解码
        final_ids = self._embed_to_tokens(x_t, target_lang)
        result = self._decode(final_ids, target_lang)
        
        if verbose:
            print(f"\n输出: {result}")
        
        return result
    
    def translate_with_reference(
        self,
        text: str,
        source_lang: str,
        reference: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
        """带参考翻译的翻译（用于评估）
        
        如果提供参考，可以展示更精确的渐变过程
        """
        target_lang = "en" if source_lang == "zh" else "zh"
        
        if verbose:
            print(f"\n翻译模式: {source_lang.upper()} → {target_lang.upper()}")
            print(f"输入: {text}")
            if reference:
                print(f"参考: {reference}")
            print(f"\n跨语言渐变扩散过程:")
        
        # 编码源语言
        source_ids = self._encode(text, source_lang)
        source_len = torch.tensor([source_ids.size(1)])
        source_emb = self.embedding(source_ids, source_lang, source_len)
        
        # 如果有参考，编码参考
        if reference:
            target_ids = self._encode(reference, target_lang)
            target_len = torch.tensor([target_ids.size(1)])
            target_emb = self.embedding(target_ids, target_lang, target_len)
        else:
            target_emb = None
        
        # 执行扩散
        return self._cross_lingual_ddim_reverse(
            source_emb, source_lang, target_lang, verbose
        )
    
    def interactive(self):
        """交互模式"""
        print("\n" + "=" * 50)
        print("Diffutslator 交互翻译模式")
        print("跨语言渐变扩散翻译")
        print("=" * 50)
        print("输入 'zh: 文本' 翻译中文到英文")
        print("输入 'en: text' 翻译英文到中文")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 50 + "\n")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break
                
                if not user_input:
                    continue
                
                # 解析输入
                if user_input.lower().startswith('zh:'):
                    text = user_input[3:].strip()
                    source_lang = "zh"
                elif user_input.lower().startswith('en:'):
                    text = user_input[3:].strip()
                    source_lang = "en"
                else:
                    # 自动检测（简单判断）
                    if any('\u4e00' <= c <= '\u9fff' for c in user_input):
                        text = user_input
                        source_lang = "zh"
                    else:
                        text = user_input
                        source_lang = "en"
                
                # 翻译
                result = self.translate(text, source_lang, verbose=True)
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diffutslator 推理脚本")
    
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    parser.add_argument("--text", type=str, default=None, help="要翻译的文本")
    parser.add_argument("--zh", action="store_true", help="输入是中文")
    parser.add_argument("--en", action="store_true", help="输入是英文")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--quiet", "-q", action="store_true", help="安静模式，不打印过程")
    parser.add_argument("--ddim-steps", type=int, default=50, help="DDIM步数")
    
    args = parser.parse_args()
    
    # 配置
    config = Config()
    config.diffusion.ddim_steps = args.ddim_steps
    
    # 找检查点
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_dir = os.path.join(config.project_dir, config.training.checkpoint_dir)
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            checkpoint_path = best_path
        else:
            # 找最新的检查点
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    
    if checkpoint_path is None:
        print("错误: 未找到检查点，请先训练模型")
        return
    
    # 创建翻译器
    translator = Translator(config, checkpoint_path)
    
    # 模式
    if args.interactive:
        translator.interactive()
    elif args.text:
        if args.zh:
            source_lang = "zh"
        elif args.en:
            source_lang = "en"
        else:
            # 自动检测
            if any('\u4e00' <= c <= '\u9fff' for c in args.text):
                source_lang = "zh"
            else:
                source_lang = "en"
        
        result = translator.translate(args.text, source_lang, verbose=not args.quiet)
        if args.quiet:
            print(result)
    else:
        # 默认交互模式
        translator.interactive()


if __name__ == "__main__":
    main()
