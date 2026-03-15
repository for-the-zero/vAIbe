"""
Diffutslator 主入口
基于扩散模型的中英互译系统
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Diffutslator - 基于扩散模型的翻译系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速验证训练
  python main.py train --quick
  
  # 完整训练
  python main.py train --full
  
  # 从检查点恢复训练
  python main.py train --resume checkpoints/epoch_5.pt
  
  # 交互式翻译
  python main.py translate
  
  # 翻译单个句子
  python main.py translate --text "你好世界" --zh
  
  # 使用更多DDIM步数
  python main.py translate --text "Hello world" --en --ddim-steps 100
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--quick", action="store_true", help="快速验证模式")
    train_parser.add_argument("--full", action="store_true", help="完整训练模式")
    train_parser.add_argument("--samples", type=int, default=None, help="使用的数据量")
    train_parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=None, help="批量大小")
    train_parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点")
    
    # 翻译命令
    translate_parser = subparsers.add_parser("translate", help="翻译文本")
    translate_parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    translate_parser.add_argument("--text", type=str, default=None, help="要翻译的文本")
    translate_parser.add_argument("--zh", action="store_true", help="输入是中文")
    translate_parser.add_argument("--en", action="store_true", help="输入是英文")
    translate_parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    translate_parser.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    translate_parser.add_argument("--ddim-steps", type=int, default=50, help="DDIM步数")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # 导入并运行训练
        from train import main as train_main
        sys.argv = ["train.py"]
        
        if args.quick:
            sys.argv.append("--quick")
        if args.full:
            sys.argv.append("--full")
        if args.samples:
            sys.argv.extend(["--samples", str(args.samples)])
        if args.epochs:
            sys.argv.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(["--batch-size", str(args.batch_size)])
        if args.resume:
            sys.argv.extend(["--resume", args.resume])
        
        train_main()
    
    elif args.command == "translate":
        # 导入并运行推理
        from inference import main as inference_main
        sys.argv = ["inference.py"]
        
        if args.checkpoint:
            sys.argv.extend(["--checkpoint", args.checkpoint])
        if args.text:
            sys.argv.extend(["--text", args.text])
        if args.zh:
            sys.argv.append("--zh")
        if args.en:
            sys.argv.append("--en")
        if args.interactive:
            sys.argv.append("--interactive")
        if args.quiet:
            sys.argv.append("--quiet")
        if args.ddim_steps:
            sys.argv.extend(["--ddim-steps", str(args.ddim_steps)])
        
        inference_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
