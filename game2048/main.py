"""
2048 AI Trainer - 主入口
基于Transformer的2048游戏AI训练器

使用方法:
    python main.py              # 启动GUI界面
    python main.py --train      # 命令行训练模式
    python main.py --demo       # 演示模式（加载已有模型）
"""
import sys
import os
import argparse
import torch
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import Game2048
from model import Game2048Transformer, count_parameters
from trainer import PPOTrainer, RolloutBuffer, TrainingStats
from parallel import TrainingLoop, ParallelGameEnv, TrainingWorker
from utils import (
    set_seed, get_device, print_model_info, save_checkpoint,
    load_checkpoint, EarlyStopping, format_time, format_number
)


def run_gui():
    """运行GUI界面"""
    from gui import main
    main()


def run_training(
    num_games: int = 10000,
    num_envs: int = 4,
    save_interval: int = 100,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42
):
    """
    命令行训练模式
    
    Args:
        num_games: 总游戏局数
        num_envs: 并行环境数
        save_interval: 保存间隔
        checkpoint_dir: 检查点目录
        seed: 随机种子
    """
    print("=" * 50)
    print("2048 AI Training")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(seed)
    
    # 获取设备
    device = get_device()
    print(f"Device: {device}")
    
    # 创建模型
    model = Game2048Transformer()
    print_model_info(model)
    model.to(device)
    
    # 创建训练器
    trainer = PPOTrainer(model, lr=1e-4, device=device)
    
    # 创建训练循环
    training_loop = TrainingLoop(
        model=model,
        trainer=trainer,
        num_envs=num_envs,
        device=device,
        steps_per_update=256,
        checkpoint_dir=checkpoint_dir
    )
    
    # 训练统计
    stats = TrainingStats()
    start_time = time.time()
    best_score = 0
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nStarting training for {num_games} games...")
    print(f"Parallel environments: {num_envs}")
    print("-" * 50)
    
    try:
        games_completed = 0
        
        def on_game_end(game_stats):
            nonlocal games_completed, best_score
            
            stats.record_game(
                score=game_stats['score'],
                situational_score=game_stats['situational_score'],
                max_tile=game_stats['max_tile'],
                steps=game_stats['moves']
            )
            
            games_completed += 1
            
            if game_stats['score'] > best_score:
                best_score = game_stats['score']
            
            # 定期打印统计
            if games_completed % 10 == 0:
                elapsed = time.time() - start_time
                avg_stats = stats.get_avg_stats(window=100)
                
                print(
                    f"Games: {games_completed} | "
                    f"Avg Score: {avg_stats['avg_score']:.0f} | "
                    f"Best: {best_score} | "
                    f"Max Tile: {avg_stats['avg_max_tile']:.0f} | "
                    f"Speed: {games_completed/elapsed:.2f} games/s"
                )
            
            # 保存检查点
            if games_completed % save_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"checkpoint_{games_completed}.pt"
                )
                save_checkpoint(
                    model, trainer.optimizer, games_completed,
                    avg_stats, checkpoint_path
                )
                print(f"Checkpoint saved: {checkpoint_path}")
        
        training_loop.on_game_end_callback = on_game_end
        training_loop.train(total_games=num_games, stop_threshold=200)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # 训练结束统计
    elapsed = time.time() - start_time
    final_stats = stats.get_avg_stats()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total games: {format_number(final_stats['games_played'])}")
    print(f"Total time: {format_time(elapsed)}")
    print(f"Average score: {final_stats['avg_score']:.0f}")
    print(f"Best score: {final_stats['best_score']}")
    print(f"Best max tile: {final_stats['best_max_tile']}")
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    save_checkpoint(
        model, trainer.optimizer, final_stats['games_played'],
        final_stats, final_path
    )
    print(f"Final model saved: {final_path}")


def run_demo(model_path: str = None, num_games: int = 5):
    """
    演示模式
    
    Args:
        model_path: 模型路径
        num_games: 演示游戏数
    """
    print("=" * 50)
    print("2048 AI Demo Mode")
    print("=" * 50)
    
    device = get_device()
    print(f"Device: {device}")
    
    # 创建模型
    model = Game2048Transformer()
    
    # 加载模型
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        checkpoint = load_checkpoint(model_path, model, device=device)
        print(f"Loaded checkpoint: {checkpoint.get('epoch', 'unknown')} games")
    else:
        print("No model loaded, using random weights.")
    
    model.to(device)
    model.eval()
    print_model_info(model)
    
    # 运行演示游戏
    print(f"\nRunning {num_games} demo games...")
    print("-" * 50)
    
    total_scores = []
    
    for game_idx in range(num_games):
        game = Game2048()
        game.reset()
        
        steps = 0
        while not game.game_over and steps < 10000:
            # 获取状态
            state = game.get_state()
            scores = game.get_state_with_scores()[-2:]
            valid_actions = game.get_valid_actions()
            
            # 转换为张量
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            scores_t = torch.FloatTensor(scores).unsqueeze(0).to(device)
            valid_t = torch.BoolTensor(valid_actions).unsqueeze(0).to(device)
            
            # 获取动作
            with torch.no_grad():
                action, _, _ = model.get_action(state_t, scores_t, valid_t, deterministic=True)
            
            # 执行动作
            game.move(action)
            steps += 1
        
        total_scores.append(game.accumulated_score)
        
        print(
            f"Game {game_idx + 1}: "
            f"Score = {game.accumulated_score}, "
            f"Max Tile = {game.get_max_tile()}, "
            f"Steps = {steps}"
        )
    
    # 统计
    print("\n" + "-" * 50)
    print(f"Average score: {sum(total_scores) / len(total_scores):.0f}")
    print(f"Best score: {max(total_scores)}")
    print(f"Worst score: {min(total_scores)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='2048 AI Trainer - 基于Transformer的2048游戏AI'
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='运行命令行训练模式'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行演示模式'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型路径（用于演示模式或继续训练）'
    )
    
    parser.add_argument(
        '--games',
        type=int,
        default=10000,
        help='训练游戏数（默认10000）'
    )
    
    parser.add_argument(
        '--envs',
        type=int,
        default=4,
        help='并行环境数（默认4）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    
    args = parser.parse_args()
    
    if args.train:
        run_training(
            num_games=args.games,
            num_envs=args.envs,
            seed=args.seed
        )
    elif args.demo:
        run_demo(model_path=args.model, num_games=5)
    else:
        # 默认启动GUI
        run_gui()


if __name__ == "__main__":
    main()
