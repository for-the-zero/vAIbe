"""
简单训练脚本 - 直接训练并保存模型
"""
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import Game2048
from model import Game2048Transformer
from trainer import PPOTrainer, RolloutBuffer

def train_simple(
    num_games: int = 1000,
    save_path: str = "checkpoints/model.pt",
    print_interval: int = 10
):
    """简单训练"""
    print("=" * 50)
    print("2048 AI Simple Training")
    print("=" * 50)
    
    device = "cpu"
    model = Game2048Transformer().to(device)
    trainer = PPOTrainer(model, lr=3e-4, device=device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_games} games...")
    print("-" * 50)
    
    # 统计
    scores = []
    max_tiles = []
    best_score = 0
    
    start_time = time.time()
    
    for game_idx in range(num_games):
        # 运行一局游戏
        game = Game2048()
        game.reset()
        
        buffer = RolloutBuffer(capacity=10000)
        
        while not game.game_over:
            state = game.get_state()
            scores_feat = np.array([
                min(game.accumulated_score / 50000, 1.0),
                min(game.situational_score / 200, 1.0)
            ], dtype=np.float32)
            valid = game.get_valid_actions()
            
            # 转换张量
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            scores_t = torch.FloatTensor(scores_feat).unsqueeze(0).to(device)
            valid_t = torch.BoolTensor(valid).unsqueeze(0).to(device)
            
            # 选择动作
            with torch.no_grad():
                action, log_prob, value = model.get_action(state_t, scores_t, valid_t, deterministic=False)
            
            # 执行动作
            old_state = state.copy()
            old_scores = scores_feat.copy()
            
            next_state, reward, moved, done = game.move(action)
            
            # 存储转移
            from trainer import Transition
            transition = Transition(
                state=old_state,
                scores=old_scores,
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                next_scores=np.array([
                    min(game.accumulated_score / 50000, 1.0),
                    min(game.situational_score / 200, 1.0)
                ], dtype=np.float32),
                done=done,
                log_prob=log_prob,
                value=value,
                valid_actions=valid
            )
            buffer.push(transition)
            
            # 每步更新
            if len(buffer) >= 64:
                trainer.update(buffer)
                buffer.clear()
        
        # 记录结果
        scores.append(game.accumulated_score)
        max_tiles.append(game.get_max_tile())
        
        if game.accumulated_score > best_score:
            best_score = game.accumulated_score
        
        # 打印进度
        if (game_idx + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            avg_score = np.mean(scores[-print_interval:])
            avg_max_tile = np.mean(max_tiles[-print_interval:])
            speed = (game_idx + 1) / elapsed
            
            print(
                f"Game {game_idx + 1}/{num_games} | "
                f"Avg Score: {avg_score:.0f} | "
                f"Best: {best_score} | "
                f"Max Tile: {avg_max_tile:.0f} | "
                f"Speed: {speed:.2f} games/s"
            )
            
            # 保存模型
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'game_idx': game_idx,
                'best_score': best_score,
                'avg_score': avg_score
            }, save_path)
    
    # 最终保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'game_idx': num_games,
        'best_score': best_score,
        'avg_score': np.mean(scores[-100:])
    }, save_path)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total games: {num_games}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average score (last 100): {np.mean(scores[-100:]):.0f}")
    print(f"Best score: {best_score}")
    print(f"Best max tile: {max(max_tiles)}")
    print(f"Model saved to: {save_path}")
    
    return model

if __name__ == "__main__":
    train_simple(num_games=500)
