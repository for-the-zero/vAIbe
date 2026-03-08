"""
多进程并行训练环境
"""
import numpy as np
import torch
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from game import Game2048
from trainer import Transition, TrainingStats
import time


class ParallelGameEnv:
    """
    并行游戏环境
    使用多线程同时运行多个游戏实例
    """
    
    def __init__(self, num_envs: int = 4):
        """
        初始化并行环境
        
        Args:
            num_envs: 并行游戏数量
        """
        self.num_envs = num_envs
        self.envs = [Game2048() for _ in range(num_envs)]
        self.states = [env.get_state() for env in self.envs]
        self.scores = [env.accumulated_score for env in self.envs]
        self.situational_scores = [env.situational_score for env in self.envs]
        
        # 游戏结束回调
        self.on_game_end = None
        
        # 已完成游戏计数
        self.games_completed = 0
        
        # 使用线程池
        self.executor = ThreadPoolExecutor(max_workers=num_envs)
    
    def reset(self, indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        重置指定环境
        
        Args:
            indices: 要重置的环境索引，None表示全部重置
            
        Returns:
            states: (num_envs, 4, 4) 状态数组
            score_features: (num_envs, 2) 分数特征数组
        """
        if indices is None:
            indices = range(self.num_envs)
        
        for i in indices:
            self.states[i] = self.envs[i].reset()
            self.scores[i] = self.envs[i].accumulated_score
            self.situational_scores[i] = self.envs[i].situational_score
        
        return self._get_batch_state()
    
    def reset_single(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """重置单个环境"""
        self.states[idx] = self.envs[idx].reset()
        self.scores[idx] = self.envs[idx].accumulated_score
        self.situational_scores[idx] = self.envs[idx].situational_score
        return self.states[idx], self._get_score_features(idx)
    
    def step(self, actions: List[int]) -> List[Transition]:
        """
        并行执行动作
        
        Args:
            actions: 每个环境要执行的动作列表
            
        Returns:
            transitions: 状态转移列表
        """
        transitions = []
        
        for i, action in enumerate(actions):
            old_state = self.states[i].copy()
            old_scores = self._get_score_features(i)
            old_situational = self.situational_scores[i]
            
            # 执行动作
            new_state, reward, moved, done = self.envs[i].move(action)
            
            # 更新状态
            self.states[i] = new_state
            self.scores[i] = self.envs[i].accumulated_score
            self.situational_scores[i] = self.envs[i].situational_score
            
            # 创建转移记录
            transition = Transition(
                state=old_state,
                scores=old_scores,
                action=action,
                reward=reward,
                next_state=new_state.copy(),
                next_scores=self._get_score_features(i),
                done=done,
                log_prob=0.0,  # 需要在外部填充
                value=0.0,     # 需要在外部填充
                valid_actions=self.envs[i].get_valid_actions()
            )
            transitions.append(transition)
            
            # 如果游戏结束，记录统计并重置
            if done:
                # 记录游戏统计
                game_stats = {
                    'score': self.scores[i],
                    'situational_score': self.situational_scores[i],
                    'max_tile': self.envs[i].get_max_tile(),
                    'moves': self.envs[i].moves_count
                }
                self.games_completed += 1
                
                # 调用回调
                if self.on_game_end:
                    self.on_game_end(game_stats)
                
                self.reset_single(i)
        
        return transitions
    
    def _get_batch_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取批量状态"""
        states = np.array(self.states, dtype=np.float32)
        score_features = np.array([
            self._get_score_features(i) for i in range(self.num_envs)
        ], dtype=np.float32)
        return states, score_features
    
    def _get_score_features(self, idx: int) -> np.ndarray:
        """获取单个环境的分数特征"""
        max_accumulated = 50000
        max_situational = 200
        return np.array([
            min(self.scores[idx] / max_accumulated, 1.0),
            min(self.situational_scores[idx] / max_situational, 1.0)
        ], dtype=np.float32)
    
    def get_valid_actions(self) -> np.ndarray:
        """获取所有环境的有效动作"""
        return np.array([env.get_valid_actions() for env in self.envs])
    
    def get_game_stats(self) -> List[dict]:
        """获取所有游戏的统计信息"""
        return [
            {
                'score': env.accumulated_score,
                'situational_score': env.situational_score,
                'max_tile': env.get_max_tile(),
                'moves': env.moves_count,
                'game_over': env.game_over
            }
            for env in self.envs
        ]
    
    def close(self):
        """关闭环境"""
        self.executor.shutdown(wait=False)


class TrainingWorker:
    """
    训练工作器
    负责收集轨迹数据
    """
    
    def __init__(
        self,
        model,
        env: ParallelGameEnv,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.env = env
        self.device = device
        
        self.stats = TrainingStats()
    
    def collect_trajectories(
        self,
        num_steps: int = 256,
        deterministic: bool = False
    ) -> List[Transition]:
        """
        收集轨迹数据
        
        Args:
            num_steps: 每个环境收集的步数
            deterministic: 是否确定性选择动作
            
        Returns:
            transitions: 收集的转移数据
        """
        all_transitions = []
        
        for _ in range(num_steps):
            # 获取当前状态
            states = np.array(self.env.states, dtype=np.float32)
            score_features = np.array([
                self.env._get_score_features(i) 
                for i in range(self.env.num_envs)
            ], dtype=np.float32)
            valid_actions = self.env.get_valid_actions()
            
            # 转换为张量
            states_t = torch.FloatTensor(states).to(self.device)
            scores_t = torch.FloatTensor(score_features).to(self.device)
            valid_t = torch.BoolTensor(valid_actions).to(self.device)
            
            # 选择动作
            actions = []
            log_probs = []
            values = []
            
            with torch.no_grad():
                for i in range(self.env.num_envs):
                    action, log_prob, value = self.model.get_action(
                        states_t[i:i+1],
                        scores_t[i:i+1],
                        valid_t[i:i+1],
                        deterministic=deterministic
                    )
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
            
            # 执行动作
            transitions = self.env.step(actions)
            
            # 填充log_prob和value
            for i, t in enumerate(transitions):
                t.log_prob = log_probs[i]
                t.value = values[i]
                all_transitions.append(t)
        
        return all_transitions
    
    def run_episode(
        self,
        deterministic: bool = True,
        max_steps: int = 10000
    ) -> dict:
        """
        运行一局演示游戏
        
        Args:
            deterministic: 是否确定性选择
            max_steps: 最大步数
            
        Returns:
            游戏统计信息
        """
        # 重置环境
        env = Game2048()
        state = env.reset()
        
        total_reward = 0
        steps = 0
        
        while not env.game_over and steps < max_steps:
            # 获取分数特征
            max_accumulated = 50000
            max_situational = 200
            scores = np.array([
                min(env.accumulated_score / max_accumulated, 1.0),
                min(env.situational_score / max_situational, 1.0)
            ], dtype=np.float32)
            
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            # 转换为张量
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            scores_t = torch.FloatTensor(scores).unsqueeze(0).to(self.device)
            valid_t = torch.BoolTensor(valid_actions).unsqueeze(0).to(self.device)
            
            # 选择动作
            with torch.no_grad():
                action, _, _ = self.model.get_action(
                    state_t, scores_t, valid_t, deterministic=deterministic
                )
            
            # 执行动作
            state, reward, moved, done = env.move(action)
            total_reward += reward
            steps += 1
        
        return {
            'score': env.accumulated_score,
            'situational_score': env.situational_score,
            'max_tile': env.get_max_tile(),
            'steps': steps,
            'total_reward': total_reward
        }


class TrainingLoop:
    """
    完整的训练循环
    """
    
    def __init__(
        self,
        model,
        trainer,
        num_envs: int = 4,
        device: str = "cpu",
        steps_per_update: int = 256,
        save_interval: int = 100,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.trainer = trainer
        self.num_envs = num_envs
        self.device = device
        self.steps_per_update = steps_per_update
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        
        # 初始化环境和工作器
        self.env = ParallelGameEnv(num_envs=num_envs)
        self.worker = TrainingWorker(model, self.env, device)
        
        # 训练状态
        self.training = False
        self.paused = False
        self.stats = TrainingStats()
        
        # 回调函数
        self.on_update_callback = None
        self.on_game_end_callback = None
    
    def train(
        self,
        total_games: int = 10000,
        stop_threshold: int = 100,
        min_improvement: float = 0.01
    ) -> None:
        """
        训练循环
        
        Args:
            total_games: 总游戏局数
            stop_threshold: 无提升停止阈值（局数）
            min_improvement: 最小提升比例
        """
        self.training = True
        
        games_since_improvement = 0
        best_avg_score = 0
        
        # 设置游戏结束回调
        def on_game_end(game_stats):
            self.stats.record_game(
                score=game_stats['score'],
                situational_score=game_stats['situational_score'],
                max_tile=game_stats['max_tile'],
                steps=game_stats['moves']
            )
            if self.on_game_end_callback:
                self.on_game_end_callback(game_stats)
        
        self.env.on_game_end = on_game_end
        
        try:
            while self.training and self.env.games_completed < total_games:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 收集轨迹
                transitions = self.worker.collect_trajectories(
                    num_steps=self.steps_per_update // self.num_envs,
                    deterministic=False
                )
                
                # 更新模型
                from trainer import RolloutBuffer
                buffer = RolloutBuffer(capacity=len(transitions))
                buffer.push_batch(transitions)
                
                update_stats = self.trainer.update(buffer)
                
                if self.on_update_callback:
                    self.on_update_callback(update_stats)
                
                # 检查停止条件（每10次更新检查一次）
                if self.env.games_completed % 10 == 0 and self.env.games_completed > 0:
                    current_avg = self.stats.get_avg_stats(window=100)['avg_score']
                    if current_avg > best_avg_score * (1 + min_improvement):
                        best_avg_score = current_avg
                        games_since_improvement = 0
                    else:
                        games_since_improvement = self.env.games_completed - int(best_avg_score / 100 * 100) if best_avg_score > 0 else 0
                    
                    if games_since_improvement >= stop_threshold:
                        print(f"No improvement for {stop_threshold} updates, stopping.")
                        break
                
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            self.training = False
            self.env.close()
    
    def stop(self) -> None:
        """停止训练"""
        self.training = False
    
    def pause(self) -> None:
        """暂停训练"""
        self.paused = True
    
    def resume(self) -> None:
        """恢复训练"""
        self.paused = False
    
    def get_stats(self) -> dict:
        """获取当前统计信息"""
        return self.stats.get_avg_stats()
    
    def save_checkpoint(self, path: str) -> None:
        """保存模型检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'stats': self.stats.get_avg_stats(),
            'games_played': self.stats.games_played
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    from model import Game2048Transformer
    from trainer import PPOTrainer
    
    # 测试并行环境
    env = ParallelGameEnv(num_envs=4)
    env.reset()
    
    print("Testing parallel environment...")
    for i in range(10):
        actions = [np.random.randint(0, 4) for _ in range(env.num_envs)]
        transitions = env.step(actions)
        print(f"Step {i}: collected {len(transitions)} transitions")
    
    env.close()
    print("Parallel environment test passed!")
