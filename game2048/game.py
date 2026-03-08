"""
2048游戏核心逻辑
"""
import numpy as np
from typing import Tuple, Optional
import random


class Game2048:
    """2048游戏核心类"""
    
    # 动作定义
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(self):
        self.board: np.ndarray = np.zeros((4, 4), dtype=np.int64)
        self.accumulated_score: int = 0
        self.situational_score: float = 0.0
        self.game_over: bool = False
        self.moves_count: int = 0
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置游戏，返回初始状态"""
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.accumulated_score = 0
        self.situational_score = 0.0
        self.game_over = False
        self.moves_count = 0
        
        # 开局生成一个2
        self._spawn_tile(value=2)
        self._update_situational_score()
        return self.get_state()
    
    def _spawn_tile(self, value: Optional[int] = None) -> bool:
        """
        在空格生成新砖块
        开局时value=2，后续随机2或4
        返回是否成功生成
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        
        row, col = random.choice(empty_cells)
        if value is None:
            # 90%概率生成2，10%概率生成4
            value = 2 if random.random() < 0.9 else 4
        self.board[row, col] = value
        return True
    
    def _compress(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        压缩一行/列，将非零元素移到一端
        返回压缩后的行和合并得分
        """
        # 移除零，填充到末尾
        non_zero = line[line != 0]
        new_line = np.zeros_like(line)
        score = 0
        
        pos = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # 合并
                new_line[pos] = non_zero[i] * 2
                score += new_line[pos]
                i += 2
            else:
                new_line[pos] = non_zero[i]
                i += 1
            pos += 1
        
        return new_line, score
    
    def _move_left(self) -> Tuple[bool, int]:
        """向左移动，返回(是否移动, 得分)"""
        moved = False
        total_score = 0
        
        for i in range(4):
            original = self.board[i].copy()
            new_line, score = self._compress(self.board[i])
            self.board[i] = new_line
            total_score += score
            if not np.array_equal(original, new_line):
                moved = True
        
        return moved, total_score
    
    def _move_right(self) -> Tuple[bool, int]:
        """向右移动"""
        moved = False
        total_score = 0
        
        for i in range(4):
            original = self.board[i].copy()
            new_line, score = self._compress(self.board[i][::-1])
            self.board[i] = new_line[::-1]
            total_score += score
            if not np.array_equal(original, self.board[i]):
                moved = True
        
        return moved, total_score
    
    def _move_up(self) -> Tuple[bool, int]:
        """向上移动"""
        moved = False
        total_score = 0
        
        for j in range(4):
            original = self.board[:, j].copy()
            new_line, score = self._compress(self.board[:, j])
            self.board[:, j] = new_line
            total_score += score
            if not np.array_equal(original, new_line):
                moved = True
        
        return moved, total_score
    
    def _move_down(self) -> Tuple[bool, int]:
        """向下移动"""
        moved = False
        total_score = 0
        
        for j in range(4):
            original = self.board[:, j].copy()
            new_line, score = self._compress(self.board[:, j][::-1])
            self.board[:, j] = new_line[::-1]
            total_score += score
            if not np.array_equal(original, self.board[:, j]):
                moved = True
        
        return moved, total_score
    
    def move(self, direction: int) -> Tuple[np.ndarray, float, bool, bool]:
        """
        执行移动
        
        Args:
            direction: 0=上, 1=下, 2=左, 3=右
            
        Returns:
            state: 新状态
            reward: 奖励（累积分数增量 + 局面分数变化）
            moved: 是否成功移动
            done: 游戏是否结束
        """
        if self.game_over:
            return self.get_state(), 0.0, False, True
        
        old_accumulated = self.accumulated_score
        old_situational = self.situational_score
        
        # 执行移动
        if direction == self.UP:
            moved, score = self._move_up()
        elif direction == self.DOWN:
            moved, score = self._move_down()
        elif direction == self.LEFT:
            moved, score = self._move_left()
        elif direction == self.RIGHT:
            moved, score = self._move_right()
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        if moved:
            self.accumulated_score += score
            self.moves_count += 1
            self._spawn_tile()
            self._update_situational_score()
            
            # 检查游戏是否结束
            self.game_over = self._check_game_over()
        
        # 计算奖励
        accumulated_delta = self.accumulated_score - old_accumulated
        situational_delta = self.situational_score - old_situational
        
        # 奖励 = 局面分数变化 * 0.7 + 累积分数增量 * 0.3 / 100 (归一化)
        reward = situational_delta * 0.7 + accumulated_delta * 0.003
        
        # 游戏结束惩罚
        if self.game_over:
            reward -= 10.0
        
        return self.get_state(), reward, moved, self.game_over
    
    def _check_game_over(self) -> bool:
        """检查游戏是否结束"""
        # 还有空格
        if np.any(self.board == 0):
            return False
        
        # 检查是否还能合并
        for i in range(4):
            for j in range(4):
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:
                    return False
                if j < 3 and self.board[i, j] == self.board[i, j + 1]:
                    return False
        
        return True
    
    def _update_situational_score(self) -> None:
        """
        更新局面分数
        局面分数 = 空格数 * 10 + 最大连续相邻数 * 15 + log2(最大数字) * 5 + 单调性奖励
        """
        empty_cells = np.sum(self.board == 0)
        
        # 计算最大连续相邻数字
        max_consecutive = self._calculate_max_consecutive()
        
        # 最高数字的对数
        max_tile = np.max(self.board)
        max_tile_log = np.log2(max_tile) if max_tile > 0 else 0
        
        # 单调性评估（鼓励数字按方向排列）
        monotonicity = self._calculate_monotonicity()
        
        # 局面分数
        self.situational_score = (
            empty_cells * 10 +
            max_consecutive * 15 +
            max_tile_log * 5 +
            monotonicity * 5
        )
    
    def _calculate_max_consecutive(self) -> int:
        """
        计算最大连续相邻数字数量
        相邻砖块拥有相邻数字，如512 1024 2048为3
        """
        max_count = 0
        
        # 检查所有行
        for i in range(4):
            count = self._count_consecutive_in_line(self.board[i])
            max_count = max(max_count, count)
        
        # 检查所有列
        for j in range(4):
            count = self._count_consecutive_in_line(self.board[:, j])
            max_count = max(max_count, count)
        
        return max_count
    
    def _count_consecutive_in_line(self, line: np.ndarray) -> int:
        """计算一行/列中的最大连续相邻数字"""
        non_zero = line[line != 0]
        if len(non_zero) < 2:
            return 0
        
        max_count = 1
        current_count = 1
        
        for i in range(1, len(non_zero)):
            # 相邻数字：2的幂次相邻
            if abs(np.log2(non_zero[i]) - np.log2(non_zero[i-1])) == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
        
        return max_count
    
    def _calculate_monotonicity(self) -> float:
        """
        计算单调性
        鼓励数字在行/列上递增或递减
        """
        score = 0.0
        
        # 行单调性
        for i in range(4):
            row = self.board[i]
            row = row[row != 0]
            if len(row) >= 2:
                # 检查递增
                if all(row[i] <= row[i+1] for i in range(len(row)-1)):
                    score += len(row) - 1
                # 检查递减
                elif all(row[i] >= row[i+1] for i in range(len(row)-1)):
                    score += len(row) - 1
        
        # 列单调性
        for j in range(4):
            col = self.board[:, j]
            col = col[col != 0]
            if len(col) >= 2:
                if all(col[i] <= col[i+1] for i in range(len(col)-1)):
                    score += len(col) - 1
                elif all(col[i] >= col[i+1] for i in range(len(col)-1)):
                    score += len(col) - 1
        
        return score
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态表示
        返回: (4, 4) 棋盘，值为log2(value)，空格为0
        """
        state = np.zeros((4, 4), dtype=np.float32)
        non_zero_mask = self.board > 0
        state[non_zero_mask] = np.log2(self.board[non_zero_mask])
        return state
    
    def get_state_with_scores(self) -> np.ndarray:
        """
        获取带分数的状态表示
        返回: (18,) 包含16个棋盘位置 + 2个分数
        """
        board_state = self.get_state().flatten()
        
        # 归一化分数
        max_accumulated = 50000  # 假设最大累积分数
        max_situational = 200    # 假设最大局面分数
        
        normalized_accumulated = min(self.accumulated_score / max_accumulated, 1.0)
        normalized_situational = min(self.situational_score / max_situational, 1.0)
        
        return np.concatenate([
            board_state / 15.0,  # 归一化到 [0, 1]，最大2048 = log2(2048) = 11
            [normalized_accumulated, normalized_situational]
        ]).astype(np.float32)
    
    def get_valid_actions(self) -> np.ndarray:
        """获取当前可执行的动作"""
        valid = np.zeros(4, dtype=bool)
        
        # 临时保存状态
        old_board = self.board.copy()
        old_accumulated = self.accumulated_score
        
        for direction in range(4):
            if direction == self.UP:
                moved, _ = self._move_up()
            elif direction == self.DOWN:
                moved, _ = self._move_down()
            elif direction == self.LEFT:
                moved, _ = self._move_left()
            else:
                moved, _ = self._move_right()
            
            valid[direction] = moved
            self.board = old_board.copy()
        
        self.accumulated_score = old_accumulated
        return valid
    
    def get_max_tile(self) -> int:
        """获取最大砖块值"""
        return int(np.max(self.board))
    
    def get_empty_cells_count(self) -> int:
        """获取空格数量"""
        return int(np.sum(self.board == 0))
    
    def __str__(self) -> str:
        """字符串表示"""
        result = []
        for row in self.board:
            result.append(" | ".join(f"{int(x):4d}" if x > 0 else "   ." for x in row))
        return "\n".join(result)


if __name__ == "__main__":
    # 测试游戏
    game = Game2048()
    print("Initial state:")
    print(game)
    print(f"Accumulated score: {game.accumulated_score}")
    print(f"Situational score: {game.situational_score}")
    
    # 测试一些移动
    moves = ['UP', 'LEFT', 'DOWN', 'RIGHT']
    for i in range(10):
        direction = i % 4
        state, reward, moved, done = game.move(direction)
        print(f"\nMove {moves[direction]}: moved={moved}, done={done}")
        print(game)
        print(f"Reward: {reward:.2f}")
