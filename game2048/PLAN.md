# 项目计划：基于Transformer的2048游戏AI

## 技术方案概述

### 硬件约束
- AMD Ryzen 5 PRO 4650U (6核12线程) CPU
- 8GB RAM
- 无NVIDIA GPU，纯CPU训练
- 需要小型高效的模型架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    GUI主窗口 (PyQt5)                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────────────────────┐ │
│  │  2048游戏面板 │  │         训练状态面板              │ │
│  │   (4x4网格)   │  │  - 当前分数/局面分数              │ │
│  │              │  │  - 训练速度 (games/sec)          │ │
│  │              │  │  - 累积分数变化曲线               │ │
│  │              │  │  - 局面分数变化曲线               │ │
│  └──────────────┘  └──────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  控制面板: [训练模式] [演示模式] [开始/停止] [AI托管]     │
└─────────────────────────────────────────────────────────┘
```

## 模块设计

### 1. 游戏核心模块 (`game.py`)

```python
class Game2048:
    """2048游戏核心逻辑"""
    
    def __init__(self):
        self.board: np.ndarray  # 4x4棋盘
        self.accumulated_score: int  # 累积分数
        self.situational_score: float  # 局面分数
        
    def reset(self) -> None: ...
    def move(self, direction: int) -> tuple[bool, bool]: ...  # (moved, game_over)
    def get_state(self) -> np.ndarray: ...  # 返回当前局面
    def calculate_situational_score(self) -> float: ...
```

**局面分数计算公式：**
```
situation_score = (
    empty_cells * 10 +                           # 空格越多越好
    max_consecutive_adjacent * 15 +              # 连续相邻数字越多越好
    log2(max_tile) * 5 -                         # 最高数字的对数
    monotonicity_penalty                         # 单调性惩罚（避免混乱）
)
```

### 2. Transformer模型 (`model.py`)

考虑到CPU训练的限制，采用小型Transformer：

```python
class Game2048Transformer(nn.Module):
    """小型Transformer用于2048决策"""
    
    def __init__(self):
        # 输入: 4x4棋盘 + 2个分数特征
        # 将棋盘展平为16个token，每个token代表一个格子的状态
        
        self.embedding = nn.Embedding(16, 64)  # 0-15 表示 log2(value)，16表示空
        self.score_embedding = nn.Linear(2, 64)  # 两种分数的embedding
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.policy_head = nn.Linear(64, 4)   # 输出4个动作的概率
        self.value_head = nn.Linear(64, 1)    # 输出状态价值
```

**模型大小估算：**
- Embedding: 17 * 64 = 1,088 参数
- Transformer (2层): ~50,000 参数
- 输出头: ~300 参数
- **总计: ~52,000 参数** - 非常小，适合CPU训练

### 3. 训练模块 (`trainer.py`)

采用 **Actor-Critic + PPO** 策略：

```python
class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def compute_advantage(self, rewards, values, dones):
        # 计算GAE (Generalized Advantage Estimation)
        ...
        
    def update(self, trajectories):
        # PPO更新逻辑
        ...
```

**奖励设计：**
```python
reward = (
    accumulated_score_delta * 0.3 +     # 累积分数增量（权重低）
    situational_score * 0.7 +           # 局面分数（权重高）
    game_over_penalty * (-100)          # 游戏结束惩罚
)
```

### 4. 多进程训练 (`parallel.py`)

利用6核CPU，同时运行多个游戏实例：

```python
class ParallelGameEnv:
    """并行游戏环境"""
    
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.envs = [Game2048() for _ in range(num_envs)]
        
    def step(self, actions: list[int]) -> list[Transition]:
        # 并行执行动作，返回状态转移
        ...
```

### 5. GUI模块 (`gui.py`)

使用 PyQt5 构建界面：

```python
class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        self.game_widget = GameBoardWidget()
        self.stats_widget = StatsWidget()
        self.control_widget = ControlWidget()
        
        # 训练线程
        self.training_thread = TrainingThread()
        
    def switch_mode(self, mode: str): ...
    def update_display(self): ...
```

## 文件结构

```
game2048/
├── TASK.md          # 任务描述
├── PLAN.md          # 本文件
├── main.py          # 入口文件
├── game.py          # 游戏核心逻辑
├── model.py         # Transformer模型定义
├── trainer.py       # PPO训练器
├── parallel.py      # 多进程训练
├── gui.py           # GUI界面
├── utils.py         # 工具函数
├── requirements.txt # 依赖
└── checkpoints/     # 模型保存目录
```

## 实现步骤

### 阶段1: 核心游戏逻辑
1. 实现 `game.py` - 2048游戏规则
2. 实现局面分数计算
3. 编写游戏逻辑单元测试

### 阶段2: 模型与训练
4. 实现 `model.py` - Transformer模型
5. 实现 `trainer.py` - PPO训练器
6. 实现 `parallel.py` - 多进程环境
7. 验证训练流程可以运行

### 阶段3: GUI界面
8. 实现 `gui.py` - 主窗口和游戏面板
9. 实现训练状态可视化（分数曲线）
10. 实现模式切换（训练/演示）

### 阶段4: 整合与优化
11. 整合所有模块
12. 性能优化
13. 模型保存/加载功能

## 依赖

```
torch>=2.0.0
numpy>=1.24.0
PyQt5>=5.15.0
matplotlib>=3.7.0
```

## 训练策略细节

### 状态表示
- 棋盘状态：将每个格子的值转换为 log2(value)，空格为0
- 分数归一化：累积分数和局面分数归一化到 [0, 1]

### 动作空间
- 0: 上
- 1: 下  
- 2: 左
- 3: 右

### 训练超参数
- Learning rate: 1e-4
- Batch size: 64
- PPO clip ratio: 0.2
- GAE lambda: 0.95
- Discount factor (gamma): 0.99
- 并行环境数: 4 (根据CPU核心数调整)

### 停止条件
- 连续100局游戏平均分数无提升
- 用户手动停止
