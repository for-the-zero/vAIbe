# 2048 AI Trainer

基于 Transformer 的 2048 游戏人工智能训练器，使用 PPO（Proximal Policy Optimization）强化学习算法，让 AI 学会玩 2048 游戏。

## 项目简介

本项目实现了一个完整的 2048 游戏 AI 训练系统，包括：

- **游戏引擎**: 完整的 2048 游戏逻辑实现
- **深度学习模型**: 基于 Transformer 架构的策略网络
- **强化学习训练**: PPO 算法实现
- **可视化界面**: PyQt5 图形界面，支持训练监控和演示
- **命令行工具**: 支持无 GUI 的训练和演示模式

## 功能特点

### 1. Transformer 模型

采用小型 Transformer 架构，专为 CPU 训练优化：

- **参数量**: 约 77,000 个参数（~300KB）
- **输入处理**: 
  - 棋盘状态编码为 16 个 token（每个格子对应一个 token）
  - 分数特征（累积分数、局面分数）作为额外输入
  - 位置编码：行/列位置嵌入
- **网络结构**:
  - 2 层 Transformer Encoder
  - 4 个注意力头
  - 隐藏维度 64
  - 前馈网络维度 128
- **输出**:
  - 策略头：4 个动作（上/下/左/右）的概率分布
  - 价值头：当前状态的价值评估

### 2. 双评分机制

#### 累积分数（Accumulated Score）
传统 2048 计分方式，每次合成砖块获得合成后砖块的数值作为分数。

#### 局面分数（Situational Score）
综合评估当前局面的质量，鼓励 AI 保持良好局面：

```
局面分数 = 空格数 × 10 + 最大连续相邻数 × 15 + log₂(最大砖块) × 5 + 单调性奖励
```

- **空格数**: 空格越多，操作空间越大
- **连续相邻数**: 如 512-1024-2048 连续排列，便于后续合并
- **单调性**: 鼓励数字按方向有序排列

### 3. PPO 训练算法

使用 Proximal Policy Optimization 算法进行训练：

- **优势估计**: GAE（Generalized Advantage Estimation）
- **策略裁剪**: 防止策略更新过大
- **价值函数**: 辅助训练，提供状态价值估计
- **熵正则化**: 鼓励探索

### 4. GUI 界面

基于 PyQt5 的图形界面：

- **训练模式**: 
  - 设置训练局数
  - 实时显示训练进度
  - 分数曲线可视化
  - 训练完成后自动保存模型
  
- **演示模式**:
  - 键盘手动操作
  - AI 托管模式
  - 单步执行
  - 自动连续执行
  - 实时局面分数曲线

## 安装

### 环境要求

- Python 3.8+
- Windows / Linux / macOS

### 安装依赖

```bash
cd game2048
pip install -r requirements.txt
```

### 依赖列表

```
torch>=2.0.0      # 深度学习框架
numpy<2           # 数值计算
PyQt5>=5.15.0     # GUI 框架
matplotlib>=3.7.0 # 绘图库
```

## 使用方法

### 1. GUI 模式

```bash
python main.py
```

启动图形界面后：

**训练模式**:
1. 选择 "Training Mode"
2. 设置训练局数（默认 500）
3. 点击 "Start Training" 开始训练
4. 训练完成后自动保存到 `checkpoints/model.pt`
5. 可随时点击 "Stop Training" 停止

**演示模式**:
1. 选择 "Demo Mode"
2. 点击 "Load Model" 加载已训练模型
3. 使用方式：
   - 键盘方向键：手动操作
   - "AI Mode"：切换 AI 托管
   - "Step"：AI 单步执行
   - "Auto"：AI 自动连续执行
   - "Reset"：重新开始游戏

### 2. 命令行训练

```bash
# 训练 1000 局
python main.py --train --games 1000

# 使用 4 个并行环境
python main.py --train --games 1000 --envs 4

# 设置随机种子
python main.py --train --games 1000 --seed 42
```

### 3. 演示模式

```bash
# 加载模型并演示 5 局
python main.py --demo --model checkpoints/model.pt --games 5

# 不加载模型（随机权重）
python main.py --demo --games 3
```

### 4. 简单训练脚本

```bash
python train_simple.py
```

修改脚本末尾可调整训练参数：

```python
train_simple(num_games=500, save_path="checkpoints/model.pt")
```

## 项目结构

```
game2048/
├── TASK.md              # 任务需求文档
├── PLAN.md              # 项目计划文档
├── README.md            # 本文件
├── main.py              # 程序入口
├── game.py              # 2048 游戏核心逻辑
│   ├── Game2048         # 游戏类
│   ├── move()           # 移动操作
│   ├── get_state()      # 获取状态
│   └── calculate_situational_score()  # 计算局面分数
│
├── model.py             # Transformer 模型
│   ├── Game2048Transformer  # Transformer 模型
│   ├── Game2048CNN          # CNN 备选模型
│   └── get_action()         # 动作选择
│
├── trainer.py           # PPO 训练器
│   ├── PPOTrainer       # PPO 训练类
│   ├── RolloutBuffer    # 经验缓冲区
│   ├── Transition       # 状态转移数据结构
│   └── TrainingStats    # 训练统计
│
├── parallel.py          # 并行训练环境
│   ├── ParallelGameEnv  # 并行游戏环境
│   ├── TrainingWorker   # 训练工作器
│   └── TrainingLoop     # 训练循环
│
├── gui.py               # GUI 界面
│   ├── MainWindow       # 主窗口
│   ├── GameBoardWidget  # 游戏面板
│   ├── ScoreWidget      # 分数显示
│   ├── PlotCanvas       # 曲线绑图
│   └── SimpleTrainingThread  # 训练线程
│
├── train_simple.py      # 简化训练脚本
├── utils.py             # 工具函数
├── requirements.txt     # 依赖列表
└── checkpoints/         # 模型保存目录
    └── model.pt         # 训练好的模型
```

## 模型架构详解

### 输入表示

```python
# 棋盘状态 (4, 4)
# 每个格子值转换为 log₂(value)，空格为 0
state = [[0, 1, 2, 0],   # 对应 [空, 2, 4, 空]
         [1, 2, 3, 1],   # 对应 [2, 4, 8, 2]
         ...]

# 分数特征 (2,)
# [归一化累积分数, 归一化局面分数]
scores = [0.05, 0.85]
```

### 网络结构

```
Input: (batch, 4, 4) board + (batch, 2) scores
    ↓
Position Embedding: (batch, 16, 64)
    + Spatial Embedding: (batch, 16, 64)
    + Score Embedding: (batch, 1, 64)
    ↓
Transformer Encoder (2 layers)
    - Multi-Head Attention (4 heads)
    - Feed-Forward Network (dim=128)
    ↓
Global Mean Pooling: (batch, 64)
    ↓
    ├── Policy Head → (batch, 4)  # 动作概率
    └── Value Head  → (batch, 1)  # 状态价值
```

## 训练策略

### 奖励设计

```python
reward = 局面分数变化 × 0.7 + 累积分数增量 × 0.003

# 游戏结束惩罚
if game_over:
    reward -= 10.0
```

### 超参数

| 参数 | 值 |
|------|-----|
| Learning Rate | 3e-4 |
| Batch Size | 64 |
| PPO Clip Ratio | 0.2 |
| GAE Lambda | 0.95 |
| Discount Factor (γ) | 0.99 |
| Entropy Coefficient | 0.01 |

## 训练结果

### 500 局训练后

| 指标 | 数值 |
|------|------|
| 平均分数 | ~2500 |
| 最高分数 | 6812 |
| 最大砖块 | 512 |
| 训练时间 | ~9 分钟 |

### 分数分布

```
随机权重: 平均 ~800, 最高 ~2000
训练 500 局: 平均 ~2500, 最高 ~6800
```

## 开发说明

### 添加新功能

1. **修改局面评分**: 编辑 `game.py` 中的 `calculate_situational_score()`
2. **调整模型**: 修改 `model.py` 中的网络结构
3. **优化训练**: 调整 `trainer.py` 中的超参数

### 调试模式

```python
# 在 game.py 中测试游戏逻辑
if __name__ == "__main__":
    game = Game2048()
    print(game)
    game.move(Game2048.LEFT)
    print(game)
```

## 已知问题

- Windows 下 PyTorch 可能需要特定版本以避免 DLL 加载问题
- NumPy 2.x 与 PyTorch 存在兼容性问题，建议使用 NumPy < 2

## 参考资料

- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [Transformer 论文](https://arxiv.org/abs/1706.03762)
- [2048 游戏](https://play2048.co/)

## 许可证

MIT License

---

*本项目由 GLM-5 开发实现*