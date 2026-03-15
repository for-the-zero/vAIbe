# Diffutslator 实现计划

基于扩散模型的中英互译系统

## 一、架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                        噪声空间 (共享)                           │
│                          [L×D]                                   │
│              ┌─────────────────────────┐                         │
│              │                         │                         │
│    中文扩散 ↗     语言切换判断器     ↖ 英文扩散                  │
│    (加噪)          [分类器]          (加噪)                      │
│              │                         │                         │
│              └─────────────────────────┘                         │
│              ↓                         ↓                         │
│        中文逆扩散                 英文逆扩散                      │
│         (去噪)                    (去噪)                         │
│              ↓                         ↓                         │
│     ┌────────────┐            ┌────────────┐                    │
│     │ 中文解码器  │            │ 英文解码器  │                    │
│     └────────────┘            └────────────┘                    │
│           ↓                         ↓                           │
│       中文输出                  英文输出                         │
└─────────────────────────────────────────────────────────────────┘
```

### 核心设计决策

| 问题 | 决策 | 理由 |
|------|------|------|
| 扩散空间 | 词嵌入连续空间 | 实现成熟、CPU友好、训练稳定 |
| 长度处理 | 变长序列 + 长度嵌入 | 扩散可变长，逆扩散收敛到目标长度 |
| 双向切换 | 可学习分类器 | 让模型自己判断何时切换 |

## 二、模块设计

### 2.1 分词器 (tokenizer.py)

**中文分词**：字符级 + BPE
- 字符级处理中文字符
- BPE处理罕见词和数字

**英文分词**：BPE
- 使用相同的BPE算法
- 与中文共享词表大小设置

**词表**：
- 中文词表：8000 tokens
- 英文词表：8000 tokens
- 特殊token：`<pad>`, `<sos>`, `<eos>`, `<mask>`, `<unk>`

### 2.2 嵌入层 (embedding.py)

```python
class LanguageEmbedding:
    """语言特定的嵌入层"""
    - token_embedding: [vocab_size, d_model]
    - position_embedding: [max_len, d_model]
    - length_embedding: [max_len, d_model]  # 长度编码
```

**参数**：
- `d_model = 256`（CPU环境下适中）
- `max_len = 128`（最大序列长度）

### 2.3 扩散核心 (diffusion.py)

**前向扩散（加噪）**：
```python
def forward_diffusion(x_0, t):
    """
    x_0: 初始嵌入 [batch, len, d_model]
    t: 时间步 [batch]
    返回: x_t, noise
    """
    # 线性噪声调度
    alpha_t = 1 - t / T  # 简化调度
    noise = randn_like(x_0)
    x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
    return x_t, noise
```

**反向扩散（去噪）**：
```python
def reverse_diffusion(x_t, t, model):
    """
    x_t: 当前噪声状态
    t: 当前时间步
    model: 噪声预测网络
    """
    predicted_noise = model(x_t, t)
    x_t_minus_1 = denoise_step(x_t, predicted_noise, t)
    return x_t_minus_1
```

**时间调度**：
- 训练时：T = 1000 步
- 推理时：DDIM加速，可降到 10-50 步

### 2.4 噪声预测网络 (model.py)

```python
class DiffusionTransformer:
    """预测噪声的Transformer"""
    - 输入: x_t [batch, len, d_model], t [batch]
    - 输出: predicted_noise [batch, len, d_model]
    
    结构:
    - 语言特定的输入投影
    - 时间步嵌入 (sinusoidal)
    - N层 Transformer blocks
    - 语言特定的输出投影
```

**参数**（CPU优化）：
- `n_layers = 4`
- `n_heads = 4`
- `d_ff = 512`
- 总参数量：约 2M

### 2.5 语言切换器 (switcher.py)

```python
class LanguageSwitcher:
    """判断当前噪声更接近哪种语言"""
    - 输入: x_t [batch, len, d_model]
    - 输出: 语言概率 [batch, 2]  # [中文, 英文]
    
    结构:
    - 全局平均池化
    - 2层MLP
    - Softmax输出
```

### 2.6 训练流程 (train.py)

```
训练步骤:
1. 加载中英平行句对 (zh, en)
2. 分别嵌入到连续空间
3. 随机采样时间步 t
4. 对中文嵌入做前向扩散到 t 步 → zh_t
5. 对英文嵌入做前向扩散到 t 步 → en_t
6. 训练噪声预测网络预测噪声
7. 训练切换器判断语言
8. 反向传播更新参数
```

**损失函数**：
```python
L_total = L_noise_zh + L_noise_en + λ * L_switcher

L_noise: 噪声预测MSE损失
L_switcher: 语言分类交叉熵损失
```

### 2.7 推理流程 (inference.py)

**中文→英文翻译**：
```
1. 中文输入 → 中文嵌入
2. 完整前向扩散到纯噪声 (T步)
3. 迭代反向扩散:
   for t in [T, T-1, ..., 1]:
       - 切换器判断当前语言
       - 若判断为中文→用中文去噪
       - 若判断为英文→切换到英文去噪
       - 输出当前步骤状态（可视化）
4. 最终噪声状态 → 英文解码 → 英文输出
```

**英文→中文翻译**：对称过程

## 三、文件结构

```
diffutslator/
├── TASK.md              # 任务描述
├── PLAN.md              # 本文件
├── config.py            # 超参数配置
├── tokenizer.py         # 分词器
├── embedding.py         # 嵌入层
├── model.py             # 扩散模型
├── diffusion.py         # 扩散过程
├── switcher.py          # 语言切换器
├── dataset.py           # 数据集加载
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── main.py              # 主入口
├── utils.py             # 工具函数
└── checkpoints/         # 模型检查点
```

## 四、实现步骤

### Phase 1: 基础框架（确保可训练）

1. **配置文件** - 定义所有超参数
2. **分词器** - 实现中英文分词
3. **数据集** - 加载tatoeba数据
4. **嵌入层** - 简单的token嵌入
5. **扩散核心** - 前向和反向扩散
6. **简单模型** - 基础噪声预测网络
7. **训练脚本** - 带进度条的训练循环

**验证目标**：能在少量数据上跑通训练，loss下降

### Phase 2: 完整架构

1. **语言切换器** - 实现切换判断
2. **变长处理** - 实现长度嵌入
3. **完整模型** - 整合所有模块
4. **推理脚本** - 可视化扩散过程

**验证目标**：完整训练流程，能输出翻译结果

### Phase 3: 优化加速

1. **DDIM采样** - 减少推理步数
2. **训练加速** - 混合精度、梯度累积
3. **模型调优** - 调整超参数

**验证目标**：提升训练和推理速度，改善翻译质量

## 五、训练策略

### 快速验证模式

```bash
# 使用tatoeba前1000条数据
# batch_size=8, epochs=10
python train.py --quick --samples 1000
```

### 完整训练模式

```bash
# 使用全部数据
# 支持暂停/继续
python train.py --full
# Ctrl+C 暂停，自动保存检查点
# python train.py --resume 继续训练
```

### 训练输出

```
Epoch 1/10: 100%|████████| 125/125 [02:30<00:00, loss=0.452]
  预计剩余: 22:30 | 速度: 0.5 it/s
  最新检查点: checkpoints/model_epoch1.pt
  
按 Ctrl+C 停止训练（自动保存）
```

## 六、推理展示

```
$ python inference.py --zh "你好世界"

翻译模式: 中文 → 英文
输入: 你好世界

扩散过程:
Step 1000: [噪声状态 - 切换器: 中文 95%]
Step 900:  [噪声状态 - 切换器: 中文 78%]
Step 800:  [噪声状态 - 切换器: 中文 52%]
Step 700:  [噪声状态 - 切换器: 英文 61%] ← 语言切换!
Step 600:  [噪声状态 - 切换器: 英文 89%]
...
Step 50:   [接近完整句子 - 切换器: 英文 99%]
Step 1:    [完整句子]

输出: Hello world
```

## 七、环境适配

针对CPU环境的优化：

1. **小模型**：参数量控制在2-5M
2. **小批量**：batch_size = 4-16
3. **梯度累积**：模拟更大batch
4. **简单架构**：减少层数和维度
5. **内存优化**：及时释放中间变量

## 八、预期效果

| 指标 | 目标 |
|------|------|
| 训练速度 | 1-2 it/s (CPU) |
| 推理速度 | 1-5秒/句 (DDIM 50步) |
| 翻译质量 | 简单句子可理解 |
| 模型大小 | < 50MB |

---

*计划制定完成，待用户确认后开始实现*
