# Diffutslator

基于扩散模型的中英互译系统。使用非自回归并行生成，通过DDIM加速推理

## 原理

### 扩散翻译的核心思想

传统翻译模型（如Transformer）是自回归的，逐token生成。扩散模型则是非自回归的，并行生成所有token：

```
自回归:  [SOS] → [token1] → [token2] → [token3] → [EOS]
                  ↓           ↓           ↓
扩散:    噪声 ──同时去噪──→ 完整句子（一步生成所有token）
```

### 双向翻译架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           噪声空间 (共享)                            │
│                             [L × D]                                 │
│                                                                     │
│    中文嵌入 ──前向扩散(q_sample)──→ 噪声 ←──前向扩散── 英文嵌入      │
│                    ↓                       ↓                        │
│              中文去噪网络            英文去噪网络                    │
│                    ↓                       ↓                        │
│              中文逆扩散              英文逆扩散                      │
│                    ↓                       ↓                        │
│                中文输出               英文输出                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 翻译流程

以 **中译英** 为例：

1. **编码**: 中文句子 → 中文token → 中文嵌入向量
2. **前向扩散**: 中文嵌入添加噪声到指定时间步（或到纯噪声）
3. **逆扩散去噪**: 
   - 前半段：用中文去噪网络（保持源语言特征）
   - 后半段：切换到英文去噪网络（转向目标语言）
4. **解码**: 最终嵌入 → 英文token → 英文句子

### 为什么扩散能做翻译？

扩散过程将数据逐步加噪变成纯噪声，逆扩散则从噪声恢复数据。关键洞察：

- 两种语言嵌入经过充分加噪后，在噪声空间中变得"不可区分"
- 从这个共享噪声空间出发，用不同语言的去噪路径，可以恢复到不同语言
- 类比：把中文和英文都"打散"成同样的积木，再用英文的说明书拼回去

## 安装

### 依赖

```bash
pip install torch tqdm
```

### 硬件要求

- CPU训练可用（本项目针对CPU优化）
- 内存：至少4GB
- 推荐：GPU可大幅加速

## 快速开始

### 训练

```bash
# 快速验证模式（1000条数据，5轮）
python train.py --quick

# 完整训练
python train.py

# 从检查点续训
python train.py --resume checkpoints/epoch_1.pt
```

训练中按 `Ctrl+C` 可安全中断，自动保存 `checkpoints/interrupted.pt`。

### 推理

```bash
# 中译英
python inference.py --text "你好世界" --zh

# 英译中
python inference.py --text "Hello world" --en

# 交互模式
python inference.py --interactive
```

## 详细使用

### 训练命令

```bash
# 基本训练
python train.py

# 快速验证（小数据集，少轮次）
python train.py --quick

# 从检查点续训
python train.py --resume checkpoints/best.pt

# 使用更多数据
python train.py --max-samples 10000

# 指定轮次和批量
python train.py --epochs 20 --batch-size 32
```

### 推理命令

```bash
# 基本推理（中译英）
python inference.py --text "今天天气很好" --zh

# 英译中
python inference.py --text "The weather is nice today" --en

# 使用DDPM（更慢但可能更准）
python inference.py --text "你好" --zh --ddpm

# 交互模式
python inference.py --interactive

# 指定检查点
python inference.py --text "你好" --zh --checkpoint checkpoints/best.pt

# 静默模式（不显示扩散过程）
python inference.py --text "你好" --zh --quiet
```

## 配置参数

### 模型配置 (ModelConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 256 | 嵌入维度，影响模型容量 |
| `n_heads` | 4 | 多头注意力头数 |
| `n_layers` | 4 | Transformer编码器层数 |
| `d_ff` | 512 | 前馈网络隐藏层维度 |
| `max_len` | 128 | 最大序列长度 |
| `dropout` | 0.1 | Dropout比率 |
| `vocab_size_zh` | 8000 | 中文词表大小 |
| `vocab_size_en` | 8000 | 英文词表大小 |

### 扩散配置 (DiffusionConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `timesteps` | 1000 | 训练时的扩散总步数 |
| `ddim_steps` | 50 | DDIM推理采样步数 |
| `beta_start` | 0.0001 | 噪声调度起始值 |
| `beta_end` | 0.02 | 噪声调度结束值 |

### 训练配置 (TrainingConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 批量大小 |
| `learning_rate` | 1e-4 | 学习率 |
| `weight_decay` | 0.01 | 权重衰减 |
| `warmup_steps` | 500 | 学习率预热步数 |
| `epochs` | 10 | 训练轮次 |
| `save_every` | 1 | 每N轮保存检查点 |

### 数据配置 (DataConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_samples` | None | 最大样本数（None=全部） |
| `min_len` | 2 | 最小句子长度 |
| `max_len` | 128 | 最大句子长度 |

## 架构说明

### 分词器 (tokenizer.py)

使用BPE（Byte Pair Encoding）算法：

- **中文**: 字符级为主，BPE处理罕见词和数字
- **英文**: 标准BPE子词分割
- 词表大小：各8000 tokens
- 特殊token: `<pad>`, `<sos>`, `<eos>`, `<unk>`, `<mask>`

```python
# 示例
tokenizer_zh.encode("你好世界")  # [123, 456, 789]
tokenizer_en.encode("hello world")  # [234, 567]
```

### 嵌入层 (embedding.py)

```python
class LanguageEmbedding:
    token_embedding    # [vocab_size, d_model]
    position_embedding # [max_len, d_model]
    length_embedding   # [max_len, d_model]
```

将离散token转换为连续向量，加入位置信息。

### 噪声预测网络 (model.py)

```python
class DiffusionTransformer:
    """基于Transformer的噪声预测网络"""
    
    # 输入: x_t [batch, len, d_model], t [batch], lang [str]
    # 输出: predicted_noise [batch, len, d_model]
    
    # 结构:
    # 1. 时间步嵌入 (sinusoidal)
    # 2. 语言特定输入投影
    # 3. N层 Transformer blocks
    # 4. 语言特定输出投影
```

### 扩散过程 (diffusion.py)

```python
# 前向扩散（加噪）
x_t, noise = diffusion.q_sample(x_0, t)  # x_0 → x_t

# 反向扩散（去噪）
x_t_minus_1 = diffusion.p_sample(x_t, t, predicted_noise)
```

使用线性噪声调度，支持DDIM加速采样。

### 语言切换器 (switcher.py)

```python
class LanguageSwitcher:
    """判断当前噪声状态更接近哪种语言"""
    
    # 输入: x_t [batch, len, d_model]
    # 输出: lang_prob [batch, 2]  # [中文概率, 英文概率]
```

在推理时判断何时切换去噪路径。

## 文件结构

```
diffutslator/
├── config.py       # 超参数配置
├── tokenizer.py    # BPE分词器
├── embedding.py    # 嵌入层
├── model.py        # 噪声预测网络 (Transformer)
├── diffusion.py    # 扩散过程 + DDIM采样
├── switcher.py     # 语言切换分类器
├── dataset.py      # 数据加载（流式）
├── train.py        # 训练脚本
├── inference.py    # 推理脚本
├── main.py         # 主入口
├── utils.py        # 工具函数
├── .cache/         # 分词器缓存
│   ├── tokenizer_zh.json
│   └── tokenizer_en.json
└── checkpoints/    # 模型检查点
    ├── best.pt
    ├── epoch_1.pt
    └── interrupted.pt
```

## 数据集

- `_dataset/cveto/`
- `_dataset/tatoeba.tsv`

---

上面是AI生成的，我到这补充一下

生成这个项目的模型是GLM-5，用iflow cli，在我的电脑上训练了九个半小时，用了2.8w条数据，权重在checkpoints下