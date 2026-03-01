# 环境配置

## 硬件

| 组件 | 规格 |
|------|------|
| CPU | AMD Ryzen 5 PRO 4650U (6核12线程) |
| RAM | 8GB DDR4 3200MHz |
| GPU | AMD Radeon Graphics (集成, 512MB) |
| 存储 | C: 269GB / D: 206GB |

## 硬件加速

| 技术 | 支持 | 说明 |
|------|------|------|
| AVX2 | ✓ | llama.cpp 可利用 |
| MKL/OpenBLAS | ✓ | PyTorch CPU矩阵加速 |
| DirectML | ⚠ | 可用但生态弱 |

### 实际可用
- **llama.cpp** - AVX2优化，CPU推理效率最高
- **ONNX Runtime** - DirectML/CPU后端
- **OpenVINO** - Intel方案，AMD CPU兼容

## 软件

| 软件 | 版本 |
|------|------|
| OS | Windows 10 专业版 (Build 19044) |
| Python | 3.12.10 |
| FFmpeg | 已安装 |
| Git | 2.49.0 |

## AI开发建议

### 不推荐
- 本地大规模模型训练 (无NVIDIA GPU)
- 本地运行大参数模型 (>3B)

### 推荐
- **推理**: llama.cpp / Ollama (GGUF量化模型)
- **框架**: PyTorch (CPU版) + Hugging Face
- **模型**: 优先量化版 (Q4/Q5)、小型模型 (TinyLlama, Phi-3, Gemma-2B)
- **数据**: 本地预处理 → 云端训练