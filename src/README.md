
# src/ - 核心代码

这个目录存放项目的核心代码，是所有阶段都可能用到的基础模块。

## 📁 文件说明

### config.py
- 模型和训练配置管理
- 支持多种场景配置（tiny/demo/local/cloud）
- 定义了完整的参数配置结构

### model.py
- Tokenizer：字符和子词级别 tokenizer
- MiniMind：完整的 LLM 模型（LLaMA 风格架构）
- RMSNorm、RoPE、Attention、FeedForward 等组件
- 训练工具函数
- 推理工具
- SFT Dataset

