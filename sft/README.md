
# sft/ - 监督微调阶段

第二阶段，让模型学会遵循人类指令。

## 📁 文件说明

### example_sft.py
- 监督微调示例代码
- 演示如何使用 SFT 数据集
- 展示从预训练 checkpoint 继续训练

## 📚 什么是 SFT?

Supervised Fine-Tuning（监督微调），使用带标注的数据让模型学会：
- 回答问题
- 遵循指令
- 保持对话风格

## 🚀 使用方法

```bash
cd sft
python example_sft.py
```

