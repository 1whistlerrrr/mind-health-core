
# MiniMind - 从零开始的大语言模型项目

一个完整的、可运行的大语言模型项目，包括预训练、监督微调、强化学习等完整流程，专为学习和面试准备设计。

## 📁 目录结构

```
python-leo/
├── README.md                    # 项目说明（本文件）
├── .gitignore
├── src/                       # 核心代码目录
│   ├── config.py              # 配置管理
│   └── model.py               # 模型定义、Tokenizer、训练工具
├── pretrain/                  # 预训练阶段
│   ├── README.md
│   ├── train.py
│   ├── train_minimind.py
│   └── quick_demo.py
├── sft/                       # 监督微调 (SFT) 阶段
│   ├── README.md
│   └── example_sft.py
├── rl/                        # 强化学习 (RL) 阶段
│   └── README.md
├── inference/                 # 推理/生成阶段
│   ├── README.md
│   ├── 完整推理示例.py
│   ├── load_model_check.py
│   └── use_trained_model.py
├── data/                       # 训练数据
│   ├── chinese.txt
│   └── cleaned_corpus.txt
├── checkpoints/               # 保存的模型检查点
│   └── minimind_model.pth
├── docs/                      # 文档和学习笔记
│   ├── README.md
│   ├── 学习笔记.md
│   ├── FAQ_详解.md
│   ├── README_完整项目说明.md
│   ├── 模型保存与加载详解.md
│   └── 面试必备-大模型深度解析.md
└── scripts/                   # 快速脚本目录
└── old_files/                # 旧文件备份
```

## 🚀 快速开始

1. **激活环境**
```bash
# 如果你没有虚拟环境，创建一个
python -m venv venv
source venv/bin/activate
pip install torch tqdm
```

2. **预训练（可选）**
```bash
cd pretrain
python train.py
```

3. **推理（使用已有的模型）**
```bash
cd inference
python 完整推理示例.py
```

## 📚 学习指南

- 首先阅读 `docs/` 下的文档，按顺序学习：
1. `docs/学习笔记.md` - 基础入门
2. `docs/模型保存与加载详解.md` - 理解模型如何保存
3. `docs/面试必备-大模型深度解析.md` - 面试准备

## 📝 各阶段说明

1. **pretrain/** - 模型预训练，学习基本的语言建模
2. **sft/** - 监督微调，让模型学会遵循指令
3. **rl/** - 强化学习阶段，对齐人类偏好
4. **inference/** - 推理/生成，使用训练好的模型

## 🔧 核心文件说明

- **src/config.py** - 所有配置管理
- **src/model.py** - 模型架构、Tokenizer、训练逻辑

