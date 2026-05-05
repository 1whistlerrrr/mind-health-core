
# pretrain/ - 预训练阶段

模型的第一阶段，学习通用的语言建模能力。

## 📁 文件说明

### train.py
- 交互式训练脚本
- 支持多种训练配置
- 实时生成示例

### train_minimind.py
- 快速训练脚本
- 原始训练流程

### quick_demo.py
- 最小化演示
- 快速跑通整个流程

## 🚀 使用方法

```bash
cd pretrain
python train.py
```

选择模式（推荐 demo），等待训练完成，模型会保存到 checkpoints/ 目录。

