
# checkpoints/ - 模型检查点

存放所有训练好的模型权重文件。

## 📁 文件说明

### minimind_model.pth
- 完整的 checkpoint
- 包括：
  - model_state_dict - 模型权重
  - tokenizer_vocab - tokenizer
  - config - 配置信息

## 📝 目录结构建议

```
checkpoints/
├── minimind_model.pth       # 预训练模型
├── sft/
│   └── minimind_sft.pth     # SFT 模型
└── rl/
    └── minimind_rlhf.pth   # RL 模型
```

## ⚠️ 注意

- 模型文件通常很大，考虑是否加入 gitignore
- 使用 git-lfs 或者只在本地保存模型
- 已经在 .gitignore 中配置了忽略 .pth 文件

