
# 🐋 AutoDL 脚本文件夹说明

这个文件夹专门为 AutoDL 环境准备的完整小模型微调方案！

## 🎯 完整步骤

### 第 1 步：去 AutoDL 租机器
推荐配置：
- GPU：RTX 3060 / 3070 / 3080
- 显存：≥ 8GB
- 镜像：选预装 PyTorch 2.x + CUDA

### 第 2 步：上传这些脚本到 AutoDL
把这个文件夹的内容全部上传到 AutoDL 实例上！

### 第 3 步：准备环境
```bash
cd /path/to/autodl_scripts
python3 -m venv qwen-env
source qwen-env/bin/activate
pip install --upgrade pip
pip install transformers datasets peft accelerate torch sentencepiece protobuf
```

### 第 4 步：开始微调！
```bash
python train_lora.py
```

### 第 5 步：测试效果！
```bash
python infer_lora.py
```

---

## 📝 脚本说明
| 脚本 | 功能 |
|-----|------|
| `train_lora.py` | Qwen2.5-0.5B-Instruct LoRA 微调主脚本 |
| `infer_lora.py` | 测试微调后的效果 |
| `README.md` | 本说明文件 |
