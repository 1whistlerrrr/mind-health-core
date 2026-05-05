
# 🐋 AutoDL 一步步操作指南（非常详细！

## 第一步：连接到 AutoDL
你已经有了连接信息：
```
ssh -p 33458 root@connect.bjb1.seetacloud.com
# 密码：SuQGCuKpLQTi
```

在你的本地终端运行这个命令，输入密码登录！

---

## 第二步：上传脚本到 AutoDL
登录进去后，你可以通过以下方式之一把脚本上传到服务器：

### 方式 A：用 AutoDL 的 JupyterLab 上传（最简单！
1. 在 AutoDL 控制台打开 JupyterLab
2. 直接把你的 `autodl_scripts` 文件夹拖进去
3. 或者用上传按钮

### 方式 B：用 `scp` 命令
```bash
# 在你的本地终端（不是 AutoDL 里）运行：
scp -P 33458 -r /path/to/python-leo/autodl_scripts root@connect.bjb1.seetacloud.com:/root/
```

---

## 第三步：在 AutoDL 上准备环境

```bash
# 1. 进入文件夹
cd /root/autodl_scripts

# 2. 创建虚拟环境
python3 -m venv qwen-env

# 3. 激活环境
source qwen-env/bin/activate

# 4. 安装必要的库（这个可能要花点时间）
pip install --upgrade pip
pip install transformers datasets peft accelerate torch sentencepiece protobuf tqdm bitsandbytes
```

---

## 第四步：开始微调！

```bash
# 确保在虚拟环境里
source qwen-env/bin/activate

# 运行微调脚本！
python train_lora.py
```

这个脚本会：
- 1. 自动下载 `Qwen/Qwen2.5-0.5B-Instruct` 模型
- 2. 准备公开的 `alpaca` 数据集
- 3. 用 LoRA 进行微调（只训练少量参数
- 4. 保存微调后的模型

---

## 第五步：测试微调后的效果！

```bash
# 运行测试脚本
python infer_lora.py
```

你会看到模型回答几个测试问题！

---

## ⚠️ 注意事项

### 1️⃣ 下载模型可能会有点慢
- 首次下载模型需要一些时间
- `Qwen2.5-0.5B` 很小，很快就能下完！

### 2️⃣ 监控显存
可以在另一个终端里运行 `nvidia-smi` 看显存占用

### 3️⃣ 不用强化学习！
我们这次只做 SFT 微调，先把这个流程跑通！

---

## 📝 如果遇到问题怎么办？

1. 先看 `autodl_scripts/README.md`
2. 如果有报错，把报错信息发给我看看
