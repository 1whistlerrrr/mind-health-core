
# 🐋 AutoDL 小模型微调完整方案

## 🎯 我们的目标
- 在 AutoDL 上租一个便宜的 GPU 机器
- 选择一个超强的小模型（中文好、显存小
- 用 LoRA 做微调（又快又省
- 测试微调效果，体验完整流程

---

## 💰 第一步：去 AutoDL 租机器

### 推荐配置（性价比最高！
| 配置项 | 推荐选择 | 价格 |
|-------|----------|------|
| GPU | RTX 3060 / RTX 3070 / RTX 3080 | 0.5 - 2 元/小时 |
| 显存 | ≥ 8GB (最好 12GB) | - |
| 镜像 | PyTorch 2.x (选预装 CUDA 的) | 免费 |
| 存储 | 50GB - 100GB | 很便宜 |

### 大概多少钱
- 完整流程跑下来，1-2小时搞定！
- 总花费：5-10 块钱！学生党友好！

---

## 🤖 第二步：模型选择（我们的推荐

### 首选：Qwen2.5-0.5B-Instruct
- **参数量**：0.5B (5亿参数！
- **显存需求**：Q4量化后 ~0.4GB
- **亮点**：
  - 中文特别好（训练语料 65% 中文
  - 极小极快
  - 已经会问答（Instruct 版本
  - Apache 2.0 协议（商用免费！

### 备选：Qwen3.5-0.8B-Instruct
- **参数量**：0.8B
- **2026年3月刚发布！更强！
- 中文依然很好，效果比 0.5B 更好

---

## 📝 第三步：完整微调步骤

### 1️⃣ 1. 准备环境
登录你的 AutoDL 实例后：
```bash
# 创建虚拟环境
python3 -m venv qwen-env
source qwen-env/bin/activate

# 安装必要的库
pip install --upgrade pip
pip install transformers datasets peft accelerate torch sentencepiece protobuf
```

### 2️⃣ 2. 准备微调数据
我们用现成的公开中文微调数据集！推荐：
- `huoshan/alpaca-zh` - Alpaca 中文翻译版
- `wikipedia-zh` - 更有数据，但可以先用小的

### 3️⃣ 3. 加载并 LoRA 微调 Qwen2.5-0.5B
我们用 LoRA（Low-Rank Adaptation）微调！
- 只需训练少量参数（通常不到 1%
- 极快！
- 显存占用小

---

## 🔧 接下来我们为你写完整的 Python 脚本！
