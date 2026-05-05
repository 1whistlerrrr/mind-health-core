
# 🤖 SFT 监督微调深度详解

## 目录
1. **高质量微调数据集推荐**
2. **SFT 是如何一步一步实现的**
3. **LoRA 是什么？它算 SFT 吗？如何实现？**

---

## 1️⃣ 高质量微调数据集推荐

### 中文可用的 SFT 数据集（开源免费）

| 数据集 | 大小 | 说明 |
|--------|------|------|
| **alpaca-zh** | ~5万条 | Alpaca 中文翻译版 |
| **Firefly-zh** | ~10万条 | 中文开源微调数据集 |
| **bell** | ~10万条 | 中文双语对话数据集 |
| **BELLE-zh** | 大规模 | 中文开源对话数据集 |
| **ShareGPT-zh** | 中等规模 | 中文的 ShareGPT 对话 |

### 特别适合小模型的小而精的数据集（推荐你试试！）

对于 6.2M 这种小模型，**不需要太多数据！**几百到几千条就够了！

| 数据集 | 大小 | 说明 |
|--------|------|------|
| **tiny_sharegpt_zh** | ~1000条 | 小而精的中文对话 |
| **BELLE_1k** | 1000条 | BELLE 的 1000 条精选 |
| **Alpaca-zh 1k** | 1000条 | Alpaca 中文 1000 条精选 |

### 最推荐你试试的（超级快速就能上手！）

- 🚀 **HuggingFace 上搜索**: `alpaca-zh` 或 `firefly`
- 💡 你只需要 100-1000 条数据，效果就会很明显！
- 📂 把它们整理成我们刚才用的 `{instruction, input, output}` 格式就行！

---

## 2️⃣ SFT 是如何一步一步实现的？

### 📝 SFT 的数据格式（先看明白数据）

SFT 数据必须是 **问答对** 格式：

```python
[
  {
    "instruction": "给我讲个笑话",  # 指令
    "input": "",                   # 可选的额外输入
    "output": "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25！"  # 希望的答案
  },
  {
    "instruction": "什么是机器学习？",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  }
]
```

### 🔧 完整 SFT 实现步骤

#### 第 1 步：把问答对组装成训练用的样本

我们需要把问题和答案拼起来：

```python
# 原始问答对
"instruction": "你好",
"input": "",
"output": "你好！有什么可以帮你的吗？"

# 组装后的完整样本
prompt = "问题: 你好 回答:"
answer = "你好！有什么可以帮你的吗？"
full_text = "[SOS]问题: 你好 回答:你好！有什么可以帮你的吗？[EOS]"
```

#### 第 2 步：准备 Loss Mask（关键！）

**重点：我们只希望模型学习怎么回答，不希望它学习怎么提问！**

```python
完整文本: [SOS]问题: 你好 回答:你好！有什么可以帮你的吗？[EOS]
          ↑______ prompt部分____↑↑_______ response部分____↑
          ↓                      ↓
         Loss=0                Loss=有梯度
```

实现上就是用一个 mask 数组：

```python
token_ids:  [1,   5,  100, 200, 300, 400, 500, 600, 2]
labels:     [5,  100, 200, 300, 400, 500, 600, 2,   0]
loss_mask:  [0,   0,   0,   0,   1,   1,   1,  1,   0]
              ↑              ↑_______________↑
              只对这里计算梯度
```

#### 第 3 步：开始训练（和预训练类似，但更小的学习率！）

```python
# 加载预训练模型
model = load_pretrained_model()

# 关键区别：SFT 用的学习率更小！
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 不是 3e-4 哦！

for epoch in range(3):
    for inputs, targets, loss_mask in dataloader:
        # 前向传播
        logits = model(inputs)

        # 计算 loss，但只对 response 部分计算！
        loss = cross_entropy_with_mask(logits, targets, loss_mask)

        # 反向传播和更新
        loss.backward()
        optimizer.step()
```

### 🎯 完整 SFT 实现（代码版）

```python
import torch
import torch.nn as nn

class SimpleSFTTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_sample(self, instruction, output):
        """准备一个 SFT 样本"""
        prompt = f"问题: {instruction} 回答:"
        prompt_tokens = self.tokenizer.encode(prompt)
        answer_tokens = self.tokenizer.encode(output)

        # 拼起来
        full_tokens = (
            [self.tokenizer.sos_token_id] +
            prompt_tokens +
            answer_tokens +
            [self.tokenizer.eos_token_id]
        )

        # 准备 loss mask：只对答案部分计算梯度
        loss_mask = [0.0] * (len(prompt_tokens) + 1) + [1.0] * len(answer_tokens) + [0.0]

        return full_tokens, loss_mask

    def train(self, data, epochs=3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            total_loss = 0
            for sample in data:
                full_tokens, loss_mask = self.prepare_sample(
                    sample["instruction"], sample["output"]
                )

                inputs = torch.tensor([full_tokens[:-1]])
                targets = torch.tensor([full_tokens[1:]])
                mask = torch.tensor([loss_mask[1:]])

                # 前向传播
                logits = self.model(inputs)

                # 计算 loss，只用 mask 的部分
                loss = self.compute_masked_loss(logits, targets, mask)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Loss={total_loss/len(data):.4f}")

    def compute_masked_loss(self, logits, targets, mask):
        """只计算 masked 部分的 loss"""
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none"
        )
        return (loss * mask.view(-1)).sum() / mask.sum()
```

---

## 3️⃣ LoRA 是什么？它算 SFT 吗？如何实现？

### 🔍 LoRA 是什么？

**LoRA = Low-Rank Adaptation（低秩适应）**

简单理解：
- **全量 SFT**：更新模型的所有权重
- **LoRA SFT**：只更新模型里的一小部分（用低秩矩阵“补丁”）

---

### ✅ LoRA 算 SFT 吗？

**答案：算！**

- SFT = 监督微调（指的是任务目标）
- LoRA = 一种参数高效的微调方法（指的是具体技术）

**关系：**

```
SFT（监督微调）
├── 全量 SFT（更新所有参数）
└── LoRA SFT（用低秩矩阵，只更新小部分参数）
```

---

### 🔧 LoRA 的原理（用一张图说清楚）

想象模型原本的权重是这样的：

```
原来的权重:  W (256×256 = 65536 个参数)
↓
LoRA 的做法:  W + A×B
              ↑   ↑ ↑
              |   | └── LoRA 的小矩阵 B (256×r)
              |   └── LoRA 的小矩阵 A (r×256)
              └── 原来的 W，保持不动
```

**关键点：**
- 我们只训练 A 和 B！
- r 叫做「秩」(rank)，通常取 8 或 16
- 这样需要训练的参数量只有 2×256×r = 4096 或 8192！（超级小！）

---

### 💻 极简 LoRA 实现代码（你也可以写一个！）

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """一个简单的 LoRA 层"""
    def __init__(self, original_layer, r=8):
        super().__init__()
        self.original_layer = original_layer
        self.r = r

        # 把原来的层冻住
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA 的小矩阵
        self.lora_A = nn.Linear(self.original_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.original_layer.out_features, bias=False)

        # 初始化 LoRA 矩阵为 0
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 输出 = 原始输出 + LoRA 输出
        original_out = self.original_layer(x)
        lora_out = self.lora_B(self.lora_A(x))
        return original_out + lora_out


def add_lora_to_attention(model, r=8):
    """把 LoRA 加到注意力的 QKV 层"""
    for layer in model.layers:
        # 把 Q, K, V 层换成 LoRA 层
        layer.attention.wq = LoRALayer(layer.attention.wq, r=r)
        layer.attention.wk = LoRALayer(layer.attention.wk, r=r)
        layer.attention.wv = LoRALayer(layer.attention.wv, r=r)
```

---

### 🎯 你的情况：用不用 LoRA？

**因为你的模型只有 6.2M，**：
- 全量微调完全没问题，**不需要 LoRA！**
- 等你以后玩 7B、13B 大模型时，再用 LoRA 吧！

---

## 📝 总结

### SFT 要点
1. SFT = 用问答对微调
2. 关键：只对回答部分计算梯度（Loss Mask）
3. 学习率要比预训练小：1e-5 而不是 3e-4

### LoRA 要点
1. LoRA = 低秩适应，只用小矩阵
2. LoRA 是 SFT 的一种（参数高效版）
3. 你的模型太小，不用 LoRA 也行

### 你现在可以做的
1. 去 HuggingFace 下载点小数据集（比如 alpaca-zh）
2. 整理成 `{instruction, input, output}` 格式
3. 用 `sft/quick_sft.py` 的代码跑一下！

---

## 🚀 快速尝试高质量数据集的脚本（给你准备好了！）

让我们去 sft 文件夹里创建一个可以下载公开数据的脚本！
