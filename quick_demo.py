
"""
MiniMind - 快速演示版本
使用小数据集快速训练和测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple, List
from collections import Counter

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============= 1. 语料准备 =============
# 使用一段关于数学的中文文本
corpus = """数学，是研究数量、结构以及空间等概念及其变化的一门学科，属于形式科学的一种。
数学利用抽象化和逻辑推理，从计数、计算、量度、对物体形状及运动的观察发展而成。
数学家们拓展这些概念，以公式化新的猜想，以及从选定的公理及定义出发，严谨地推导出一些定理。
基础数学的知识与运用是生活中不可或缺的一环。对数学基本概念的完善，早在古埃及、美索不达米亚及古印度历史上的古代数学文本便可观见。
在古希腊那里有更为严谨的处理。从那时开始，数学的发展便持续不断地小幅进展，至16世纪的文艺复兴时期，因为新的科学发现和数学革新两者的交互，致使数学的加速发展，直至今日。
数学在许多领域都有应用，包括科学、工程、医学、经济学和金融学等。
数学对这些领域的应用通常被称为应用数学，有时亦会激起新的数学发现，并导致全新学科的发展。
数学家也研究纯粹数学，就是数学本身的实质性内容，而不以任何实际应用为目标。
许多研究虽然以纯粹数学开始，但其过程中也发现许多可用之处。
几何学是数学的一个重要分支，它研究空间、形状、大小以及图形的性质。
微积分是数学的另一个重要分支，它研究变化、极限、导数、积分和无穷级数。
中国古代在数学方面有许多重要贡献，包括十进制、算盘、勾股定理的发现等。
"""

print(f"语料长度: {len(corpus)} 字符")

# ============= 2. Tokenizer =============
class Tokenizer:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        self.pad_token = "[PAD]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
    
    def train(self, text):
        tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        chars = list(text)
        char_counts = Counter(chars)
        
        for char, _ in char_counts.most_common():
            if char not in tokens:
                tokens.append(char)
        
        if len(tokens) < self.vocab_size:
            bigrams = []
            for i in range(len(chars) - 1):
                bigrams.append(chars[i] + chars[i+1])
            bigram_counts = Counter(bigrams)
            for bigram, _ in bigram_counts.most_common(self.vocab_size - len(tokens)):
                if bigram not in tokens:
                    tokens.append(bigram)
        
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.inv_vocab = {idx: token for idx, token in enumerate(tokens)}
        print(f"词汇表大小: {len(self.vocab)}")
    
    def encode(self, text, add_special_tokens=True):
        sorted_tokens = sorted(self.vocab.keys(), key=lambda x: (-len(x), x))
        sorted_tokens = [t for t in sorted_tokens if t not in [self.pad_token, self.sos_token, self.eos_token, self.unk_token]]
        
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for token in sorted_tokens:
                if text.startswith(token, i):
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(self.vocab[self.unk_token])
                i += 1
        
        if add_special_tokens:
            ids = [self.vocab[self.sos_token]] + ids + [self.vocab[self.eos_token]]
        return ids
    
    def decode(self, ids):
        tokens = []
        for idx in ids:
            token = self.inv_vocab.get(idx, self.unk_token)
            if token not in [self.pad_token, self.sos_token, self.eos_token]:
                tokens.append(token)
        return ''.join(tokens)

tokenizer = Tokenizer(vocab_size=1000)
tokenizer.train(corpus)

test_text = "数学是研究数量结构的学科"
encoded = tokenizer.encode(test_text)
print(f"\\n测试: {test_text}")
print(f"编码: {encoded}")
print(f"解码: {tokenizer.decode(encoded)}")

# ============= 3. Transformer 组件 =============
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, freqs_cis, mask=None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        output = torch.matmul(attn_weights, xv)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    
    def forward(self, x, freqs_cis, mask=None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))

# ============= 4. MiniMind 模型 =============
class MiniMind(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=2, n_heads=4, max_seq_len=64):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, dim * 4) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
    
    def forward(self, tokens, start_pos=0):
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask[None, None, :, :]
        
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        return self.output(self.norm(h))
    
    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=50, temperature=0.8):
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        generated = prompt_tokens.copy()
        
        for _ in range(max_new_tokens):
            if tokens.shape[1] > self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len:]
            logits = self(tokens)
            next_token_logits = logits[0, -1, :]
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
        
        return generated

# ============= 5. 数据集 =============
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"语料token数: {len(self.tokens)}")
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        inputs = self.tokens[idx: idx + self.seq_len]
        targets = self.tokens[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# ============= 6. 训练 =============
SEQ_LEN = 32
BATCH_SIZE = 8
DIM = 128
N_LAYERS = 2
N_HEADS = 4
EPOCHS = 50

dataset = TextDataset(corpus, tokenizer, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MiniMind(
    vocab_size=len(tokenizer.vocab),
    dim=DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_seq_len=SEQ_LEN
)

n_params = sum(p.numel() for p in model.parameters())
print(f"\\n模型参数量: {n_params:,}")

def train_model(model, dataloader, epochs=50, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model = model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            inputs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            prompt = "数学"
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            generated = model.generate(prompt_tokens, max_new_tokens=40, temperature=0.7)
            print(f"  生成: {tokenizer.decode(generated)}")
            model.train()
    
    return model

print("\\n开始训练...")
model = train_model(model, dataloader, epochs=EPOCHS)

# ============= 7. 测试生成 =============
print("\\n" + "="*60)
print("测试生成效果")
print("="*60)

test_prompts = [
    "数学",
    "几何学",
    "微积分",
    "中国古代",
]

for prompt in test_prompts:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    generated = model.generate(prompt_tokens, max_new_tokens=60, temperature=0.7)
    print(f"\\n输入: {prompt}")
    print(f"生成: {tokenizer.decode(generated)}")

print("\\n" + "="*60)
print("演示完成！")
