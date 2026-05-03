
"""
MiniMind - 手搓大模型完整实现
可以直接运行这个脚本来训练模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple, List
from collections import Counter
import sys

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

# =============================================================================
# 第1部分：数据准备
# =============================================================================

def load_corpus(file_path: str) -> str:
    """加载语料文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

print("加载语料...")
corpus = load_corpus('chinese.txt')
print(f"语料总长度: {len(corpus)} 字符")
print(f"前200字符预览:\\n{corpus[:200]}\\n")

# =============================================================================
# 第2部分：Tokenizer
# =============================================================================

class Tokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        
        # 特殊token
        self.pad_token = "[PAD]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
    
    def train(self, text: str):
        """训练分词器 - 使用简单的字符级+最常见子词"""
        # 初始化：先添加特殊token
        tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        # 统计字符频率
        chars = list(text)
        char_counts = Counter(chars)
        
        # 添加所有出现的字符
        for char, _ in char_counts.most_common():
            if char not in tokens:
                tokens.append(char)
        
        # 如果还不够，尝试添加一些常见的二元组（简单的BPE）
        if len(tokens) < self.vocab_size:
            # 统计二元组
            bigrams = []
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i+1]
                bigrams.append(bigram)
            
            bigram_counts = Counter(bigrams)
            for bigram, _ in bigram_counts.most_common(self.vocab_size - len(tokens)):
                if bigram not in tokens:
                    tokens.append(bigram)
        
        # 构建词汇表
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.inv_vocab = {idx: token for idx, token in enumerate(tokens)}
        print(f"词汇表大小: {len(self.vocab)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本 - 使用贪心最大匹配"""
        # 首先获取所有token，按长度降序排列
        sorted_tokens = sorted(self.vocab.keys(), key=lambda x: (-len(x), x))
        # 移除特殊token，因为它们不需要参与匹配
        sorted_tokens = [t for t in sorted_tokens if t not in [self.pad_token, self.sos_token, self.eos_token, self.unk_token]]
        
        ids = []
        i = 0
        while i < len(text):
            matched = False
            # 尝试从最长的token开始匹配
            for token in sorted_tokens:
                if text.startswith(token, i):
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                # 没有匹配到，添加UNK
                ids.append(self.vocab[self.unk_token])
                i += 1
        
        if add_special_tokens:
            ids = [self.vocab[self.sos_token]] + ids + [self.vocab[self.eos_token]]
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """解码id序列为文本"""
        tokens = []
        for idx in ids:
            token = self.inv_vocab.get(idx, self.unk_token)
            if token not in [self.pad_token, self.sos_token, self.eos_token]:
                tokens.append(token)
        return ''.join(tokens)
    
    @property
    def pad_id(self) -> int:
        return self.vocab[self.pad_token]
    
    @property
    def sos_id(self) -> int:
        return self.vocab[self.sos_token]
    
    @property
    def eos_id(self) -> int:
        return self.vocab[self.eos_token]

print("训练Tokenizer...")
tokenizer = Tokenizer(vocab_size=4000)
tokenizer.train(corpus)

# 测试tokenizer
test_text = "数学是研究数量结构的学科"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"\\n测试Tokenizer:")
print(f"原文: {test_text}")
print(f"编码: {encoded}")
print(f"解码: {decoded}\\n")

# =============================================================================
# 第3部分：Transformer Decoder 组件
# =============================================================================

class RMSNorm(nn.Module):
    """RMSNorm归一化层"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """预计算RoPE的旋转频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用RoPE旋转位置编码"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim必须能被n_heads整除"
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # 多头切分
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # 应用RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])
        
        # 转置用于注意力计算
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        output = torch.matmul(attn_weights, xv)
        
        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# =============================================================================
# 第4部分：完整LLM模型
# =============================================================================

class MiniMind(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 6, n_heads: int = 8, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, hidden_dim=dim * 4) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
    
    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
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
        
        h = self.norm(h)
        logits = self.output(h)
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 40) -> List[int]:
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
            
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[-1]] = -float('inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
        
        return generated

# =============================================================================
# 第5部分：数据集和训练
# =============================================================================

class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, seq_len: int = 128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"语料token数: {len(self.tokens)}")
    
    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len - 1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokens[idx: idx + self.seq_len]
        targets = self.tokens[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def train_model(model: nn.Module, dataloader: DataLoader, epochs: int = 10, lr: float = 3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
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
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 每2轮生成一点文本看看效果
        if (epoch + 1) % 2 == 0:
            model.eval()
            prompt = "数学"
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            generated = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.7)
            print(f"  生成示例: {tokenizer.decode(generated)}")
            model.train()
    
    return model

# =============================================================================
# 第6部分：开始训练
# =============================================================================

# 超参数设置
SEQ_LEN = 64
BATCH_SIZE = 16
DIM = 256
N_LAYERS = 4
N_HEADS = 8
EPOCHS = 20

# 创建数据集和dataloader
print("\\n准备数据集...")
dataset = TextDataset(corpus, tokenizer, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Batch数量: {len(dataloader)}")

# 创建模型
print("\\n创建模型...")
model = MiniMind(
    vocab_size=len(tokenizer.vocab),
    dim=DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_seq_len=SEQ_LEN
)

n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {n_params:,}")
print(model)

# 开始训练
print("\\n开始训练...")
trained_model = train_model(model, dataloader, epochs=EPOCHS, lr=3e-4)

# =============================================================================
# 第7部分：保存模型
# =============================================================================

print("\\n保存模型...")
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'tokenizer_vocab': tokenizer.vocab,
    'tokenizer_inv_vocab': tokenizer.inv_vocab,
    'config': {
        'vocab_size': len(tokenizer.vocab),
        'dim': DIM,
        'n_layers': N_LAYERS,
        'n_heads': N_HEADS,
        'max_seq_len': SEQ_LEN
    }
}, 'minimind_model.pth')

print("模型已保存为 minimind_model.pth")

# =============================================================================
# 第8部分：测试生成
# =============================================================================

print("\\n" + "="*60)
print("测试模型生成效果:")
print("="*60)

def generate_text(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8):
    model.eval()
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    generated = model.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(generated)

test_prompts = [
    "数学",
    "几何学是",
    "中国古代",
    "微积分的",
]

for prompt in test_prompts:
    result = generate_text(prompt, max_new_tokens=80, temperature=0.7)
    print(f"Prompt: {prompt}")
    print(f"生成: {result}")
    print("-"*60)

print("\\n训练完成！")
