
"""
MiniMind 完整模型实现
包含训练、保存、加载、推理、SFT等功能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple, List
from collections import Counter
import os
import json
from tqdm import tqdm
import time

from config import Config


# ===========================================
# 1. Tokenizer
# ===========================================
class Tokenizer:
    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        
        # 特殊符号
        self.pad_token = "[PAD]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
    
    def train(self, text: str):
        """训练分词器"""
        # 特殊符号
        tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        # 统计字符频率
        chars = list(text)
        char_counts = Counter(chars)
        
        # 添加字符
        for char, _ in char_counts.most_common():
            if char not in tokens:
                tokens.append(char)
        
        # 如果词表还没满，添加常用二元组
        if len(tokens) &lt; self.vocab_size:
            bigrams = []
            for i in range(len(chars) - 1):
                bigrams.append(chars[i] + chars[i+1])
            
            bigram_counts = Counter(bigrams)
            for bigram, _ in bigram_counts.most_common(self.vocab_size - len(tokens)):
                if bigram not in tokens:
                    tokens.append(bigram)
        
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.inv_vocab = {idx: token for idx, token in enumerate(tokens)}
        print(f"✅ Tokenizer训练完成，词表大小: {len(self.vocab)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -&gt; List[int]:
        """编码文本到token ids"""
        # 按长度降序排序token
        sorted_tokens = sorted(self.vocab.keys(), key=lambda x: (-len(x), x))
        sorted_tokens = [t for t in sorted_tokens if t not in 
                         [self.pad_token, self.sos_token, self.eos_token, self.unk_token]]
        
        ids = []
        i = 0
        while i &lt; len(text):
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
    
    def decode(self, ids: List[int]) -&gt; str:
        """解码token ids到文本"""
        tokens = []
        for idx in ids:
            token = self.inv_vocab.get(idx, self.unk_token)
            if token not in [self.pad_token, self.sos_token, self.eos_token]:
                tokens.append(token)
        return ''.join(tokens)
    
    def save(self, path: str):
        """保存tokenizer"""
        data = {
            "vocab": self.vocab,
            "inv_vocab": self.inv_vocab,
            "vocab_size": self.vocab_size
        }
        torch.save(data, path)
        print(f"✅ Tokenizer已保存到: {path}")
    
    @classmethod
    def load(cls, path: str) -&gt; 'Tokenizer':
        """加载tokenizer"""
        data = torch.load(path)
        tokenizer = cls(data["vocab_size"])
        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = data["inv_vocab"]
        print(f"✅ Tokenizer已从 {path} 加载")
        return tokenizer


# ===========================================
# 2. Transformer 组件
# ===========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _norm(self, x: torch.Tensor) -&gt; torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -&gt; torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -&gt; torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -&gt; Tuple[torch.Tensor, torch.Tensor]:
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
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -&gt; torch.Tensor:
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
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -&gt; torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -&gt; torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


# ===========================================
# 3. MiniMind 完整模型
# ===========================================
class MiniMind(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 4, 
                 n_heads: int = 8, max_seq_len: int = 64):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # 词嵌入
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim=dim * 4)
            for _ in range(n_layers)
        ])
        
        # 输出
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # 预计算RoPE频率
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
    
    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -&gt; torch.Tensor:
        batch_size, seq_len = tokens.shape
        
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        
        mask = None
        if seq_len &gt; 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask[None, None, :, :]
        
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        return self.output(self.norm(h))
    
    @torch.no_grad()
    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 100, 
                 temperature: float = 0.8, top_k: int = 40) -&gt; List[int]:
        """自回归生成文本"""
        self.eval()
        device = next(self.parameters()).device
        
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        generated = prompt_tokens.copy()
        
        for _ in range(max_new_tokens):
            if tokens.shape[1] &gt; self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len:]
            
            logits = self(tokens)
            next_token_logits = logits[0, -1, :]
            
            if temperature &gt; 0:
                next_token_logits = next_token_logits / temperature
            
            if top_k &gt; 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits &lt; v[-1]] = -float('inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
        
        return generated
    
    def save(self, path: str):
        """保存模型（只保存权重）"""
        torch.save(self.state_dict(), path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ 模型权重已保存到: {path} ({size_mb:.2f} MB)")
    
    @classmethod
    def load(cls, path: str, config: Config) -&gt; 'MiniMind':
        """从权重加载模型"""
        model = cls(
            vocab_size=config.model.vocab_size,
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            max_seq_len=config.model.max_seq_len
        )
        model.load_state_dict(torch.load(path))
        print(f"✅ 模型已从 {path} 加载")
        return model
    
    def save_pretrained(self, dir_path: str):
        """完整保存模型（包含配置，像HuggingFace那样）"""
        os.makedirs(dir_path, exist_ok=True)
        
        # 保存权重
        self.save(os.path.join(dir_path, "model.pt"))
        
        # 保存配置
        config_dict = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_seq_len": self.max_seq_len
        }
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 完整模型已保存到目录: {dir_path}")
    
    @classmethod
    def from_pretrained(cls, dir_path: str) -&gt; 'MiniMind':
        """从目录加载完整模型"""
        # 加载配置
        with open(os.path.join(dir_path, "config.json"), "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # 创建模型
        model = cls(**config_dict)
        
        # 加载权重
        model.load_state_dict(torch.load(os.path.join(dir_path, "model.pt")))
        
        print(f"✅ 模型已从 {dir_path} 加载")
        return model


# ===========================================
# 4. 数据集
# ===========================================
class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"📚 数据集准备完成，总token数: {len(self.tokens)}")
    
    def __len__(self) -&gt; int:
        return max(0, len(self.tokens) - self.seq_len - 1)
    
    def __getitem__(self, idx: int) -&gt; Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokens[idx: idx + self.seq_len]
        targets = self.tokens[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# ===========================================
# 5. SFT 数据集（用于监督微调）
# ===========================================
class SFTDataset(Dataset):
    """监督微调数据集 - 用于指令跟随"""
    def __init__(self, data: List[dict], tokenizer: Tokenizer, max_seq_len: int = 64):
        """
        Args:
            data: [{"instruction": "...", "input": "...", "output": "..."}, ...]
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        
        for item in data:
            prompt = f"问题: {item['instruction']}\n{item.get('input', '')}\n回答: "
            response = item['output']
            
            # 编码
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            # 拼接
            full_tokens = (
                [tokenizer.vocab[tokenizer.sos_token]] + 
                prompt_tokens + 
                response_tokens + 
                [tokenizer.vocab[tokenizer.eos_token]]
            )
            
            if len(full_tokens) &gt; max_seq_len:
                continue
            
            self.examples.append({
                "full_tokens": full_tokens,
                "prompt_length": len(prompt_tokens) + 1  # +1 for SOS
            })
        
        print(f"📚 SFT数据集准备完成，样本数: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens = example["full_tokens"]
        prompt_len = example["prompt_length"]
        
        # padding
        padding_len = self.max_seq_len - len(tokens)
        padded = tokens + [self.tokenizer.vocab[self.tokenizer.pad_token]] * padding_len
        
        inputs = padded[:-1]
        targets = padded[1:]
        
        # 只对response部分计算loss（忽略prompt和padding）
        loss_mask = [0] * len(targets)
        for i in range(prompt_len, len(tokens) - 1):
            loss_mask[i] = 1
        
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float)
        )


# ===========================================
# 6. 训练工具
# ===========================================
def estimate_training_time(dataloader: DataLoader, epochs: int, batch_time_sec: float = 0.1):
    """
    估算训练时间（非常实用的功能！）
    
    Args:
        batch_time_sec: 每个batch的预估时间（秒），可以先用小批量测试
    """
    num_batches = len(dataloader)
    total_batches = num_batches * epochs
    total_sec = total_batches * batch_time_sec
    
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    
    print(f"\n⏱️ 训练时间估算:")
    print(f"  - Batch数/epoch: {num_batches}")
    print(f"  - 总Batch数: {total_batches}")
    print(f"  - 预估时间: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"  - (假设每个batch {batch_time_sec:.2f}秒)\n")


def train(model: MiniMind, dataloader: DataLoader, config: Config, tokenizer: Tokenizer, output_dir: str = "./output"):
    """完整训练流程"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train.epochs)
    
    # 设备
    device_str = config.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    model = model.to(device)
    print(f"🚀 训练开始，设备: {device}")
    
    # 估算时间
    estimate_training_time(dataloader, config.train.epochs)
    
    # 训练循环
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config.train.epochs):
        model.train()
        total_loss = 0
        
        # 进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.train.epochs}")
        
        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (config.train.epochs - epoch - 1)
        
        print(f"\n✨ Epoch {epoch+1} 完成 | Avg Loss: {avg_loss:.4f} | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        # 定期生成样本
        if (epoch + 1) % config.train.eval_every == 0:
            model.eval()
            prompt = "数学"
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            generated = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.7)
            print(f"📝 生成样本: {tokenizer.decode(generated)}\n")
        
        # 保存检查点
        if (epoch + 1) % config.train.save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            model.save(checkpoint_path)
            
            # 同时保存tokenizer
            tokenizer.save(os.path.join(output_dir, "tokenizer.pt"))
            
            # 保存最佳模型
            if avg_loss &lt; best_loss:
                best_loss = avg_loss
                best_path = os.path.join(output_dir, "best_model.pt")
                model.save(best_path)
                print(f"🏆 新最佳模型已保存 (Loss: {best_loss:.4f})")
    
    # 最终保存
    final_path = os.path.join(output_dir, "final_model.pt")
    model.save(final_path)
    tokenizer.save(os.path.join(output_dir, "tokenizer.pt"))
    
    # 保存完整版本（像HuggingFace那样）
    model.save_pretrained(os.path.join(output_dir, "minimind_final"))
    
    total_time = time.time() - start_time
    print(f"\n🎉 训练完成！总耗时: {total_time/60:.1f} 分钟")


# ===========================================
# 7. SFT训练（监督微调）
# ===========================================
def train_sft(model: MiniMind, sft_dataloader: DataLoader, config: Config, output_dir: str = "./sft_output"):
    """监督微调 - 在预训练基础上继续训练"""
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 微调学习率更小
    device = next(model.parameters()).device
    
    model.train()
    for epoch in range(3):  # SFT通常不需要太多epoch
        total_loss = 0
        pbar = tqdm(sft_dataloader, desc=f"SFT Epoch {epoch+1}")
        
        for inputs, targets, loss_mask in pbar:
            inputs, targets, loss_mask = inputs.to(device), targets.to(device), loss_mask.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # 带mask的loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"SFT Loss": f"{loss.item():.4f}"})
        
        print(f"SFT Epoch {epoch+1} Avg Loss: {total_loss/len(sft_dataloader):.4f}")
    
    # 保存SFT模型
    model.save_pretrained(os.path.join(output_dir, "minimind_sft"))
    print("✅ SFT模型已保存")


# ===========================================
# 8. 推理工具
# ===========================================
class InferencePipeline:
    """推理管道 - 方便使用"""
    def __init__(self, model_dir: str, device: str = "auto"):
        # 加载模型和tokenizer
        self.model = MiniMind.from_pretrained(model_dir)
        self.tokenizer = Tokenizer.load(os.path.join(model_dir, "tokenizer.pt"))
        
        # 设备
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.model = self.model.to(torch.device(device))
        self.model.eval()
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -&gt; str:
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = self.model.generate(prompt_tokens, max_new_tokens, temperature)
        return self.tokenizer.decode(generated)
