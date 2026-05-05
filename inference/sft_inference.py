
"""
🎯 使用 SFT 微调后的模型进行推理
"""
import torch
import sys
import os

# 添加 src 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import MiniMind


class SimpleTokenizer:
    """简化的 Tokenizer，用于加载"""
    def __init__(self, vocab, inv_vocab):
        self.vocab = vocab
        self.inv_vocab = inv_vocab

    def encode(self, text):
        """简单的编码（单字符）"""
        return [self.vocab.get(ch, 0) for ch in text]

    def decode(self, tokens):
        """简单的解码"""
        return "".join([self.inv_vocab.get(tok, "?") for tok in tokens])


def load_sft_model():
    """加载 SFT 模型"""
    sft_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'sft', 'minimind_sft.pth')
    original_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'minimind_model.pth')

    # 优先使用 SFT 模型，如果没有就用原始模型
    if os.path.exists(sft_path):
        print("📦 加载 SFT 微调后模型...")
        checkpoint_path = sft_path
    else:
        print("⚠️  没找到 SFT 模型，使用原始预训练模型...")
        checkpoint_path = original_path

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    model = MiniMind(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = SimpleTokenizer(checkpoint['tokenizer_vocab'], checkpoint['tokenizer_inv_vocab'])

    print(f"✅ 加载完成！参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def interactive_chat():
    """交互式聊天"""
    model, tokenizer = load_sft_model()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)

    print("\n" + "="*60)
    print("🤖 MiniMind 聊天")
    print("="*60)
    print("输入 'quit' 退出\n")

    while True:
        prompt = input("你: ").strip()

        if prompt.lower() == 'quit':
            break

        if not prompt:
            continue

        with torch.no_grad():
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            generated = model.generate(prompt_tokens, max_length=100, temperature=0.7)
            response = tokenizer.decode(generated)
            print(f"\n模型: {response}\n")


if __name__ == "__main__":
    interactive_chat()

