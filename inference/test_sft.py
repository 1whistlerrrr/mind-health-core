
"""
🧪 测试 SFT 微调后的模型
"""
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniMind


class SimpleTokenizer:
    def __init__(self, vocab, inv_vocab):
        self.vocab = vocab
        self.inv_vocab = inv_vocab

    def encode(self, text):
        return [self.vocab.get(ch, 0) for ch in text]

    def decode(self, tokens):
        return "".join([self.inv_vocab.get(tok, "?") for tok in tokens])


def load_model():
    sft_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sft", "minimind_sft.pth")
    original_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "minimind_model.pth")

    if os.path.exists(sft_path):
        print("📦 加载 SFT 微调后的模型...")
        checkpoint = torch.load(sft_path, map_location="cpu")
    else:
        print("⚠️  没找到 SFT 模型，使用原始模型...")
        checkpoint = torch.load(original_path, map_location="cpu")

    config = checkpoint["config"]
    model = MiniMind(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(checkpoint["tokenizer_vocab"], checkpoint["tokenizer_inv_vocab"])
    return model, tokenizer


def test_generation():
    print("="*60)
    print("🧪 测试 SFT 微调后的模型")
    print("="*60)

    model, tokenizer = load_model()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    test_prompts = ["你好", "什么是机器学习", "如何学习编程", "什么是大语言模型"]

    print("\n🤖 生成结果：")
    for prompt in test_prompts:
        with torch.no_grad():
            prompt_tokens = tokenizer.encode(prompt)
            generated = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.7)
            output = tokenizer.decode(generated)
            print(f"\n你: {prompt}")
            print(f"模型: {output}")

    print("\n" + "="*60)
    print("💡 提示：因为只用了 4 条演示数据微调，效果有限！")
    print("    想得到更好的效果，需要准备更多高质量的数据！")
    print("="*60)


if __name__ == "__main__":
    test_generation()

