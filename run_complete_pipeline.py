
"""
🎯 完整走一遍：预训练 → SFT → RL 流程
"""
import os
import sys

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))


def print_stage(stage, emoji, title):
    print("\n" + "="*60)
    print(f"{emoji} {stage}: {title}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("🚀 MiniMind 完整流程")
    print("="*60)
    print("""
1️⃣ 预训练 (Pre-train) - 现在做
2️⃣ 监督微调 (SFT) - 下一步
3️⃣ 强化学习 (RL) - 理解方向

我们一步步走！
    """)

    print("\n按回车键开始看各个阶段的效果...")
    input()

    # ========== 阶段 1: 预训练 ==========
    print_stage("1", "📚", "预训练 (Pre-train)")
    print("""
什么是预训练？
- 模型在大量无标注文本上学习
- 学会"猜下一个词"
- 这是你现在的 minimind_model.pth 已经完成的！

我们先看看预训练模型的效果...
""")
    input("\n按回车键看一下预训练模型的生成效果...")

    # 测试预训练模型
    import torch
    from src.model import MiniMind

    class SimpleTokenizer:
        def __init__(self, vocab, inv_vocab):
            self.vocab = vocab
            self.inv_vocab = inv_vocab
        def encode(self, text):
            return [self.vocab.get(ch, 0) for ch in text]
        def decode(self, tokens):
            return "".join([self.inv_vocab.get(tok, "?") for tok in tokens])

    # 加载预训练模型
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "minimind_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

    print("\n🤖 预训练模型 - 随便说点什么...")
    prompts = [
        "今天",
        "数学",
        "计算机",
    ]
    for p in prompts:
        with torch.no_grad():
            prompt_tokens = tokenizer.encode(p, add_special_tokens=False)
            generated = model.generate(prompt_tokens, max_length=50, temperature=0.7)
            print(f"\n你: {p}")
            print(f"模型: {tokenizer.decode(generated)}")

    print("\n" + "-"*60)
    print("📌 观察：预训练模型会连续说话，但不会回答问题！")
    print("-"*60)

    input("\n按回车键进入第 2 阶段：SFT 监督微调...")

    # ========== 阶段 2: SFT ==========
    print_stage("2", "🎯", "监督微调 (SFT - Supervised Fine-Tuning)")
    print("""
什么是 SFT？
- 用问答对训练模型，让模型学会听指令
- 比如: "什么是机器学习？" → "机器学习是..."
- 你看 sft/sft_train.py 可以做这件事！
""")

    print("\nSFT 的数据是这样的：")
    demo_data = [
        {
            "instruction": "什么是大语言模型？",
            "input": "",
            "output": "大语言模型是一种基于深度学习的人工智能系统，通过学习海量文本数据来预测下一个词，从而生成连贯的文本。"
        },
        {
            "instruction": "如何学习编程？",
            "input": "",
            "output": "学习编程建议从 Python 开始，因为它简单易学且用途广泛。多写代码、做项目是最快的成长方式。"
        },
    ]
    for item in demo_data:
        print(f"\n问: {item['instruction']}")
        print(f"答: {item['output']}")

    print("\n" + "-"*60)
    print("📌 SFT 之后模型就会回答问题了！")
    print("-"*60)

    input("\n按回车键看第 3 阶段：RL...")

    # ========== 阶段 3: RL ==========
    print_stage("3", "🧠", "强化学习 (RL - Reinforcement Learning)")
    print("""
什么是 RL？
- SFT 之后模型能说话，但可能说得不够好
- RL 用来对齐人类偏好，比如：
  - 让模型更有帮助
  - 更礼貌
  - 更安全
- 这个比较复杂，等你前两步搞明白了再深入！

简单流程：
1. 训练一个「奖励模型」(Reward Model)，给不同回答打分
2. 用 PPO 强化学习算法，让模型往高分方向改进
""")

    print("\n" + "="*60)
    print("🎉 完整流程讲完了！")
    print("="*60)
    print("""
现在你可以：
1. 去 sft/ 目录运行 sft_train.py 试试 SFT 微调
2. 然后用 inference/sft_inference.py 测试效果
3. 或者先看看文档再玩！
""")

