
"""
🚀 快速 SFT 微调（不需要交互输入，直接用演示数据）
现在支持加载刚才生成的 JSON 数据文件！
"""
import sys
import os
import json
import torch
from tqdm import tqdm
import time

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniMind, Tokenizer, SFTDataset


def load_json_data(json_filepath):
    """加载 JSON 格式的 SFT 数据"""
    print(f"📖 加载数据: {json_filepath}")
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ 加载了 {len(data)} 条数据")
    return data


def get_demo_data():
    """演示用的 SFT 数据（旧的备选）"""
    return [
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
        {
            "instruction": "你好",
            "input": "",
            "output": "你好！很高兴见到你，我可以帮助你解答问题或聊天。"
        },
        {
            "instruction": "什么是机器学习？",
            "input": "",
            "output": "机器学习是人工智能的一个分支，让计算机能够通过从数据中学习规律来执行任务，而不需要显式编程。"
        },
    ]


def train_sft(data=None, json_filepath=None):
    print("="*60)
    print("🚀 开始 SFT 监督微调")
    print("="*60)

    # 加载预训练模型
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "minimind_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    print(f"\n📦 加载预训练模型...")
    model = MiniMind(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = Tokenizer(config["vocab_size"])
    tokenizer.vocab = checkpoint["tokenizer_vocab"]
    tokenizer.inv_vocab = checkpoint["tokenizer_inv_vocab"]

    # 准备数据
    if json_filepath and os.path.exists(json_filepath):
        data = load_json_data(json_filepath)
    elif data is None:
        data = get_demo_data()
        print(f"📖 使用演示数据，共 {len(data)} 条")

    # 创建数据集
    dataset = SFTDataset(data, tokenizer, max_seq_len=64)
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器配置（SFT 用更小的学习率）
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"🚀 使用设备: {device}")

    # 训练
    epochs = 3
    print(f"\n🎯 开始训练 {epochs} 个 epoch...")

    start_time = time.time()
    model.train()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"  Training")
        for inputs, targets, loss_mask in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss_mask = loss_mask.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            # 计算 loss（只对 response 部分计算）
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            )
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch 完成，平均 Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\n✨ 训练完成！共耗时 {total_time/60:.1f} 分钟")

    # 保存模型
    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sft")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "minimind_sft.pth")

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer_vocab": tokenizer.vocab,
        "tokenizer_inv_vocab": tokenizer.inv_vocab,
        "config": config,
    }
    torch.save(final_checkpoint, save_path)
    print(f"\n💾 模型已保存到: {save_path}")

    return model, tokenizer, save_path


def test_model(model, tokenizer):
    """简单测试效果"""
    model.eval()
    device = next(model.parameters()).device

    print("\n" + "="*60)
    print("🧪 测试微调后的效果")
    print("="*60)

    test_prompts = ["你好", "什么是机器学习", "如何学习编程"]

    for prompt in test_prompts:
        with torch.no_grad():
            prompt_tokens = tokenizer.encode(prompt)
            generated = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.7)
            output_text = tokenizer.decode(generated)
            print(f"\n问: {prompt}")
            print(f"答: {output_text}")


def main():
    # 检查是否有数据文件
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    json_files = []

    if os.path.exists(data_dir):
        json_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json")
        ]

    if len(json_files) &gt; 0:
        print("📋 找到的数据文件:")
        for i, f in enumerate(json_files, 1):
            print(f"  {i}. {os.path.basename(f)}")

        choice = input("\n选择要使用的数据 (1-{}): ".format(len(json_files))).strip()

        if choice.isdigit() and 1 &lt;= int(choice) &lt;= len(json_files):
            selected_file = json_files[int(choice)-1]
            model, tokenizer, save_path = train_sft(json_filepath=selected_file)
        else:
            print("⚠️  无效选择，用演示数据")
            model, tokenizer, save_path = train_sft()
    else:
        print("⚠️  没有找到数据文件，先用演示数据")
        print("💡 提示: 可以先运行 download_sft_data.py 生成数据\n")
        model, tokenizer, save_path = train_sft()

    test_model(model, tokenizer)

    print("\n" + "="*60)
    print("🎉 SFT 微调全部完成！")
    print("="*60)
    print("""
你可以继续：
1. 测试更多对话：cd inference &amp;&amp; python test_sft.py
2. 准备你自己的数据进行更多微调！
""")


if __name__ == "__main__":
    main()

