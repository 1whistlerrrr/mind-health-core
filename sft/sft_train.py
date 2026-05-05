
"""
🎯 MiniMind 监督微调 (SFT) 训练脚本
让你的模型学会回答问题！
"""
import sys
import os
import json
import torch
from tqdm import tqdm
import time

# 添加 src 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import MiniMind, Tokenizer, SFTDataset


def load_model():
    """加载预训练模型"""
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'minimind_model.pth')

    print("📦 加载预训练模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['config']

    print(f"  模型配置: {config}")

    # 创建模型
    model = MiniMind(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len']
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    # 创建 tokenizer
    tokenizer = Tokenizer(config['vocab_size'])
    tokenizer.vocab = checkpoint['tokenizer_vocab']
    tokenizer.inv_vocab = checkpoint['tokenizer_inv_vocab']

    return model, tokenizer, config


def get_demo_data():
    """获取演示数据（如果没有准备好数据，先玩一下）"""
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
        }
    ]


def train_sft(model, tokenizer, data, config):
    """SFT 训练"""

    # 准备输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'sft')
    os.makedirs(output_dir, exist_ok=True)

    # 超参数设置（为 Mac 优化）
    learning_rate = 1e-5  # 微调学习率更小
    batch_size = 4
    epochs = 3
    weight_decay = 0.01

    print("\n" + "="*60)
    print("🎯 开始监督微调 (SFT)")
    print("="*60)
    print(f"  数据量: {len(data)} 条问答")
    print(f"  学习率: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print("="*60 + "\n")

    # 创建数据集
    dataset = SFTDataset(data, tokenizer, max_seq_len=64)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 设备
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    print(f"🚀 使用设备: {device}\n")

    # 估算时间
    total_steps = len(dataloader) * epochs
    print(f"⏱️  总步数: {total_steps}")
    print(f"  预计时间: 5-10 分钟 (取决于硬件)\n")

    # 训练循环
    model.train()

    start_time = time.time()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0

        # 进度条
        pbar = tqdm(dataloader, desc=f"  Training")

        for step, (inputs, targets, loss_mask) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss_mask = loss_mask.to(device)

            optimizer.zero_grad()

            # 前向传播
            logits = model(inputs)

            # 计算 loss（只对 response 部分计算）
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none'
            )
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch 完成，Avg Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\n✨ SFT 训练完成！耗时: {total_time/60:.1f} 分钟")

    # 保存模型
    print("\n💾 保存模型...")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'tokenizer_inv_vocab': tokenizer.inv_vocab,
        'config': {
            'vocab_size': config['vocab_size'],
            'dim': config['dim'],
            'n_layers': config['n_layers'],
            'n_heads': config['n_heads'],
            'max_seq_len': config['max_seq_len']
        }
    }
    save_path = os.path.join(output_dir, 'minimind_sft.pth')
    torch.save(final_checkpoint, save_path)
    print(f"✅ 模型已保存至: {save_path}")

    return save_path


def test_model(model, tokenizer, device):
    """测试一下微调后的模型"""
    model.eval()
    print("\n" + "="*60)
    print("🧪 简单测试一下微调效果")
    print("="*60)

    test_prompts = [
        "你好",
        "什么是机器学习",
        "如何学习编程"
    ]

    for prompt in test_prompts:
        with torch.no_grad():
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            generated = model.generate(prompt_tokens, max_length=50, temperature=0.7)
            output_text = tokenizer.decode(generated)
            print(f"\nQ: {prompt}")
            print(f"A: {output_text}")


def main():
    # 加载模型
    model, tokenizer, config = load_model()

    # 问用户用什么数据
    print("\n📊 数据准备")
    print("="*60)
    print("1. 使用演示数据 (先试试效果)")
    print("2. 我准备好了自己的 JSON 数据")

    choice = input("\n请选择 (1/2): ").strip()

    if choice == "1":
        data = get_demo_data()
        print(f"✅ 使用演示数据，共 {len(data)} 条")
    else:
        json_path = input("请输入 JSON 数据文件路径: ").strip()
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 已加载 {len(data)} 条数据")
        else:
            print("❌ 文件不存在，使用演示数据")
            data = get_demo_data()

    # 开始训练
    save_path = train_sft(model, tokenizer, data, config)

    # 测试效果
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    test_model(model, tokenizer, device)

    print("\n" + "="*60)
    print("🎉 SFT 微调全部完成！")
    print(f"   模型保存在: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()

