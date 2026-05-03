
"""
SFT（监督微调）示例脚本
展示如何在预训练模型基础上进行微调
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from model import MiniMind, Tokenizer, SFTDataset, train_sft, InferencePipeline
import torch


def main():
    print("\n" + "="*60)
    print("🎯 MiniMind SFT 示例")
    print("="*60)
    
    # 1. 准备SFT数据
    print("\n📚 准备SFT数据...")
    sft_data = [
        {
            "instruction": "什么是数学？",
            "input": "",
            "output": "数学是研究数量、结构、变化以及空间等概念的学科。它使用抽象化和逻辑推理，从计数、计算、量度和对物体形状及运动的观察中发展而来。"
        },
        {
            "instruction": "勾股定理讲了什么？",
            "input": "",
            "output": "勾股定理是数学中一个重要的几何定理。它指出，在一个直角三角形中，两条直角边的平方和等于斜边的平方。也就是说，如果a和b是直角边，c是斜边，那么a² + b² = c²。"
        },
        {
            "instruction": "中国古代有哪些数学成就？",
            "input": "",
            "output": "中国古代有很多重要的数学成就。首先是十进制计数法的发明，这是数学发展的基础。其次是算盘的发明，它是世界上最早的计算工具之一。另外，中国数学家还发现了勾股定理，比西方早了几百年。"
        },
        {
            "instruction": "什么是几何学？",
            "input": "",
            "output": "几何学是数学的一个重要分支，主要研究空间、形状、大小以及图形的性质。它包括平面几何、立体几何、解析几何等多个领域。几何学不仅在数学中重要，在建筑、设计、艺术等方面也有广泛应用。"
        },
        {
            "instruction": "什么是微积分？",
            "input": "",
            "output": "微积分是数学的重要分支，主要研究极限、微分、积分和无穷级数。它由牛顿和莱布尼茨在17世纪独立发明。微积分在物理学、工程学、经济学等领域有广泛应用，是现代科学和技术的基础工具。"
        },
        {
            "instruction": "为什么数学很重要？",
            "input": "",
            "output": "数学非常重要，因为它是科学和技术的基础。从简单的购物算账，到复杂的火箭发射，数学无处不在。它能培养逻辑思维能力，帮助我们分析和解决问题。数学还能锻炼抽象思维，让我们看到事物之间的本质联系。"
        },
        {
            "instruction": "1+1等于几？",
            "input": "",
            "output": "1+1等于2。这是数学中最基本的加法运算，也是整个数学体系的基础。虽然看起来简单，但它蕴含了数学的核心概念。"
        },
        {
            "instruction": "2+3等于几？",
            "input": "",
            "output": "2+3等于5。这是基础的加法运算。"
        },
        {
            "instruction": "什么是实数？",
            "input": "",
            "output": "实数是数学中的一个概念，包括有理数和无理数。有理数可以表示为两个整数的比值，而无理数是无限不循环小数，如根号2和π。实数在数轴上都有对应的点。"
        }
    ]
    print(f"  准备了 {len(sft_data)} 条SFT数据")
    
    # 2. 检查是否有预训练模型
    print("\n🔍 查找预训练模型...")
    model_dirs = [d for d in os.listdir(".") if d.startswith("output_") and os.path.isdir(d)]
    
    if not model_dirs:
        print("⚠️  找不到预训练模型，请先运行 python train.py 进行预训练!")
        print("   建议先用 Demo 模式预训练")
        return
    
    print(f"  找到 {len(model_dirs)} 个模型目录:")
    for i, d in enumerate(model_dirs, 1):
        print(f"    {i}. {d}")
    
    choice = input("\n选择一个模型来做SFT (1-{}): ".format(len(model_dirs))).strip()
    idx = int(choice) - 1 if choice.isdigit() and 1 &lt;= int(choice) &lt;= len(model_dirs) else 0
    model_dir = model_dirs[idx]
    
    # 查找完整模型
    final_dir = os.path.join(model_dir, "minimind_final")
    if os.path.exists(final_dir):
        model_dir = final_dir
    
    print(f"\n✅ 使用: {model_dir}")
    
    # 3. 加载模型和Tokenizer
    print("\n📦 加载模型...")
    model = MiniMind.from_pretrained(model_dir)
    
    tokenizer_path = os.path.join(model_dir, "tokenizer.pt")
    if not os.path.exists(tokenizer_path):
        # 试着从上级目录找
        tokenizer_path = os.path.join(os.path.dirname(model_dir), "tokenizer.pt")
    
    tokenizer = Tokenizer.load(tokenizer_path)
    
    # 4. 创建SFT数据集
    print("\n🎯 创建SFT数据集...")
    config = get_config("demo")
    dataset = SFTDataset(sft_data, tokenizer, max_seq_len=config.model.max_seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 5. 展示一下SFT前的效果
    print("\n" + "="*60)
    print("📝 SFT前的效果:")
    print("="*60)
    
    model.eval()
    for item in sft_data[:3]:  # 展示前3条
        prompt = f"问题: {item['instruction']}\n回答:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        generated = model.generate(prompt_tokens, max_new_tokens=60, temperature=0.7)
        print(f"\nQ: {item['instruction']}")
        print(f"A: {tokenizer.decode(generated)}")
    
    # 6. 开始SFT训练
    print("\n" + "="*60)
    print("🚀 开始SFT训练...")
    print("="*60)
    
    confirm = input("\n开始SFT微调? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消了")
        return
    
    train_sft(model, dataloader, config, output_dir="./sft_result")
    
    # 7. 展示SFT后的效果
    print("\n" + "="*60)
    print("✨ SFT后的效果:")
    print("="*60)
    
    model.eval()
    for item in sft_data[:3]:  # 展示前3条
        prompt = f"问题: {item['instruction']}\n回答:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        generated = model.generate(prompt_tokens, max_new_tokens=60, temperature=0.7)
        print(f"\nQ: {item['instruction']}")
        print(f"A: {tokenizer.decode(generated)}")
    
    print("\n" + "="*60)
    print("✅ SFT完成！模型保存在 sft_result/")
    print("="*60)


if __name__ == "__main__":
    main()
