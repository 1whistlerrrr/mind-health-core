
"""
MiniMind 训练主脚本
包含完整的功能：预训练、SFT微调、推理
"""
import sys
import os

# 确保能找到当前目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, print_config
from model import Tokenizer, MiniMind, TextDataset, train
import torch


def main():
    # 1. 选择模式
    print("\n" + "="*60)
    print("🤖 MiniMind - 你的第一个大模型")
    print("="*60)
    print("\n请选择运行模式:")
    print("  1. Tiny模式   - 超小模型，几秒跑完 (适合测试)")
    print("  2. Demo模式   - 小模型，几分钟 (推荐首次使用)")
    print("  3. Local模式  - 中等模型 (Mac本地)")
    print("  4. Cloud模式  - 大模型 (需要GPU)")
    print("  5. 仅测试推理 (从output目录加载已有模型)")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    mode_map = {
        "1": "tiny",
        "2": "demo", 
        "3": "local",
        "4": "cloud"
    }
    
    if choice in mode_map:
        # 训练模式
        mode = mode_map[choice]
        config = get_config(mode)
        print_config(config)
        
        # 2. 加载数据
        print("\n📖 加载数据...")
        with open("chinese.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        # 3. 训练Tokenizer
        print("🎯 训练Tokenizer...")
        tokenizer = Tokenizer(config.model.vocab_size)
        tokenizer.train(text)
        
        # 4. 创建数据集
        dataset = TextDataset(text, tokenizer, seq_len=config.model.max_seq_len)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.train.batch_size, 
            shuffle=True
        )
        
        # 5. 创建模型
        print("\n🏗️ 创建模型...")
        model = MiniMind(
            vocab_size=len(tokenizer.vocab),
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            max_seq_len=config.model.max_seq_len
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {num_params:,}")
        
        # 6. 开始训练
        confirm = input("\n✅ 准备完毕，开始训练? (y/n): ").strip().lower()
        if confirm == 'y':
            train(model, dataloader, config, tokenizer, output_dir=f"./output_{mode}")
    
    elif choice == "5":
        # 推理模式
        print("\n🔍 推理模式...")
        
        from model import InferencePipeline
        
        # 查找模型目录
        model_dirs = [d for d in os.listdir(".") if d.startswith("output_") and os.path.isdir(d)]
        if not model_dirs:
            print("❌ 找不到模型目录，请先训练!")
            return
        
        print("\n找到的模型目录:")
        for i, d in enumerate(model_dirs, 1):
            print(f"  {i}. {d}")
        
        choice = input("\n选择模型 (1-{}): ".format(len(model_dirs))).strip()
        idx = int(choice) - 1 if choice.isdigit() else 0
        model_dir = model_dirs[idx]
        
        # 查找里面的minimind_final
        final_dir = os.path.join(model_dir, "minimind_final")
        if os.path.exists(final_dir):
            model_dir = final_dir
        
        print(f"\n正在加载: {model_dir}")
        pipe = InferencePipeline(model_dir)
        
        # 交互式推理
        print("\n" + "="*60)
        print("✨ 推理模式 - 输入你的提示")
        print("="*60)
        print("输入 'quit' 退出\n")
        
        while True:
            prompt = input("提示: ").strip()
            if prompt.lower() == 'quit':
                break
            
            if not prompt:
                continue
            
            output = pipe.generate(prompt, max_new_tokens=100, temperature=0.7)
            print(f"\n输出: {output}\n")
            print("-"*60)
    
    else:
        print("❌ 无效选项")


if __name__ == "__main__":
    # 安装依赖
    try:
        import tqdm
    except ImportError:
        print("📦 安装依赖 tqdm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    
    main()
