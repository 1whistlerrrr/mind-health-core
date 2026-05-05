
"""
🎯 使用训练好的 MiniMind 模型
这个脚本演示如何加载和使用已保存的模型
"""
import torch
import sys
sys.path.insert(0, '/Users/liuchuyao/Documents/Code/python-leo')
from model import MiniMind
from config import Config

def load_model(checkpoint_path="minimind_model.pth"):
    """加载训练好的模型"""
    print("📦 正在加载模型...")

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 从 checkpoint 中获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 如果没保存 config，使用默认的
        config = Config()

    print(f"📋 模型配置: vocab={config.vocab_size}, d_model={config.d_model}")

    # 创建模型
    model = MiniMind(config)

    # 加载参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 切换到评估模式

    print("✅ 模型加载成功！")
    return model, config

def generate_text(model, config, start_text="今天天气", max_length=100, temperature=0.7):
    """使用模型生成文本"""
    model.eval()

    # 注意：这里你需要实现你的 tokenizer 来编码 start_text
    # 由于我们没有保存 tokenizer，这个部分需要根据你的情况补充

    print("\n⚠️  注意：")
    print("  要完整使用模型，你还需要：")
    print("  1. 保存和加载 tokenizer")
    print("  2. 实现从文本到 token 的编码")
    print("  3. 实现从 token 到文本的解码\n")

    print(f"📝 尝试以 '{start_text}' 开头生成 {max_length} 个 token...")

    # 这里只是一个框架，实际代码需要根据你的 tokenizer 调整
    print("(这是一个演示，需要和你的 tokenizer 配合使用)")

if __name__ == "__main__":
    print("="*60)
    print("🎯 MiniMind 模型加载演示")
    print("="*60)

    # 1. 加载模型
    model, config = load_model()

    # 2. 查看模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数量: {total_params:,} (全部)")
    print(f"📊 可训练参数量: {trainable_params:,}\n")

    # 3. 演示如何做推理（需要 tokenizer 配合）
    generate_text(model, config)

    print("\n" + "="*60)
    print("✅ 模型已准备好！")
    print("="*60)

