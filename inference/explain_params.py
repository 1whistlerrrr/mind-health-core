
"""
🧮 解释你的模型为什么是 6.2M 参数
"""
import torch
import os
import sys

# 添加 src 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_and_explain():
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'minimind_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("="*60)
    print("🧮 你的模型为什么是 6.2M 参数？")
    print("="*60)

    total_params = 0

    for name, param in checkpoint['model_state_dict'].items():
        params_count = param.numel()
        total_params += params_count

        # 简单的分类打印
        if 'embedding' in name or 'tok_embeddings' in name:
            print(f"📖 {name:30} : {param.shape} = {params_count:10,} (词嵌入)")
        elif 'attention' in name:
            print(f"🧠 {name:30} : {param.shape} = {params_count:10,} (注意力)")
        elif 'feed_forward' in name or 'ffn' in name:
            print(f"⚙️ {name:30} : {param.shape} = {params_count:10,} (前馈)")
        elif 'norm' in name:
            print(f"📏 {name:30} : {param.shape} = {params_count:10,} (归一化)")
        elif 'output' in name:
            print(f"🎯 {name:30} : {param.shape} = {params_count:10,} (输出层)")
        else:
            print(f"🤖 {name:30} : {param.shape} = {params_count:10,}")

    print("="*60)
    print(f"✨ 总共: {total_params:,} 个参数 = {total_params / 1_000_000:.1f}M")
    print("="*60)

    print("""
📝 这是怎么算的？
-------------------
你模型的配置:
- vocab_size = 4000 (词表大小)
- dim = 256      (隐藏层大小)
- n_layers = 4    (层数)
- n_heads = 8     (注意力头数)

主要大头:
1. tok_embeddings.weight: 4000*256 = 1,024,000 (~1M)
2. output.weight: 4000*256 = 1,024,000 (~1M)
3. 每层的 attention + feed_forward: 约 1M/层
4. 4 层就是约 4M

加起来一共就是 ~6.2M！

""")


if __name__ == "__main__":
    load_and_explain()

