
"""
🎯 完整加载和推理示例 - 使用 minimind_model.pth
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

def load_complete_model(checkpoint_path=None):
    """完整加载模型和tokenizer"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'checkpoints', 
            'minimind_model.pth'
        )
    
    print("="*60)
    print("📦 正在加载...")
    print("="*60)
    
    # 1. 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 2. 创建模型
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
    
    # 3. 创建 tokenizer
    tokenizer = SimpleTokenizer(
        vocab=checkpoint['tokenizer_vocab'],
        inv_vocab=checkpoint['tokenizer_inv_vocab']
    )
    
    print(f"✅ 加载成功！")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  词表大小: {len(tokenizer.vocab)}")
    print(f"  上下文长度: {config['max_seq_len']}\n")
    
    return model, tokenizer

def generate_example(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """生成文本"""
    model.eval()
    device = next(model.parameters()).device
    
    prompt_tokens = tokenizer.encode(prompt)
    print(f"📝 输入: {prompt}")
    print(f"🔢 Tokenized: {prompt_tokens}")
    
    with torch.no_grad():
        generated = model.generate(prompt_tokens, max_length, temperature)
    
    output = tokenizer.decode(generated)
    return output

if __name__ == "__main__":
    print("\n" + "="*60)
    print("✨ MiniMind 完整推理示例")
    print("="*60)
    
    # 1. 加载模型
    model, tokenizer = load_complete_model()
    
    # 2. 交互式生成示例
    prompts = [
        "今天",
        "人工智能",
        "数学"
    ]
    
    print("\n" + "="*60)
    print("🎨 生成示例")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        try:
            print(f"\n--- 示例 {i} ---")
            output = generate_example(model, tokenizer, prompt, max_length=50)
            print(f"✨ 生成: {output}")
        except Exception as e:
            print(f"⚠️  错误: {e}")
    
    print("\n" + "="*60)
    print("🎮 交互式生成")
    print("="*60)
    print("输入 'quit' 退出\n")
    
    while True:
        user_input = input("输入提示: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            output = generate_example(model, tokenizer, user_input, max_length=100)
            print(f"输出: {output}\n")

