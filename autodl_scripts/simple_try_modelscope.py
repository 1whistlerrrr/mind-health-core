
# 🧪 用魔搭社区（国内更快！）加载 Qwen2.5-0.5B-Instruct！
import torch
import os

print("="*60)
print("🐋 用魔搭社区试试 Qwen2.5-0.5B-Instruct！")
print("="*60)

# 第 1 步：先安装 modelscope
print("\n📦 检查并安装 modelscope...")
try:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("正在安装 modelscope...")
    import subprocess
    subprocess.run(["pip", "install", "modelscope"], check=True)
    from modelscope import AutoModelForCausalLM, AutoTokenizer

# 第 2 步：加载模型
MODEL_NAME = "qwen/Qwen2.5-0.5B-Instruct"
print(f"\n📥 正在下载并加载 {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("✅ 模型加载完成！")

# 第 3 步：定义推理函数
def chat(prompt):
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
    
    response = tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response

# 第 4 步：试几个例子！
print("\n🤖 开始测试！")

test_prompts = [
    "你好！简单介绍一下你自己",
    "什么是机器学习？",
    "Python 和 C++ 有什么区别？",
    "给我讲一个关于程序员的笑话",
]

for i, prompt in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"🎯 问题 {i+1}: {prompt}")
    print(f"{'='*60}")
    try:
        answer = chat(prompt)
        print(f"\n{answer}")
    except Exception as e:
        print(f"❌ 报错: {e}")

print("\n" + "="*60)
print("🎉 测试完成！怎么样？")
print("="*60)
print("""
接下来你可以：
1. 如果想玩更多，直接修改 test_prompts 数组
2. 或者运行我们的微调脚本
""")
