
# 🐋 测试微调后的 LoRA 模型！
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

print("="*60)
print("🧪 开始测试 LoRA 微调后的模型！")
print("="*60)

# ============ 配置 ============
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL = "./qwen-0.5b-lora-output/final_checkpoint"

print(f"📦 加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

print(f"🔗 加载 LoRA 适配器...")
model = PeftModel.from_pretrained(model, LORA_MODEL)
model = model.merge_and_unload()
print("✅ 模型加载完成！")

# ============ 推理函数 ============
def generate_response(prompt, max_length=256, temperature=0.7, top_p=0.95):
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    response = tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response

# ============ 测试问题！ ============
test_prompts = [
    "你好，请简单介绍一下自己",
    "什么是机器学习？",
    "Python 和 C++ 有什么区别？",
    "给我讲一个关于程序员的笑话",
]

for idx, prompt in enumerate(test_prompts):
    print(f"\n--- 问题 {idx+1}: {prompt}")
    print("--- 回答:")
    try:
        result = generate_response(prompt)
        print(result)
    except Exception as e:
        print(f"错误: {e}")

print("\n" + "="*60)
print("🎉 测试完成！")
print("="*60)


def infer_lora_func():
    """让这个脚本可以被导入调用的版本"""
    print("这个函数会完整执行上面的推理测试！")
