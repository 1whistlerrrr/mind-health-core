
# 📊 评价模型是否训练到位！
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

print("="*60)
print("📊 模型评价工具！")
print("="*60)

# ============ 配置 ============
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "./qwen-0.5b-lora-output/final_checkpoint"

# ============ 评价问题集（可以自己加！） ============
EVAL_QUESTIONS = {
    "基础问答": [
        "什么是机器学习？",
        "你好，请简单介绍自己",
    ],
    "知识问答": [
        "中国的首都是哪里？",
        "Python 和 C++ 有什么区别？",
    ],
    "创作能力": [
        "给我讲一个关于程序员的笑话",
        "写一首关于月亮的短诗",
    ],
    "逻辑推理": [
        "小明比小红高，小红比小华高，那么谁最高？",
        "3 + 4 * 2 = ?",
    ],
}

print(f"\n📥 正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载原模型
model_original = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载微调后的模型（如果存在）
model_tuned = None
if os.path.exists(LORA_PATH):
    print(f"\n✅ 发现微调后的模型！正在加载...")
    from peft import PeftModel
    model_tuned = PeftModel.from_pretrained(model_original, LORA_PATH)
    model_tuned = model_tuned.merge_and_unload()

# ============ 生成函数 ============
def generate_with_model(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        start_time = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
        gen_time = time.time() - start_time
    return tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True), gen_time

# ============ 开始评价！ ============
print(f"\n{'='*60}")
print("🧪 开始评价...")
print(f"{'='*60}")

results_original = {}
results_tuned = {}

for category, questions in EVAL_QUESTIONS.items():
    print(f"\n\n📁 【{category}】")
    results_original[category] = []
    if model_tuned:
        results_tuned[category] = []
    
    for question in questions:
        print(f"\n  🎯 Q: {question}")
        
        # 原模型回答
        ans_original, t1 = generate_with_model(model_original, question)
        print(f"  [原模型] A: {ans_original[:100]}... (耗时: {t1:.2f}秒)")
        results_original[category].append({"question": question, "answer": ans_original, "time": t1})
        
        # 微调后回答
        if model_tuned:
            ans_tuned, t2 = generate_with_model(model_tuned, question)
            print(f"  [微调后] A: {ans_tuned[:100]}... (耗时: {t2:.2f}秒)")
            results_tuned[category].append({"question": question, "answer": ans_tuned, "time": t2})

# ============ 保存结果 ============
print(f"\n\n{'='*60}")
print("📝 正在保存详细结果...")
print(f"{'='*60}")

with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("📊 模型评价结果\n")
    f.write("="*60 + "\n")
    f.write(f"模型: {MODEL_NAME}\n")
    f.write(f"评价时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for category in EVAL_QUESTIONS.keys():
        f.write(f"\n\n📁 【{category}】\n")
        
        for i, question in enumerate(EVAL_QUESTIONS[category]):
            f.write(f"\n  🎯 Q: {question}\n")
            
            if category in results_original:
                ans_o = results_original[category][i]
                f.write(f"  [原模型] ({ans_o['time']:.2f}秒):\n{ans_o['answer']}\n")
            
            if model_tuned and category in results_tuned:
                ans_t = results_tuned[category][i]
                f.write(f"  [微调后] ({ans_t['time']:.2f}秒):\n{ans_t['answer']}\n")

print(f"✅ 详细结果已保存到 evaluation_results.txt！")

# ============ 简单人工评测引导 ============
print(f"\n\n{'='*60}")
print("🤔 现在你可以检查回答，简单评分！")
print(f"{'='*60}")

print("""
评价标准：
1. 回答是否准确？
2. 回答是否完整？
3. 回答是否符合人类偏好？
""")

if model_tuned:
    print("""
你觉得微调后的效果怎么样？
    """)
