
# 🚀 优化版！模型加载更快！
import torch
import os
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import time

print("="*60)
print("🚀 快速完整微调对比脚本！")
print("="*60)

# ============ 配置 ============
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen-0.5b-lora-output"
MAX_SEQ_LEN = 512
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_STEPS = 200  # 快速看效果

# ============ 第 1 步：快速加载模型！ ============
print(f"\n{'='*60}")
print("📥 【第 1 阶段】快速加载模型...")
print(f"{'='*60}")

print(f"\n正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"正在加载模型 (用更快的参数)...")
model_original = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,  # 低内存模式！
    use_safetensors=True,     # 用安全张量格式加载！
)
print("✅ 模型快速加载完成！")

# ============ 测试生成函数 ============
def generate_with_model(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # 缩短一点，更快
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
    return tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

test_prompts = [
    "你好！简单介绍自己",
    "什么是机器学习？",
]

original_results = {}
print(f"\n🤖 快速测试原模型...")
for prompt in test_prompts:
    print(f"\n  🎯 Q: {prompt}")
    ans = generate_with_model(model_original, prompt)
    original_results[prompt] = ans
    print(f"  A: {ans[:80]}...")

# ============ 第 2 步：快速微调！ ============
print(f"\n{'='*60}")
print("🚀 【第 2 阶段】快速 LoRA 微调！")
print(f"{'='*60}")

time.sleep(1)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("\n重新加载模型用于训练...")
model_for_train = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

lora_config = LoraConfig(
    r=8,  # 更小的秩，更快！
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_for_train = get_peft_model(model_for_train, lora_config)
model_for_train.print_trainable_parameters()

# 准备数据
print("\n📖 准备快速小数据集...")
dataset_name = "tatsu-lab/alpaca"
dataset = load_dataset(dataset_name)

def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"] if example["input"] is not None else ""
    output = example["output"]
    
    if input_text:
        prompt = f"""&lt;|im_start|&gt;user
{instruction}
{input_text}&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
{output}&lt;|im_end|&gt;"""
    else:
        prompt = f"""&lt;|im_start|&gt;user
{instruction}&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
{output}&lt;|im_end|&gt;"""
    
    tokenized = tokenizer(
        prompt, truncation=True, max_length=MAX_SEQ_LEN,
        padding="max_length", return_tensors="pt"
    )
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": tokenized["input_ids"].squeeze(),
        "attention_mask": tokenized["attention_mask"].squeeze(),
        "labels": labels.squeeze(),
    }

# 只用小部分数据快速训练！
dataset = dataset["train"].select(range(200))  # 只取前 200 条！
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# 快速训练参数！
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    logging_steps=20,
    save_steps=100,
    warmup_steps=10,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to="none",
    bf16=torch.cuda.is_bf16_supported(),
    save_total_limit=2,
    remove_unused_columns=False,
    group_by_length=True,
    ddp_timeout=900,
)

trainer = Trainer(model=model_for_train, args=training_args, train_dataset=dataset)

print("\n🚀 开始快速训练！")
start_time = time.time()
trainer.train()
print(f"✅ 训练完成！耗时 {time.time() - start_time:.1f} 秒")

final_output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
trainer.model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"💾 模型已保存到: {final_output_dir}")

# ============ 第 3 步：测试对比！ ============
print(f"\n{'='*60}")
print("🧪 【第 3 阶段】测试微调后的模型！")
print(f"{'='*60}")

print("\n📥 加载微调后的模型...")
from peft import PeftModel
model_tuned = PeftModel.from_pretrained(model_original, final_output_dir)
model_tuned = model_tuned.merge_and_unload()

tuned_results = {}
for prompt in test_prompts:
    ans = generate_with_model(model_tuned, prompt)
    tuned_results[prompt] = ans

print(f"\n\n{'='*60}")
print("📊 【最终对比】")
print(f"{'='*60}")

for prompt in test_prompts:
    print(f"\n🎯 Q: {prompt}")
    print(f"[原模型] {original_results[prompt]}")
    print(f"[微调后] {tuned_results[prompt]}")

print(f"\n\n{'='*60}")
print("🎉 全部完成！")
print(f"{'='*60}")
