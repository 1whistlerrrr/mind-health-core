
# 🎯 完整流程！先测原模型，再微调，再测微调后的效果对比！
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
print("🐋 欢迎使用完整微调对比脚本！")
print("="*60)

# ============ 配置 ============
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen-0.5b-lora-output"
MAX_SEQ_LEN = 512
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_STEPS = 200  # 小量训练，快速看效果

# ============ 第 1 步：设置 HF 镜像站（国内更快！） ============
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============ 第 2 步：测试原模型效果 ============
print(f"\n{'='*60}")
print("📝 【第 1 阶段】先测试微调前的模型效果！")
print(f"{'='*60}")

print(f"\n📥 正在加载原模型 {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model_original = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("✅ 原模型加载完成！")

def generate_with_model(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
    return tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

test_prompts = [
    "你好！简单介绍一下你自己",
    "什么是机器学习？",
    "Python 和 C++ 有什么区别？",
    "给我讲一个关于程序员的笑话",
]

original_results = {}
print(f"\n🤖 开始测试原模型...")
for prompt in test_prompts:
    print(f"\n🎯 问题: {prompt}")
    try:
        answer = generate_with_model(model_original, prompt)
        original_results[prompt] = answer
        print(f"{answer}")
    except Exception as e:
        print(f"❌ 报错: {e}")
        original_results[prompt] = f"ERROR: {e}"

# ============ 第 3 步：开始 LoRA 微调 ============
print(f"\n{'='*60}")
print("🎯 【第 2 阶段】开始 LoRA 微调！")
print(f"{'='*60}")

time.sleep(2)
print("\n准备 4bit 量化...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("重新加载模型用于训练...")
model_for_train = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 设置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_for_train = get_peft_model(model_for_train, lora_config)
model_for_train.print_trainable_parameters()

# 准备数据
print("\n📖 准备微调数据...")
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

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# 设置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=50,
    warmup_steps=10,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to="none",
    bf16=torch.cuda.is_bf16_supported(),
    save_total_limit=3,
    remove_unused_columns=False,
    group_by_length=True,
)

trainer = Trainer(model=model_for_train, args=training_args, train_dataset=dataset["train"])

print("\n🚀 开始训练！")
start_time = time.time()
trainer.train()
print(f"✅ 训练完成！耗时 {time.time() - start_time:.1f} 秒")

# 保存模型
final_output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
trainer.model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"💾 模型已保存到: {final_output_dir}")

# ============ 第 4 步：加载微调后的模型，测试效果 ============
print(f"\n{'='*60}")
print("🧪 【第 3 阶段】测试微调后的模型效果！")
print(f"{'='*60}")

print("\n📥 加载微调后的模型...")
from peft import PeftModel
model_tuned = PeftModel.from_pretrained(model_original, final_output_dir)
model_tuned = model_tuned.merge_and_unload()
print("✅ 微调后模型加载完成！")

tuned_results = {}
print(f"\n🤖 开始测试微调后的模型...")
for prompt in test_prompts:
    print(f"\n🎯 问题: {prompt}")
    try:
        answer = generate_with_model(model_tuned, prompt)
        tuned_results[prompt] = answer
        print(f"{answer}")
    except Exception as e:
        print(f"❌ 报错: {e}")
        tuned_results[prompt] = f"ERROR: {e}"

# ============ 第 5 步：对比展示！ ============
print(f"\n\n{'='*60}")
print("📊 【最终对比】微调前 vs 微调后！")
print(f"{'='*60}")

for prompt in test_prompts:
    print(f"\n\n🎯 问题: {prompt}")
    print(f"--- 原模型回答 ---")
    print(original_results[prompt])
    print(f"\n--- 微调后回答 ---")
    print(tuned_results[prompt])

print(f"\n\n{'='*60}")
print("🎉 全部完成！")
print(f"{'='*60}")
print("你觉得微调后的效果怎么样？")

if not os.path.exists("comparison_results.txt"):
    with open("comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("微调前 vs 微调后效果对比\n")
        for prompt in test_prompts:
            f.write(f"\n\n问题: {prompt}\n原模型回答:\n{original_results[prompt]}\n\n微调后回答:\n{tuned_results[prompt]}\n")
    print(f"\n💾 对比结果已保存到 comparison_results.txt")
