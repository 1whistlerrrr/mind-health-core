
# 🚀 最终版本！经过全面检查！
import torch
import os
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import time

print("="*60)
print("🚀 最终版本！经过全面检查！")
print("="*60)

# ============ 配置 ============
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen-0.5b-lora-output"
MAX_SEQ_LEN = 512
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_STEPS = 200

# ============ 第 1 步：用 ModelScope 加载！ ============
print(f"\n{'='*60}")
print("📥 【第 1 阶段】用 ModelScope 加载模型...")
print(f"{'='*60}")

print(f"\n正在安装/检查 modelscope...")
try:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "modelscope"], check=True)
    from modelscope import AutoModelForCausalLM, AutoTokenizer

print(f"\n正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"正在加载模型...")
model_original = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
print("✅ 模型加载完成！")

# ============ 生成函数 ============
def generate_with_model(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
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
    ans = generate_with_model(model_original, prompt)
    original_results[prompt] = ans
    print(f"\n  🎯 Q: {prompt}")
    print(f"  A: {ans[:80]}...")

# ============ 第 2 步：微调！ ============
print(f"\n{'='*60}")
print("🚀 【第 2 阶段】开始微调...")
print(f"{'='*60}")

time.sleep(1)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_for_train = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_for_train = get_peft_model(model_for_train, lora_config)
model_for_train.print_trainable_parameters()

# ============ 数据！内置演示数据！ ============
print("\n📖 准备内置演示数据...")
demo_data = [
    {"instruction": "什么是机器学习？", "input": "", "output": "机器学习是人工智能的一个分支，让计算机从数据中学习。"},
    {"instruction": "什么是Python？", "input": "", "output": "Python是一种简单又强大的编程语言。"},
    {"instruction": "你好！", "input": "", "output": "你好！我是AI助手。"},
    {"instruction": "什么是AI？", "input": "", "output": "AI是人工智能的缩写，也就是人工智能。"},
    {"instruction": "什么是深度学习？", "input": "", "output": "深度学习是机器学习的一部分，用深层神经网络。"},
    {"instruction": "什么是GPU？", "input": "", "output": "GPU是图形处理器，用来加速AI运算。"},
    {"instruction": "什么是LoRA？", "input": "", "output": "LoRA是低秩适应，一种参数高效微调方法。"},
    {"instruction": "什么是Transformer？", "input": "", "output": "Transformer是一种深度学习架构。"},
    {"instruction": "什么是大模型？", "input": "", "output": "大模型就是大型语言模型，比如GPT。"},
    {"instruction": "什么是微调？", "input": "", "output": "微调就是让预训练模型适应特定任务。"},
]

# ============ 数据格式化 ============
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

from datasets import Dataset as HFDataset
dataset = HFDataset.from_list(demo_data)
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

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
    ddp_timeout=900,
)

trainer = Trainer(model=model_for_train, args=training_args, train_dataset=dataset)

print("\n🚀 开始训练！")
start_time = time.time()
trainer.train()
print(f"✅ 训练完成！耗时 {time.time() - start_time:.1f} 秒")

final_output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
trainer.model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"💾 模型已保存到: {final_output_dir}")

# ============ 第 3 步：测试对比！ ============
print(f"\n{'='*60}")
print("🧪 【第 3 阶段】测试微调后的模型...")
print(f"{'='*60}")

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
