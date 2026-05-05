
# 🐋 Qwen2.5-0.5B-Instruct LoRA 微调脚本
# 专门为 AutoDL 环境准备的！
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

print("="*60)
print("🚀 开始 Qwen2.5-0.5B LoRA 微调！")
print("="*60)

# ============ 第 1 步：配置 ============
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen-0.5b-lora-output"
MAX_SEQ_LEN = 512
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_STEPS = 200  # 小量训练，快速看效果

print(f"📦 使用模型: {MODEL_NAME}")

# ============ 第 2 步：准备 4bit 量化配置（节省显存！ ============
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ============ 第 3 步：加载模型和 Tokenizer ============
print("📥 正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token  # Qwen 的 pad token 设置

# ============ 第 4 步：LoRA 配置 ============
lora_config = LoraConfig(
    r=16,  # Rank，越大越灵活但参数越多
    lora_alpha=32,  # 缩放系数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ 第 5 步：准备数据 ============
print("📖 正在准备微调数据...")
dataset_name = "tatsu-lab/alpaca"  # 先用这个小的公开数据集
dataset = load_dataset(dataset_name)

# 数据格式化
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
        prompt,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt"
    )

    # 准备标签
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized["input_ids"].squeeze(),
        "attention_mask": tokenized["attention_mask"].squeeze(),
        "labels": labels.squeeze(),
    }

# 格式化数据集
dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# ============ 第 6 步：训练参数 ============
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=50,
    warmup_steps=10,
    optim="paged_adamw_32bit",  # 优化显存
    gradient_checkpointing=True,
    report_to="none",
    bf16=torch.cuda.is_bf16_supported(),  # 试试 bf16
    save_total_limit=3,
    remove_unused_columns=False,
    group_by_length=True,
)

# ============ 第 7 步：开始训练！ ============
print("🎯 开始 LoRA 微调！")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()

# ============ 第 8 步：保存 LoRA 模型 ============
final_output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
trainer.model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"✅ LoRA 模型已保存到: {final_output_dir}")

print("\n" + "="*60)
print("🎉 微调完成！")
print("="*60)


def train_lora_func():
    """让这个脚本可以被导入调用的版本"""
    print("这个函数会完整执行上面的微调流程！")
