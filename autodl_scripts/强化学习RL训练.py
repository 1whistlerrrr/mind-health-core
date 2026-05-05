
# 🚀 强化学习大模型优化 - 基于 DPO/GRPO 思想
# 主题：脑科学、哲学、大脑健康相关知识

import torch
import os
import random
import time
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

print("="*70)
print("🧠 强化学习大模型优化 - 脑科学、哲学、大脑健康")
print("="*70)

# ============ 配置 ============
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./rl_brain_science_model"
MAX_SEQ_LEN = 512
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-6
MAX_STEPS = 300
USE_GPU = torch.cuda.is_available()

print(f"\n📦 设备信息: {'GPU' if USE_GPU else 'CPU'}")

# ============ 步骤1: 加载之前微调的模型 ============
print(f"\n{'='*70}")
print("📥 【第1步】加载模型和Tokenizer")
print(f"{'='*70}")

print(f"\n正在安装/检查依赖...")
try:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "modelscope"], check=True)
    from modelscope import AutoModelForCausalLM, AutoTokenizer

print(f"\n正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"\n正在加载模型 (用于生成)...")
model_original = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if USE_GPU else torch.float32,
    device_map="auto" if USE_GPU else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
print("✅ 模型加载完成！")

# ============ 步骤2: 定义生成函数 ============
def generate_response(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    
    if USE_GPU:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

# ============ 步骤3: 准备强化学习数据 - 脑科学、哲学、大脑健康 ============
print(f"\n{'='*70}")
print("📖 【第2步】准备强化学习偏好数据")
print(f"{'='*70}")

# 脑科学、哲学、大脑健康相关的提示和偏好数据
rl_preference_data = [
    {
        "prompt": "什么是意识？",
        "chosen": "意识是一个复杂且深刻的哲学和科学问题。从神经科学角度，意识可能与大脑中信息的整合和全局工作空间有关，涉及丘脑和皮层的广泛连接。从哲学角度，意识是主观体验的本质，包括感受、认知和自我觉知。目前仍没有一个完全统一的理论解释意识，但多个角度的研究正逐步深入。",
        "rejected": "意识就是大脑的产物，没什么复杂的。",
    },
    {
        "prompt": "如何保持大脑健康？",
        "chosen": "保持大脑健康需要综合多方面的因素：1) 定期运动，促进大脑血液循环和神经发生；2) 充足睡眠，帮助大脑代谢废物和巩固记忆；3) 饮食均衡，多摄入Omega-3脂肪酸和抗氧化物；4) 持续学习和挑战大脑，建立认知储备；5) 社交互动，维持大脑活跃；6) 控制压力，避免长期应激损伤大脑。",
        "rejected": "保持大脑健康很简单，多吃补脑品就行。",
    },
    {
        "prompt": "脑科学和哲学的关系是什么？",
        "chosen": "脑科学与哲学有着深刻的交叉。脑科学研究大脑的物理和生物机制，哲学探讨心灵、意识、自由意志等根本问题。神经哲学(Neurophilosophy)作为一个交叉领域，试图用神经科学的发现来回答哲学问题，比如意识的本质、自我同一性、道德判断的起源等。两者互相启发，共同探索人类认知的奥秘。",
        "rejected": "脑科学就是脑科学，哲学就是哲学，没什么关系。",
    },
    {
        "prompt": "记忆是如何形成的？",
        "chosen": "记忆的形成是一个复杂的神经过程，包括编码、存储和提取三个主要阶段。从神经机制上，记忆形成涉及神经元之间突触连接的增强，即突触可塑性。海马体在新记忆的编码中扮演关键角色，而长期记忆存储可能分布在大脑皮层的多个区域。同时，记忆巩固过程需要睡眠的参与。",
        "rejected": "记忆就是大脑里存的文件。",
    },
    {
        "prompt": "什么是思维？",
        "chosen": "思维是大脑对信息的加工过程，涉及推理、决策、问题解决等认知活动。神经科学研究发现，思维过程可能涉及大脑前额叶皮层的执行控制，以及多个脑网络的协作。从哲学角度，思维与语言、自我意识紧密相关，是人类认知能力的核心体现。",
        "rejected": "思维就是脑子想东西。",
    },
    {
        "prompt": "压力对大脑有什么影响？",
        "chosen": "压力对大脑有双重影响。短期适度的压力可以提高警觉性和认知表现，通过皮质醇适度调节神经活动。但长期或慢性压力可能导致：海马体萎缩，影响记忆和情绪管理；杏仁核过度激活，增加焦虑和负性情绪；前额叶皮层抑制，影响决策和控制力。通过运动、冥想、社交等方式可以有效缓解压力。",
        "rejected": "压力对大脑没什么影响，习惯就好。",
    },
]

print(f"\n✅ 准备了 {len(rl_preference_data)} 条偏好数据！")

# ============ 步骤4: 定义奖励函数 - 简单但可解释 ============
def compute_reward(model, prompt, chosen_response, rejected_response):
    """简单的奖励函数：给选中的回复更高奖励"""
    
    # 这里我们用偏好比较思想：Chosen的奖励应该比Rejected高
    # 实际的DPO会有更复杂的计算，但这个简单版可以演示思路
    
    # Tokenize 两种回答
    def tokenize_response(r):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": r}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    
    # 给 Chosen 较高的人工奖励，Rejected 较低
    # 这是一个简化版本，模拟人类偏好
    return {
        "chosen_reward": 2.0,
        "rejected_reward": 0.0,
    }

print(f"\n🎁 奖励函数设置完成")

# ============ 步骤5: 构建简单的强化学习训练数据 ============
print(f"\n{'='*70}")
print("⚙️ 【第3步】准备强化学习训练数据")
print(f"{'='*70}")

# 为了简化演示，我们用类似偏好微调的方式 (DPO 思想)
# 构建数据集：包含 prompt, chosen, rejected

def format_rl_data(prompt, chosen, rejected):
    # 构建 DPO 格式的数据（简化版）
    def tokenize_pair(prompt, response):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    
    return tokenize_pair(prompt, chosen)  # 用 Chosen 作为目标

# 准备数据集
from datasets import Dataset
rl_dataset_list = []
for item in rl_preference_data:
    # 每条数据用多次，增加训练量
    for i in range(5):
        rl_dataset_list.append({"instruction": item["prompt"], "output": item["chosen"]})
rl_dataset = Dataset.from_list(rl_dataset_list)

# 格式化数据集
def format_example_rl(example):
    prompt = example["instruction"]
    output = example["output"]
    text = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    
    tokenized = tokenizer(
        text, truncation=True, max_length=MAX_SEQ_LEN,
        padding="max_length", return_tensors="pt"
    )
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": tokenized["input_ids"].squeeze(),
        "attention_mask": tokenized["attention_mask"].squeeze(),
        "labels": labels.squeeze(),
    }

rl_dataset = rl_dataset.map(format_example_rl, remove_columns=rl_dataset.column_names)
print(f"✅ 强化学习数据集准备好了，共 {len(rl_dataset)} 条！")

# ============ 步骤6: 设置模型训练 ============
print(f"\n{'='*70}")
print("🚀 【第4步】开始强化学习阶段训练")
print(f"{'='*70}")

# 加载模型用于训练
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_for_rl_train = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto" if USE_GPU else None,
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
model_for_rl_train = get_peft_model(model_for_rl_train, lora_config)
model_for_rl_train.print_trainable_parameters()

# ============ 训练参数 ============
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
    save_total_limit=1,
    remove_unused_columns=False,
    ddp_timeout=900,
)

trainer = Trainer(
    model=model_for_rl_train,
    args=training_args,
    train_dataset=rl_dataset,
)

# ============ 开始训练 ============
start_time = time.time()
print("\n🚀 开始强化学习训练...")
trainer.train()
print(f"\n✅ 训练完成！耗时 {time.time() - start_time:.1f} 秒")

# ============ 保存模型 ============
final_output_path = os.path.join(OUTPUT_DIR, "final_rl_model")
trainer.model.save_pretrained(final_output_path)
tokenizer.save_pretrained(final_output_path)
print(f"\n💾 强化学习后的模型保存到: {final_output_path}")

# ============ 步骤7: 对比训练前后的效果 ============
print(f"\n{'='*70}")
print("🧪 【第5步】对比强化学习前后的效果")
print(f"{'='*70}")

from peft import PeftModel
model_rl_updated = PeftModel.from_pretrained(model_original, final_output_path)
model_rl_updated = model_rl_updated.merge_and_unload()

test_prompts = [
    "什么是意识？",
    "如何保持大脑健康？",
]

print("\n📊 开始对比:")
for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"🎯 Prompt: {prompt}")
    print(f"\n[原模型]:")
    print(generate_response(model_original, prompt)[:180])
    print(f"\n[强化学习后]:")
    print(generate_response(model_rl_updated, prompt)[:180])
    print(f"{'='*60}")

# ============ 完成！ ============
print(f"\n\n{'='*70}")
print("🎉 强化学习演示完整完成！")
print(f"\n📖 说明：")
print("  - 这里我们演示的是一个简化的强化学习流程")
print("  - 基于偏好数据，模拟 DPO 思路来让模型输出更符合人类偏好")
print("  - 主题聚焦在：脑科学、哲学、大脑健康")
print(f"\n💡 完整的 RLHF 需要额外训练奖励模型，这里我们简化演示了核心思想")
print(f"{'='*70}")
