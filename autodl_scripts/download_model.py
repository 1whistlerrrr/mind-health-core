
# 📥 提前下载模型到本地！避免加载慢！
import os
from huggingface_hub import snapshot_download

print("="*60)
print("📥 正在下载模型到本地...")
print("="*60)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"\n🎯 正在下载: {MODEL_NAME}")
model_path = snapshot_download(MODEL_NAME)
print(f"\n✅ 模型已下载到: {model_path}")

print("="*60)
print("提示：现在运行其他脚本时会快很多，因为模型已经在本地了！")
print("="*60)
