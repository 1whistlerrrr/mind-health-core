
# 🐋 单文件一键开始！从安装到微调一条龙！
# 这个脚本会帮你安装依赖、微调、测试！
import os
import sys
import subprocess

def run_command(cmd):
    print(f"\n🚀 运行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("警告/错误:", result.stderr)
    return result.returncode

def check_dependencies():
    print("="*60)
    print("📦 检查并安装依赖...")
    print("="*60)
    
    requirements = [
        "transformers", "datasets", "peft", "accelerate",
        "torch", "sentencepiece", "protobuf", "tqdm", "bitsandbytes"
    ]
    
    missing = []
    for req in requirements:
        try:
            __import__(req.replace("-", "_"))
        except ImportError:
            missing.append(req)
    
    if missing:
        print(f"❌ 缺少: {missing}")
        print("正在安装...")
        run_command(f"pip install --upgrade pip")
        for req in missing:
            run_command(f"pip install {req}")
    else:
        print("✅ 所有依赖都已安装！")

def main():
    print("\n" + "="*60)
    print("🐋 欢迎使用 Qwen 一键微调脚本！")
    print("="*60)
    
    # 第 1 步：检查依赖
    check_dependencies()
    
    # 第 2 步：问用户要做什么
    print("\n📋 请选择：")
    print("1. 开始 LoRA 微调 (约 10-20 分钟)")
    print("2. 只测试推理 (如果你已经微调过了)")
    print("3. 退出")
    
    choice = input("\n请输入 (1-3): ").strip()
    
    if choice == "1":
        print("\n🎯 好的，开始微调！")
        from train_lora import train_lora_func
    elif choice == "2":
        print("\n🧪 好的，开始推理测试！")
        from infer_lora import infer_lora_func
    elif choice == "3":
        print("👋 再见！")
        return
    else:
        print("❌ 无效输入！")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 再见！")
        sys.exit(0)

print("\n\n" + "="*60)
print("🎉 脚本执行完毕！")
print("="*60)
