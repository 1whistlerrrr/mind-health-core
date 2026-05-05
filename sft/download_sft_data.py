
"""
🚀 下载/准备开源 SFT 微调数据
（专门为小模型准备的小而精的数据集！）
"""
import json
import random
from typing import List, Dict
import os


def generate_dummy_sft_data(n_samples: int = 100) -&gt; List[Dict]:
    """生成一些高质量的中文 SFT 演示数据（因为现在无法联网下载）"""

    instructions = [
        "什么是机器学习？",
        "如何学习编程？",
        "给我讲个笑话",
        "帮我写个 Python 脚本",
        "什么是大语言模型？",
        "你好",
        "你能做什么？",
        "帮我翻译一下",
        "解释一下什么是深度学习",
        "给我推荐一本书",
        "写一首关于月亮的诗",
        "告诉我如何做番茄炒蛋",
        "解释一下什么是量子计算",
        "写一个简短的科幻故事",
        "告诉我如何保持健康",
        "解释一下什么是自然语言处理",
        "帮我起个英文名字",
        "推荐一部好看的电影",
        "告诉我如何学习数学",
        "解释一下什么是神经网络",
    ]

    outputs = [
        "机器学习是人工智能的一个分支，让计算机从数据中学习规律。",
        "学习编程建议从 Python 开始，多写代码，多做项目！",
        "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25！",
        "好的，我可以帮你写 Python 脚本，请告诉我具体需求。",
        "大语言模型是通过大量文本学习的 AI，可以预测下一个词。",
        "你好！有什么可以帮你的吗？",
        "我可以帮你解答问题、写代码、翻译、聊天等。",
        "请告诉我你想翻译什么内容？",
        "深度学习是机器学习的一个分支，用深层神经网络学习。",
        "我推荐《人类简史》，这是一本有趣的历史科普书。",
        "窗前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "番茄炒蛋的做法很简单：先炒鸡蛋，再炒番茄，最后一起炒。",
        "量子计算用量子力学原理进行计算，可能比普通计算机更快。",
        "在遥远的未来，一艘宇宙飞船正在探索银河系的边缘...",
        "保持健康需要规律作息、健康饮食和适量运动。",
        "自然语言处理是让计算机理解和生成人类语言的技术。",
        "你的英文名可以叫 Alex，这个名字友好又好记。",
        "我推荐《盗梦空间》，这是一部很有创意的科幻片。",
        "学习数学需要理解基础概念，多做练习题。",
        "神经网络是受大脑神经元启发的计算模型，用于深度学习。",
    ]

    data = []
    for i in range(n_samples):
        data.append({
            "instruction": instructions[i % len(instructions)],
            "input": "",
            "output": outputs[i % len(outputs)]
        })

    random.shuffle(data)
    return data


def save_data(data: List[Dict], filename: str):
    """保存数据到 JSON 文件"""
    save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 {len(data)} 条数据到: {filepath}")


def main():
    print("="*60)
    print("🚀 准备 SFT 微调数据")
    print("="*60)

    print("\n📋 可选方案：")
    print("1. 生成 100 条演示中文 SFT 数据")
    print("2. 生成 500 条演示中文 SFT 数据")
    print("3. 生成 1000 条演示中文 SFT 数据")

    choice = input("\n请选择 (1-3): ").strip()

    if choice == "1":
        data = generate_dummy_sft_data(100)
        save_data(data, "demo_sft_100.json")
    elif choice == "2":
        data = generate_dummy_sft_data(500)
        save_data(data, "demo_sft_500.json")
    elif choice == "3":
        data = generate_dummy_sft_data(1000)
        save_data(data, "demo_sft_1000.json")
    else:
        print("⚠️  无效输入，用 100 条吧")
        data = generate_dummy_sft_data(100)
        save_data(data, "demo_sft_100.json")

    print("\n💡 提示：")
    print("- 这只是演示数据，效果有限")
    print("- 真实使用时，建议去 HuggingFace 下载高质量数据")
    print("- 比如: alpaca-zh, firefly-zh, bell-zh 等")
    print("- 搜索关键词: huggingface sft chinese instruction tuning")


if __name__ == "__main__":
    main()

