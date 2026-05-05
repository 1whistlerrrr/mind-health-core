
import torch
import os

# 默认路径
checkpoint_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    'checkpoints',
    'minimind_model.pth'
)

print("📦 正在加载模型文件...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*50)
print("📋 模型文件包含内容:")
print("="*50)
for key in checkpoint.keys():
    print(f"  • {key}")

print("\n" + "="*50)
if 'model_state_dict' in checkpoint:
    print("🧠 模型参数字典信息:")
    print("="*50)
    total_params = 0
    for name, param in checkpoint['model_state_dict'].items():
        param_size = param.numel()
        total_params += param_size
        print(f"  • {name}: {param.shape} ({param_size:,} 参数")
    print(f"\n📊 总计: {total_params:,} 参数")

if 'epoch' in checkpoint:
    print(f"\n📅 训练到第 {checkpoint['epoch']} epoch")
if 'loss' in checkpoint:
    print(f"📉 最后训练 loss: {checkpoint['loss']:.4f}")
if 'config' in checkpoint:
    print(f"\n⚙️  模型配置:")
    for k, v in checkpoint['config'].items():
        print(f"  • {k}: {v}")

