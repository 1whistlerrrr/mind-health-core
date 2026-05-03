
"""
配置模块 - 支持不同场景下的参数设置
包括本地（Mac）、云服务器、演示模式等
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型架构配置"""
    vocab_size: int = 4000          # 词表大小
    dim: int = 256                  # 词嵌入维度
    n_layers: int = 4               # Transformer层数
    n_heads: int = 8                # 注意力头数
    max_seq_len: int = 64           # 最大序列长度
    hidden_dim_multiplier: int = 4  # FFN中间层倍数


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_every: int = 5            # 每多少epoch保存一次
    eval_every: int = 2            # 每多少epoch评估一次


@dataclass
class Config:
    """完整配置"""
    # 运行环境
    device: str = "auto"           # auto, cpu, mps, cuda
    
    # 模型和训练配置
    model: ModelConfig = None
    train: TrainConfig = None
    
    # 文件路径
    output_dir: str = "./output"   # 输出目录
    data_path: str = "./chinese.txt"
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.train is None:
            self.train = TrainConfig()


def get_config(mode: str = "demo") -&gt; Config:
    """
    获取不同场景下的配置
    
    Args:
        mode: 模式选择
            - 'demo': 快速演示（默认）
            - 'local': 本地Mac开发
            - 'cloud': 云服务器（假设GPU充足）
            - 'tiny': 超小模型用于测试
    """
    if mode == "demo":
        # 快速演示 - 小模型，小batch，少epochs
        return Config(
            model=ModelConfig(
                vocab_size=1000,
                dim=128,
                n_layers=2,
                n_heads=4,
                max_seq_len=32
            ),
            train=TrainConfig(
                batch_size=8,
                epochs=50,
                save_every=10
            )
        )
    
    elif mode == "local":
        # 本地Mac开发 - 适中配置
        return Config(
            device="mps",
            model=ModelConfig(
                vocab_size=4000,
                dim=256,
                n_layers=4,
                n_heads=8,
                max_seq_len=64
            ),
            train=TrainConfig(
                batch_size=16,
                epochs=30,
                save_every=5
            )
        )
    
    elif mode == "cloud":
        # 云服务器 - 可以开更大的模型
        return Config(
            device="cuda",
            model=ModelConfig(
                vocab_size=8000,
                dim=512,
                n_layers=8,
                n_heads=16,
                max_seq_len=256
            ),
            train=TrainConfig(
                batch_size=32,
                epochs=50,
                save_every=5
            )
        )
    
    elif mode == "tiny":
        # 超小模型用于测试 - 几秒钟就能跑完
        return Config(
            model=ModelConfig(
                vocab_size=500,
                dim=64,
                n_layers=1,
                n_heads=2,
                max_seq_len=16
            ),
            train=TrainConfig(
                batch_size=4,
                epochs=10,
                save_every=5
            )
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def print_config(config: Config):
    """打印配置信息"""
    print("\n" + "=" * 60)
    print("📋 MiniMind 配置信息")
    print("=" * 60)
    print(f"💻 设备: {config.device}")
    print(f"\n🤖 模型配置:")
    print(f"  - 词表大小: {config.model.vocab_size}")
    print(f"  - 嵌入维度: {config.model.dim}")
    print(f"  - 层数: {config.model.n_layers}")
    print(f"  - 注意力头数: {config.model.n_heads}")
    print(f"  - 最大序列长度: {config.model.max_seq_len}")
    
    total_params = (
        # 嵌入层
        config.model.vocab_size * config.model.dim
        # 每层: Attention + FFN + 2个Norm
        + config.model.n_layers * (
            # Attention: Q,K,V,O
            4 * config.model.dim * config.model.dim
            # FFN: w1, w2, w3
            + 3 * config.model.dim * (config.model.dim * config.model.hidden_dim_multiplier)
            # RMSNorm weights
            + 2 * config.model.dim
        )
        # 最终Norm
        + config.model.dim
        # 输出层
        + config.model.dim * config.model.vocab_size
    )
    print(f"  - 总参数量: {total_params:,}")
    
    print(f"\n🎯 训练配置:")
    print(f"  - Batch size: {config.train.batch_size}")
    print(f"  - Epochs: {config.train.epochs}")
    print(f"  - 学习率: {config.train.lr}")
    print(f"  - 保存频率: 每 {config.train.save_every} epochs")
    print("=" * 60 + "\n")
