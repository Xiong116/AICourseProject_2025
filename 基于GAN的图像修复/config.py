import torch

class Config:
    # 数据集设置
    dataset_path = "./images"
    img_size = 256
    mask_size = 64

    # 训练参数
    batch_size = 16
    epochs = 100
    lr_g = 0.0001
    lr_d = 0.0004
    beta1 = 0.5
    beta2 = 0.999

    # 优化参数
    grad_clip = 1.0          # 梯度裁剪阈值
    step_size = 10           # 学习率衰减周期
    gamma = 0.5              # 学习率衰减系数

    # 模型参数
    in_channels = 3
    out_channels = 3
    latent_dim = 100
    ngf = 64
    ndf = 64

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()