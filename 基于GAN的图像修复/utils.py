import os
import torch
import torchvision


def save_sample_images(real, masked, fake, epoch, batch_i):
    """保存样本图像（自动处理设备）"""
    os.makedirs("samples", exist_ok=True)

    # 转换到CPU并解除梯度
    real = real.cpu().detach()
    masked = masked.cpu().detach()
    fake = fake.cpu().detach()

    comparison = torch.cat([real[:3], masked[:3], fake[:3]])
    grid = torchvision.utils.make_grid(comparison, nrow=3, normalize=True)

    torchvision.utils.save_image(
        grid,
        f"samples/epoch_{epoch}_batch_{batch_i}.png"
    )