import torch
import torch.nn as nn
import os
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import VGG16_Weights

# 数据读取函数
class LOLv2Dataset(Dataset):
    def __init__(self, image_names, data_dir, transform=None):
        self.image_names = image_names
        self.data_dir = data_dir
        self.transform = transform
        self.low_light_dir = os.path.join(data_dir, "input")
        self.high_light_dir = os.path.join(data_dir, "gt")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        low_light_path = os.path.join(self.low_light_dir, image_name)
        high_light_path = os.path.join(self.high_light_dir, image_name)

        low_light_img = Image.open(low_light_path).convert("RGB")
        high_light_img = Image.open(high_light_path).convert("RGB")

        if self.transform:
            low_light_img = self.transform(low_light_img)
            high_light_img = self.transform(high_light_img)

        return low_light_img, high_light_img


# 感知损失函数
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        # 使用新的 weights 参数加载预训练的 VGG16 模型
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]  # 使用 VGG16 的前 16 层

        # 冻结 VGG16 的参数
        for param in vgg.parameters():
            param.requires_grad = False

        # 将 VGG16 设置为评估模式
        self.vgg = vgg.eval()

        # 定义 MSE 损失函数
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # 提取特征
        output_features = self.vgg(output)
        target_features = self.vgg(target)

        # 计算感知损失
        return self.mse_loss(output_features, target_features)


    # 测试图像显示
def show_comparison_image(input_img, output_img, gt_img):
    """
    显示 input、output 和 gt 的对比图。

    Args:
        input_img (Tensor): 输入图像（低光照图像）。
        output_img (Tensor): 模型输出图像（增强后的图像）。
        gt_img (Tensor): 真实值图像（高光照图像）。
    """
    # 反归一化函数
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
        return tensor * std + mean

    # 对比度拉伸函数
    def stretch_contrast(image):
        image_min = image.min()
        image_max = image.max()
        return (image - image_min) / (image_max - image_min + 1e-6)

    # 将张量转换为 numpy 数组并反归一化
    input_img = denormalize(input_img).permute(1, 2, 0).cpu().detach().numpy()
    output_img = denormalize(output_img).permute(1, 2, 0).cpu().detach().numpy()
    gt_img = denormalize(gt_img).permute(1, 2, 0).cpu().detach().numpy()

    # 对输出图像进行对比度拉伸
    output_img = stretch_contrast(output_img)

    # 裁剪到 [0, 1] 范围
    input_img = np.clip(input_img, 0, 1)
    output_img = np.clip(output_img, 0, 1)
    gt_img = np.clip(gt_img, 0, 1)

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_img)
    axes[0].set_title("Input (Low Light)")
    axes[0].axis("off")

    axes[1].imshow(output_img, vmin=0, vmax=1)  # 固定颜色范围
    axes[1].set_title("Output (Enhanced)")
    axes[1].axis("off")

    axes[2].imshow(gt_img, vmin=0, vmax=1)  # 固定颜色范围
    axes[2].set_title("Ground Truth (High Light)")
    axes[2].axis("off")

    # 显示图像
    plt.show()

# 早停法实现函数
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss