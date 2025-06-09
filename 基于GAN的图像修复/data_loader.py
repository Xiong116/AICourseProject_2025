import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from config import config
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CelebAMaskHQ(Dataset):
    def __init__(self, root, transform=None, is_train=True):
        self.root = root
        self.transform = transform
        self.is_train = is_train
        
        # 检查数据集目录
        if not os.path.exists(root):
            raise FileNotFoundError(f"数据集目录 '{root}' 不存在")
        
        # 获取所有图片文件
        self.image_files = [f for f in os.listdir(root) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not self.image_files:
            raise RuntimeError(f"在目录 '{root}' 中没有找到图片文件")
        
        logger.info(f"加载数据集: {len(self.image_files)} 张图片")

    def random_mask(self, img_size, mask_size):
        h, w = config.img_size, config.img_size
        mask = torch.ones((1, h, w))
        
        # 随机生成多个掩码区域
        num_masks = np.random.randint(1, 4)  # 1-3个掩码区域
        for _ in range(num_masks):
            y = np.random.randint(0, h - mask_size)
            x = np.random.randint(0, w - mask_size)
            mask[:, y:y + mask_size, x:x + mask_size] = 0
            
        return mask

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.image_files[index])
        try:
            image = Image.open(img_path).convert('RGB')
            
            # 数据验证
            if image.size[0] < config.img_size or image.size[1] < config.img_size:
                raise ValueError(f"图片尺寸过小: {image.size}")
            
            if self.transform:
                image = self.transform(image)
            
            mask = self.random_mask(config.img_size, config.mask_size)
            masked_image = image * mask
            
            return {
                "real": image,
                "masked": masked_image,
                "mask": mask,
                "path": img_path
            }
        except Exception as e:
            logger.error(f"处理图片 {img_path} 时出错: {str(e)}")
            # 返回一个默认值或重新尝试
            return self.__getitem__((index + 1) % len(self))

    def __len__(self):
        return len(self.image_files)

def get_transforms(is_train=True):
    """获取数据转换和增强"""
    if is_train:
        return transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.RandomCrop(config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_dataloader():
    """获取训练和验证数据加载器"""
    try:
        # 创建数据集
        full_dataset = CelebAMaskHQ(
            config.dataset_path,
            transform=get_transforms(is_train=True)
        )
        
        # 分割训练集和验证集
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 设置验证集转换
        val_dataset.dataset.transform = get_transforms(is_train=False)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        logger.info(f"创建数据加载器成功 - 训练集: {len(train_dataset)} 样本, "
                   f"验证集: {len(val_dataset)} 样本")
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"创建数据加载器时出错: {str(e)}")
        raise