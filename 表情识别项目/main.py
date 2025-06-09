import torch
import utils
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import kornia.augmentation as K

# 假设 model.py 中定义了不同的模型类或函数来创建模型
from model import get_model  # 需要在model.py中定义get_model函数


if __name__ == '__main__':
    config = utils.read_config()
    # 设置随机种子
    utils.set_seed(config['seed'])

    # 以当前时间戳的形式保存tensorboard日志文件
    t = time.localtime()
    log_dir_name = f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}"
    log_path = os.path.join(config['log_path'], log_dir_name)

    # 确保日志目录存在
    os.makedirs(log_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_path)

    # 读取数据集中所有图像的地址：
    train_images_path = os.path.join(config['save_path'], 'train_set.npy')
    train_labels_path = os.path.join(config['save_path'], 'train_labels.npy')
    test_images_path = os.path.join(config['save_path'],'test_set.npy')
    test_labels_path = os.path.join(config['save_path'], 'test_labels.npy')
    train_images = np.load(train_images_path)
    train_labels = np.load(train_labels_path)
    test_labels = np.load(test_labels_path)
    test_images = np.load(test_images_path)

    # 验证集比例
    val_rate = config['val_rate']

    # 使用 train_test_split 划分数据
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=val_rate, random_state=config['seed'], stratify=train_labels
    )

    # 从配置文件中获取批次大小
    batch_size = config['batch_size']

    # 设置数据加载的工作核心数（不超过 CPU 核心数）
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0])
    print(f"Using {nw} workers for data loading.")

    # 数据预处理
    # 数据预处理 - 使用Kornia进行GPU加速的数据增强
    data_transform = {
        "train": K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ).to('cuda'),  # 将数据增强序列放到GPU上
        "val": K.AugmentationSequential(
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ).to('cuda')  # 将数据增强序列放到GPU上
    }

    # 构建训练数据集
    train_dataset = utils.MyDataSet(
        images=train_images,
        labels=train_labels,
    )

    # 构建验证数据集
    val_dataset = utils.MyDataSet(
        images=val_images,
        labels=val_labels,
    )

    # 构建测试数据集
    test_dataset = utils.MyDataSet(
        images=test_images,
        labels=test_labels,
    )

    # 构建训练、验证、测试数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=test_dataset.collate_fn
    )

    # 定义要对比的模型列表
    models_list = ['ResNet18', 'VGG11', 'MobileNetV2']
    results = {}

    for model_name in models_list:
        # 实例化模型
        train_model = get_model(model_name, num_classes=config['num_class'])
        if config['switch'] == 1:
            save_path = config['model_save_path']
            checkpoint = torch.load(save_path)
            train_model.load_state_dict(checkpoint['model_state_dict'])

        # 将训练模型设置在显卡上
        train_model = train_model.cuda()

        # 设置优化器
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(train_model.parameters(),
                                       lr=config['learning_rate'],
                                       betas=(0.9, 0.999),
                                       eps=1e-08,
                                       weight_decay=0,
                                       amsgrad=False)
        elif config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(train_model.parameters(),
                                       lr=config['learning_rate'],
                                       momentum=0.9,
                                       dampening=0,
                                       weight_decay=0,
                                       nesterov=False)
        else:
            raise ValueError("Optimizer must be Adam or SGD, got {}".format(config['optimizer']))

        # 学习率衰减
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)

        best_val_acc = 0.0

        for epoch in range(1, config['epochs'] + 1):
            train_loss, train_acc, val_acc = utils.trainer(train_model, optimizer, train_loader, val_loader, config, epoch, model_name=model_name, transform=data_transform["train"])

            # 记录每一代的训练日志，并保存在 tensorboard 文件内
            tags = ['train_loss', 'train_acc', 'val_acc', 'learning_rate']
            tb_writer.add_scalar(f'{model_name}/{tags[0]}', train_loss, epoch)
            tb_writer.add_scalar(f'{model_name}/{tags[1]}', round(train_acc * 100, 4), epoch)
            tb_writer.add_scalar(f'{model_name}/{tags[2]}', round(val_acc * 100, 4), epoch)
            tb_writer.add_scalar(f'{model_name}/{tags[3]}', optimizer.param_groups[0]['lr'], epoch)

            # 更新学习率
            scheduler.step()

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'model_state_dict': train_model.state_dict()},
                           os.path.join(config['save_path'], f"{model_name}_best.pth"))

        # 测试模型并记录最佳准确率
        test_acc = utils.evaluater(train_model, test_loader, config, transform=data_transform["val"])
        results[model_name] = {'best_val_acc': round(best_val_acc * 100, 2), 'test_acc': round(test_acc * 100, 2)}

    with open(os.path.join(config['save_path'], 'model_comparison_results.txt'), 'w') as f:
        for model_name, metrics in results.items():
            line = f"Model: {model_name}, Best Val Acc: {metrics['best_val_acc']}%, Test Acc: {metrics['test_acc']}%\n"
            print(line, end='')  # 打印到控制台
            f.write(line)  # 写入文件

    # 输出模型对比结果
    print("\nModel Comparison:")
    for model_name, metrics in results.items():
        print(f"Model: {model_name}, Best Val Acc: {metrics['best_val_acc']}%, Test Acc: {metrics['test_acc']}%")