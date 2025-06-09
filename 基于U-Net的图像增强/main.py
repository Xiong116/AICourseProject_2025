import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import LOLv2Dataset, PerceptualLoss, EarlyStopping, show_comparison_image
from model import UNetWithSkipConnections

if __name__ == '__main__':
    # switch=0训练模式,switch=1测试模式
    switch = 1

    # 数据集路径
    data_dir = "./data"
    data_list_path = "./data/data_list.txt"

    # 模型、日志保存路径
    save_path = "model.pth"
    save_log_path = "./logs/"

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 训练轮数
    num_epochs = 100

    # -------------------------
    # 数据集处理
    # -------------------------
    # 读取数据列表
    with open(data_list_path, 'r') as f:
        all_image_names = [line.strip() for line in f.readlines()]

    # 划分训练集、验证集和测试集（80% 训练集，10% 验证集，10% 测试集）
    train_val_image_names, test_image_names = train_test_split(all_image_names, test_size=0.1, random_state=42)
    train_image_names, val_image_names = train_test_split(train_val_image_names, test_size=0.111, random_state=42)  # 0.1 / 0.9 ≈ 0.111

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集和数据加载器
    train_dataset = LOLv2Dataset(image_names=train_image_names, data_dir=data_dir, transform=transform)
    val_dataset = LOLv2Dataset(image_names=val_image_names, data_dir=data_dir, transform=transform)
    test_dataset = LOLv2Dataset(image_names=test_image_names, data_dir=data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    # -------------------------
    # 训练过程
    # -------------------------
    if switch ==0:
        # 模型初始化
        model = UNetWithSkipConnections().to(device)

        # 损失函数初始化
        perceptual_loss_fn = PerceptualLoss().to(device)
        mse_loss_fn = nn.MSELoss()

        # 优化器初始化
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 早停函数初始化
        early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

        # TensorBoard 初始化
        # 以当前时间戳的形式保存tensorboard日志文件
        t = time.localtime()
        log_dir_name = f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}"
        log_path = os.path.join(save_log_path, log_dir_name)

        # 确保日志目录存在
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)

                # 前向传播
                outputs = model(inputs)
                perceptual_loss = perceptual_loss_fn(outputs, targets)
                mse_loss = mse_loss_fn(outputs, targets)
                total_loss = perceptual_loss + mse_loss

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                progress_bar.set_postfix(loss=total_loss.item())

            avg_train_loss = running_loss / len(train_loader)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    perceptual_loss = perceptual_loss_fn(outputs, targets)
                    mse_loss = mse_loss_fn(outputs, targets)
                    total_loss = perceptual_loss + mse_loss
                    val_loss += total_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 更新学习率调度器
            scheduler.step()

            # 获取当前学习率并记录到 TensorBoard
            current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
            writer.add_scalar('Learning Rate', current_lr, epoch)

            # 早停检查
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # 关闭 TensorBoard
        writer.close()
        print("Training complete.")

    # -------------------------
    # 测试图像
    # -------------------------
    else:
        # 读入模型
        model = UNetWithSkipConnections()
        model.load_state_dict(torch.load(save_path))
        model = model.to(device)
        model.eval()

        # 图像对比显示
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                for j in range(inputs.size(0)):
                    input_img = inputs[j]
                    output_img = outputs[j]
                    gt_img = targets[j]

                    show_comparison_image(input_img, output_img, gt_img)

                if i == 0:
                    print("Testing complete.")
                    break