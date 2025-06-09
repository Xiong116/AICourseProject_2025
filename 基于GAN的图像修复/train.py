import torch
import torch.nn as nn
from torch import optim
from data_loader import get_dataloader
from models import Generator, Discriminator
from config import config
from utils import save_sample_images
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm
import logging
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), path)

def train_epoch(generator, discriminator, train_loader, opt_g, opt_d, 
                adversarial_loss, l1_loss, scaler_g, scaler_d, epoch, writer):
    generator.train()
    discriminator.train()
    total_g_loss = 0
    total_d_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    for i, batch in enumerate(pbar):
        real = batch['real'].to(config.device)
        masked = batch['masked'].to(config.device)
        mask = batch['mask'].to(config.device)

        # ===== 训练判别器 =====
        opt_d.zero_grad()
        with autocast(device_type='cuda'):
            # 真实图像损失
            validity_real = discriminator(real, mask)
            d_real = adversarial_loss(validity_real, torch.ones_like(validity_real))
            
            # 生成图像损失
            fake = generator(masked, mask)
            validity_fake = discriminator(fake.detach(), mask)
            d_fake = adversarial_loss(validity_fake, torch.zeros_like(validity_fake))
            d_total = (d_real + d_fake) / 2

        scaler_d.scale(d_total).backward()
        scaler_d.unscale_(opt_d)
        nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip)
        scaler_d.step(opt_d)
        scaler_d.update()

        # ===== 训练生成器 =====
        opt_g.zero_grad()
        with autocast(device_type='cuda'):
            # 重新生成图像
            fake = generator(masked, mask)
            validity = discriminator(fake, mask)
            
            # 动态调整重建损失的权重
            rec_weight = max(0.1, min(100.0, 100.0 * (1.0 - epoch / config.epochs)))
            g_adv = adversarial_loss(validity, torch.ones_like(validity))
            g_rec = l1_loss(fake, real) * rec_weight
            
            # 感知损失
            g_perceptual = l1_loss(fake, real) * 10.0  # 感知损失权重
            
            g_total = g_adv + g_rec + g_perceptual

        scaler_g.scale(g_total).backward()
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
        scaler_g.step(opt_g)
        scaler_g.update()

        # 更新进度条
        total_g_loss += g_total.item()
        total_d_loss += d_total.item()
        pbar.set_postfix({
            'G_loss': f'{g_total.item():.4f}',
            'D_loss': f'{d_total.item():.4f}',
            'G_adv': f'{g_adv.item():.4f}',
            'G_rec': f'{g_rec.item():.4f}'
        })

        # 记录到TensorBoard
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('Loss/Generator_Total', g_total.item(), global_step)
        writer.add_scalar('Loss/Discriminator_Total', d_total.item(), global_step)
        writer.add_scalar('Loss/Generator_Adversarial', g_adv.item(), global_step)
        writer.add_scalar('Loss/Generator_Reconstruction', g_rec.item(), global_step)
        writer.add_scalar('Loss/Generator_Perceptual', g_perceptual.item(), global_step)
        writer.add_scalar('Weights/Reconstruction', rec_weight, global_step)

        # 保存样本
        if i % 500 == 0:
            save_sample_images(real, masked, fake, epoch, i)
            writer.add_images('Samples/Real', real, global_step)
            writer.add_images('Samples/Masked', masked, global_step)
            writer.add_images('Samples/Generated', fake, global_step)

    return total_g_loss / len(train_loader), total_d_loss / len(train_loader)

def validate(generator, discriminator, val_loader, adversarial_loss, l1_loss):
    generator.eval()
    discriminator.eval()
    total_g_loss = 0
    total_d_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            real = batch['real'].to(config.device)
            masked = batch['masked'].to(config.device)
            mask = batch['mask'].to(config.device)

            # 生成器验证
            fake = generator(masked, mask)
            validity = discriminator(fake, mask)
            g_adv = adversarial_loss(validity, torch.ones_like(validity))
            g_rec = l1_loss(fake, real) * 100
            g_total = g_adv + g_rec

            # 判别器验证
            validity_real = discriminator(real, mask)
            d_real = adversarial_loss(validity_real, torch.ones_like(validity_real))
            validity_fake = discriminator(fake, mask)
            d_fake = adversarial_loss(validity_fake, torch.zeros_like(validity_fake))
            d_total = (d_real + d_fake) / 2

            total_g_loss += g_total.item()
            total_d_loss += d_total.item()

    return total_g_loss / len(val_loader), total_d_loss / len(val_loader)

def main():
    # 创建TensorBoard日志目录
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', current_time)
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard日志保存在: {log_dir}")

    # 设备配置
    logger.info(f"=== 训练设备: {config.device} ===")
    if torch.cuda.device_count() > 1:
        logger.info(f"检测到 {torch.cuda.device_count()} 个GPU!")

    # 初始化模型
    generator = Generator().to(config.device)
    discriminator = Discriminator().to(config.device)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # 优化器配置
    opt_g = optim.Adam(generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
    opt_d = optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=config.step_size, gamma=config.gamma)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=config.step_size, gamma=config.gamma)

    # 混合精度训练
    scaler_g = GradScaler('cuda')
    scaler_d = GradScaler('cuda')

    # 损失函数
    adversarial_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # 获取数据加载器
    train_loader, val_loader = get_dataloader()
    logger.info(f"=== 开始训练，总epoch数: {config.epochs} ===")

    # 早停设置
    early_stopping_g = EarlyStopping(patience=5)
    early_stopping_d = EarlyStopping(patience=5)

    # 创建检查点目录
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config.epochs):
        # 训练一个epoch
        train_g_loss, train_d_loss = train_epoch(
            generator, discriminator, train_loader, opt_g, opt_d,
            adversarial_loss, l1_loss, scaler_g, scaler_d, epoch, writer
        )

        # 验证
        val_g_loss, val_d_loss = validate(
            generator, discriminator, val_loader, adversarial_loss, l1_loss
        )

        # 记录验证损失
        writer.add_scalar('Validation/Generator_Loss', val_g_loss, epoch)
        writer.add_scalar('Validation/Discriminator_Loss', val_d_loss, epoch)

        # 更新学习率
        scheduler_g.step()
        scheduler_d.step()

        # 记录学习率
        writer.add_scalar('Learning_Rate/Generator', scheduler_g.get_last_lr()[0], epoch)
        writer.add_scalar('Learning_Rate/Discriminator', scheduler_d.get_last_lr()[0], epoch)

        # 早停检查
        early_stopping_g(val_g_loss, generator, 
                        os.path.join(checkpoint_dir, 'best_generator.pth'))
        early_stopping_d(val_d_loss, discriminator, 
                        os.path.join(checkpoint_dir, 'best_discriminator.pth'))

        if early_stopping_g.early_stop and early_stopping_d.early_stop:
            logger.info("触发早停机制，停止训练")
            break

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_d_state_dict': opt_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict(),
                'train_g_loss': train_g_loss,
                'train_d_loss': train_d_loss,
                'val_g_loss': val_g_loss,
                'val_d_loss': val_d_loss
            }
            torch.save(checkpoint, 
                      os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        logger.info(f"Epoch {epoch + 1}/{config.epochs} - "
                   f"Train G: {train_g_loss:.4f} D: {train_d_loss:.4f} - "
                   f"Val G: {val_g_loss:.4f} D: {val_d_loss:.4f}")

    writer.close()
    logger.info("训练完成")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
    input("训练完成，按Enter键退出...")