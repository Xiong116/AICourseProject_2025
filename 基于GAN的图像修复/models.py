import torch
import torch.nn as nn
from config import config


def weights_init(m):
    """自定义权重初始化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def down_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def up_block(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return layers

        # 下采样部分
        self.down1 = nn.Sequential(*down_block(4, config.ngf, False))
        self.down2 = nn.Sequential(*down_block(config.ngf, config.ngf * 2))
        self.down3 = nn.Sequential(*down_block(config.ngf * 2, config.ngf * 4))
        self.down4 = nn.Sequential(*down_block(config.ngf * 4, config.ngf * 8))
        self.down5 = nn.Sequential(
            nn.Conv2d(config.ngf * 8, config.ngf * 8, 4, 2, 1),
            nn.ReLU()
        )

        # 上采样部分
        self.up1 = nn.Sequential(*up_block(config.ngf * 8, config.ngf * 8, True))
        self.up2 = nn.Sequential(*up_block(config.ngf * 8 * 2, config.ngf * 4))
        self.up3 = nn.Sequential(*up_block(config.ngf * 4 * 2, config.ngf * 2))
        self.up4 = nn.Sequential(*up_block(config.ngf * 2 * 2, config.ngf))
        self.final = nn.Sequential(
            nn.ConvTranspose2d(config.ngf * 2, config.out_channels, 4, 2, 1),
            nn.Tanh()
        )

        # 应用权重初始化
        self.apply(weights_init)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], 1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], 1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], 1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], 1)
        return self.final(u4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *block(4, config.ndf, False),
            *block(config.ndf, config.ndf * 2),
            *block(config.ndf * 2, config.ndf * 4),
            *block(config.ndf * 4, config.ndf * 8),
            nn.Conv2d(config.ndf * 8, 1, 4, 1, 1)
        )

        # 应用权重初始化
        self.apply(weights_init)

    def forward(self, img, mask):
        return self.model(torch.cat([img, mask], 1))