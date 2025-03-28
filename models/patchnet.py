import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """带BN和LeakyReLU的残差块"""

    def __init__(self, in_channels, out_channels, bottleneck_ratio=2):
        super().__init__()
        hidden_dim = out_channels // bottleneck_ratio
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        out += residual
        return self.activation(out)


class PatchNet(nn.Module):
    """PatchNet核心网络结构"""
    # 代码是先进行去马赛克，然后将输出的rgb数据送到PatchNet进行打分。
    def __init__(self, patch_size=64, temperature=2, base_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.temperature = temperature
        self.stage_num = int(np.log2(patch_size))  # 64需要6个下采样阶段，目标是获取1x1的输出，后续的stage——RB中有pooling。

        # 初始卷积
        self.init_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        # 构建下采样阶段
        self.stages = nn.ModuleList()
        current_channels = base_channels
        for i in range(self.stage_num):
            stage = nn.Sequential(
                ResidualBlock(current_channels, current_channels * 2),
                nn.AvgPool2d(2)
            )
            self.stages.append(stage)
            current_channels *= 2

        # 最终卷积层
        self.final_conv = nn.Sequential(
            nn.Conv2d(current_channels, 1, 1),  # 回退最后一次通道扩展
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.init_conv(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)
        # 应用温度缩放
        x = torch.sigmoid(x * self.temperature)
        return x


class PatchNetWrapper(nn.Module):
    """训练封装器，包含损失计算逻辑"""
    # def __init__(self, restore_net, patch_size=64, lr=2.5e-4, device='cuda'):
    #     super().__init__()
    #     self.restore_net = restore_net.to(device)
    #     self.patchnet = PatchNet(patch_size).to(device)
    #     self.optimizer = torch.optim.Adam(
    #         list(self.restore_net.parameters()) + list(self.patchnet.parameters()),
    #         lr=lr
    #     )
    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         self.optimizer, T_0=10, T_mult=2
    #     )
    #     self.device = device
    def __init__(self, restore_net, patch_size=64, lr=2.5e-4, device='cuda'):
        super().__init__()
        self.restore_net = restore_net.to(device)
        self.patchnet = PatchNet(patch_size).to(device)
        # self.optimizer = torch.optim.Adam(
        #     list(self.restore_net.parameters()) + list(self.patchnet.parameters()),
        #     lr=lr
        # )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=10, T_mult=2
        # )
        self.device = device
        

    def compute_loss(self, pred, target, weights):
        """计算加权损失"""
        # 像素级MSE损失，每个像素都计算了损失。
        pixel_loss = F.mse_loss(pred, target, reduction='none').mean(dim=1, keepdim=True)

        # 自适应池化到权重图尺寸
        pool = nn.AdaptiveAvgPool2d(weights.shape[-2:])
        downsampled_loss = pool(pixel_loss)

        # 加权损失计算   downsampled_loss是对原始大小图像的分块池化    weights则是每个块对应的权重  这里是进行归一化。
        # 关键问题是权重是否有代表性，代表每个分块的的代表性。相当于给每个patch的loss分配了对应的权重，权重越大loss能进行更大的梯度更新。
        # 如果该区域是高频信息区域，那么重建的图像对高频的关注度也就越高。
        weighted_loss = (downsampled_loss * weights).sum() / (weights.sum() + 1e-8)
        return weighted_loss
    def compute_loss_liif(self, pred, target, weights):
        """计算加权损失"""
        # 像素级MSE损失，每个像素都计算了损失。
        l1_loss = nn.L1Loss(pred, target)
        print("why l1 loss : ", l1_loss)
        # 自适应池化到权重图尺寸
        pool = nn.AdaptiveAvgPool2d(weights.shape[-2:])
        downsampled_loss = pool(l1_loss)

        # 加权损失计算   downsampled_loss是对原始大小图像的分块池化    weights则是每个块对应的权重  这里是进行归一化。
        # 关键问题是权重是否有代表性，代表每个分块的的代表性。相当于给每个patch的loss分配了对应的权重，权重越大loss能进行更大的梯度更新。
        # 如果该区域是高频信息区域，那么重建的图像对高频的关注度也就越高。
        weighted_loss = (downsampled_loss * weights).sum() / (weights.sum() + 1e-8)
        print("why weighted loss : ", weighted_loss)
        return weighted_loss

    def train_step(self, noisy_input, clean_target):
        # 前向传播
        restored = self.restore_net(noisy_input)
        weights = self.patchnet(restored)

        # 计算损失
        loss = self.compute_loss(restored, clean_target, weights)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return restored, loss.item()
    
    def forward(self, inp, coord, cell, gt):
        # 前向传播
        restored = self.restore_net(inp, coord, cell)
        print("why restored: ", restored.shape)
        weights = self.patchnet(restored)
        
        print("why weigths: ", weights.shape)

        # 计算损失
        loss = self.compute_loss_liif(restored, gt, weights)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return restored, loss.item()

# 示例用法
if __name__ == "__main__":
    # 假设的RestoreNet（需替换为实际的demosaic网络）
    class DemoRestoreNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 64, 3, padding=1),  # 假设Bayer输入为4通道
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1)
            )

        def forward(self, x):
            return self.conv(x)


    # 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    restore_net = DemoRestoreNet()
    trainer = PatchNetWrapper(restore_net, device=device)

    # 模拟数据
    batch_size = 5
    noisy = torch.randn(batch_size, 4, 256, 256).to(device)  # Bayer格式输入
    clean = torch.randn(batch_size, 3, 256, 256).to(device)

    # 训练步骤
    loss = trainer.train_step(noisy, clean)
    print(f"Training loss: {loss:.4f}")