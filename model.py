"""
指纹深度哈希网络模型
对应论文 Section III-B 和 Section IV

结构:
  FingerprintCNN (VGG-19 backbone + fc3)
      ↓ 1024维实值特征
  HashingLayer (tanh continuation → 二值码)
      ↓ 1024维二值向量
  Softmax分类头 (训练时用)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FingerprintHashNet(nn.Module):
    """
    指纹深度哈希网络
    - 骨干: VGG-19 (ImageNet预训练)
    - 额外增加 fc3 层输出 hash_dim 维特征
    - 哈希层: tanh(beta * x)，beta 从1逐步增大趋近 sign()
    - 分类头: softmax，训练时使用
    """

    def __init__(self, num_classes, hash_dim=1024, pretrained=True):
        """
        Args:
            num_classes: 身份数量（分类头输出维度）
            hash_dim: 哈希码长度，论文中为1024
            pretrained: 是否使用ImageNet预训练权重
        """
        super().__init__()
        self.hash_dim = hash_dim

        # VGG-19 backbone，去掉原始分类器
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg.features
        self.avgpool = vgg.avgpool

        # fc1, fc2: 沿用 VGG-19 原始 classifier 的前两个全连接层
        # 直接加载 ImageNet 预训练权重（论文做法）
        # VGG-19 classifier 结构: [Linear(25088,4096), ReLU, Dropout,
        #                           Linear(4096,4096),  ReLU, Dropout,
        #                           Linear(4096,1000)]
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        if pretrained:
            # 从 VGG-19 预训练 classifier 加载 fc1/fc2 权重
            self.fc1[0].weight.data = vgg.classifier[0].weight.data.clone()
            self.fc1[0].bias.data   = vgg.classifier[0].bias.data.clone()
            self.fc2[0].weight.data = vgg.classifier[3].weight.data.clone()
            self.fc2[0].bias.data   = vgg.classifier[3].bias.data.clone()

        # fc3: 论文中新增的全连接层，输出 hash_dim 维（随机初始化）
        self.fc3 = nn.Sequential(
            nn.Linear(4096, hash_dim),
            nn.BatchNorm1d(hash_dim),
        )

        # 哈希层: 使用 tanh(beta*x)，beta 可调
        self.beta = 1.0
        # 分类头（softmax，训练时使用）
        self.classifier = nn.Linear(hash_dim, num_classes)

        # 初始化新增层（fc3 和 classifier）
        nn.init.kaiming_normal_(self.fc3[0].weight)
        nn.init.constant_(self.fc3[0].bias, 0)
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        """
        Returns:
            logits: (B, num_classes) 分类输出，用于计算 E1 损失
            hash_codes: (B, hash_dim) 哈希层输出（实值，范围[-1,1]），用于 E2/E3 损失
            binary_codes: (B, hash_dim) sign(hash_codes)，推理时使用
        """
        # 特征提取
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # (B, hash_dim)

        # 哈希层: tanh(beta * x)
        hash_codes = torch.tanh(self.beta * x)  # 实值，范围 (-1, 1)

        # 分类头
        logits = self.classifier(hash_codes)

        # 二值码（推理用）
        binary_codes = torch.sign(hash_codes)

        return logits, hash_codes, binary_codes

    def set_beta(self, beta):
        """更新 tanh 的 beta 参数（continuation method）"""
        self.beta = beta

    def get_binary_codes(self, x):
        """只返回二值哈希码，推理时使用"""
        _, _, binary = self.forward(x)
        return binary


class HashingLoss(nn.Module):
    """
    论文公式 (6): alpha*E1 + beta_loss*E2 + gamma*E3

    E1: 分类交叉熵损失（语义保持）
    E2: 激活值远离0的约束（二值化促进）
    E3: 激活值均值为0的约束（平衡性）
    """

    def __init__(self, alpha=8.0, beta_loss=2.0, gamma=2.0):
        """
        论文FCA架构超参: alpha=8, beta=2, gamma=2
        """
        super().__init__()
        self.alpha = alpha
        self.beta_loss = beta_loss
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, hash_codes, labels):
        """
        Args:
            logits: (B, num_classes)
            hash_codes: (B, hash_dim) 哈希层实值输出
            labels: (B,) 身份标签

        Returns:
            total_loss, e1, e2, e3
        """
        _, J = hash_codes.shape

        # E1: 分类交叉熵
        e1 = self.ce_loss(logits, labels)

        # E2: 最大化激活值的平方和（促进二值化）
        # 论文公式(4): E2 = -1/J * sum(||h_n||^2)
        e2 = -(hash_codes ** 2).mean()

        # E3: 最小化每个样本激活均值的平方（平衡性）
        # 论文公式(5): E3 = sum((mean(h_n))^2)
        e3 = (hash_codes.mean(dim=1) ** 2).mean()

        total = self.alpha * e1 + self.beta_loss * e2 + self.gamma * e3
        return total, e1, e2, e3
