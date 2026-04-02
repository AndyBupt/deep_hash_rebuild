"""
训练脚本
对应论文 Section IV-A: 两步训练 + continuation method

训练流程:
  Step1: 冻结backbone，只训练哈希层+分类头
  Step2: 端到端微调整个网络
  每个step内: beta 从1逐步增大（continuation method）
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import build_dataloaders
from model import FingerprintHashNet, HashingLoss


DATA_ROOT = "/root/autodl-tmp/FVC2004"
DB_NAMES = [
    "DB1_A/image", "DB1_B/image", 
    "DB2_A/image", "DB2_B/image", 
    "DB3_A/image", "DB3_B/image"
    ]
HASH_DIM = 1024
BATCH_SIZE = 32
SAVE_DIR = "checkpoints"

# 论文超参（FCA架构）
ALPHA, BETA_LOSS, GAMMA = 8.0, 2.0, 2.0

# Continuation method: beta 序列
BETA_SCHEDULE = [1, 2, 4, 8, 16, 32]


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = total_e1 = total_e2 = total_e3 = 0.0
    correct = total = 0

    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, hash_codes, _ = model(imgs)
        loss, e1, e2, e3 = criterion(logits, hash_codes, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_e1 += e1.item()
        total_e2 += e2.item()
        total_e3 += e3.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = len(loader)
    acc = correct / total * 100
    print(f"  loss={total_loss/n:.4f} E1={total_e1/n:.4f} "
          f"E2={total_e2/n:.4f} E3={total_e3/n:.4f} acc={acc:.1f}%")
    return total_loss / n


def train_step(model, train_loader, device, criterion,
               epochs, lr, lr_step_size, freeze_backbone=False, desc=""):
    """
    单步训练（对应论文两步之一）

    论文 Section IV-D (FCA):
      Step1: lr=0.1，每20 epoch衰减到90%，训练 JRL（fc3+classifier），冻结backbone
      Step2: lr=0.07（继承Step1最终lr），每5 epoch衰减到90%，端到端微调

    freeze_backbone=True 时只训练 fc3 + classifier（对应论文的 JRL 训练）
    """
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False
        trainable = list(model.fc3.parameters()) + list(model.classifier.parameters())
        print(f"[{desc}] 冻结backbone，只训练 fc3 + classifier (JRL)")
    else:
        for param in model.parameters():
            param.requires_grad = True
        trainable = list(model.parameters())
        print(f"[{desc}] 端到端微调全部参数")

    optimizer = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    # 论文: lr 每 step_size epoch 衰减到 90%
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.9)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss

    # 返回最终学习率（供 Step2 继承）
    final_lr = optimizer.param_groups[0]['lr']
    return best_loss, final_lr


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 构建数据集
    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=BATCH_SIZE
    )
    print(f"身份数量: {num_classes}")

    # 初始化模型
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=HASH_DIM, pretrained=True)
    model = model.to(device)
    criterion = HashingLoss(alpha=ALPHA, beta_loss=BETA_LOSS, gamma=GAMMA)

    # =========================================================
    # Step 1: 冻结backbone，只训练 JRL（fc3 + classifier）
    # 论文: 65 epochs, lr=0.1, 每20 epoch衰减到90%
    # 适配小数据: 每个beta跑10 epoch，共6×10=60 epoch
    # =========================================================
    print("\n========== Step 1: 训练 JRL（fc3 + classifier）==========")
    step1_lr = 0.001   # 论文初始lr
    final_lr = step1_lr
    for beta in BETA_SCHEDULE:
        model.set_beta(beta)
        print(f"\n--- beta={beta} ---")
        _, final_lr = train_step(
            model, train_loader, device, criterion,
            epochs=10,
            lr=final_lr,          # 继承上一个beta的最终lr
            lr_step_size=20,      # 论文: 每20 epoch衰减（适配小数据缩短为每5 epoch）
            freeze_backbone=True,
            desc=f"Step1 beta={beta}"
        )

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "step1_final.pth"))
    print(f"Step1 权重已保存，最终lr={final_lr:.6f}")

    # =========================================================
    # Step 2: 端到端微调
    # 论文: 25 epochs, lr=0.07（Step1最终lr），每5 epoch衰减到90%
    # =========================================================
    print("\n========== Step 2: 端到端微调 ==========")
    # 论文: Step2 初始lr = Step1 最终lr（约0.07）
    step2_lr = final_lr
    print(f"Step2 初始lr={step2_lr:.6f}（继承Step1最终lr，论文约0.07）")
    for beta in BETA_SCHEDULE:
        model.set_beta(beta)
        print(f"\n--- beta={beta} ---")
        _, step2_lr = train_step(
            model, train_loader, device, criterion,
            epochs=5,
            lr=step2_lr,
            lr_step_size=5,       # 论文: 每5 epoch衰减到90%
            freeze_backbone=False,
            desc=f"Step2 beta={beta}"
        )

    final_path = os.path.join(SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n最终模型已保存: {final_path}")

    return model, test_loader, num_classes, device


if __name__ == "__main__":
    main()
