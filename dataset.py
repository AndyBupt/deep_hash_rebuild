"""
FVC2002 指纹数据集加载模块
文件命名格式: {person_id}_{sample_id}.bmp
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FVCDataset(Dataset):
    """
    FVC2002 指纹数据集
    支持多个DB合并加载，自动构建身份标签
    """

    def __init__(self, db_dirs, transform=None):
        """
        Args:
            db_dirs: list of str, 数据库目录列表，如 ['fingerprints/DB1_B', 'fingerprints/DB2_B']
            transform: 图像变换
        """
        self.transform = transform
        self.samples = []   # (image_path, label)
        self.person_ids = []

        # 收集所有样本，person_id 全局唯一化
        global_label_map = {}
        global_label = 0

        for db_dir in db_dirs:
            db_path = Path(db_dir)
            tif_files = sorted(db_path.glob("*.bmp"))
            for f in tif_files:
                # 文件名格式: 101_1.bmp -> person_id=101, sample_id=1
                stem = f.stem  # e.g. "101_1"
                person_id = stem.split("_")[0]
                # 用 db_dir+person_id 作为全局唯一key，避免不同DB的相同ID冲突
                key = f"{db_dir}/{person_id}"
                if key not in global_label_map:
                    global_label_map[key] = global_label
                    global_label += 1
                self.samples.append((str(f), global_label_map[key]))

        self.num_classes = global_label
        print(f"数据集加载完成: {len(self.samples)} 张图片, {self.num_classes} 个身份")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train=True):
    """获取图像变换"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_dataloaders(data_root, db_names, train_ratio=0.7, batch_size=8, seed=42):
    """
    构建训练/测试 DataLoader
    按人划分，保证训练集和测试集的人完全不重叠

    Args:
        data_root: 数据根目录
        db_names: 使用的DB列表，如 ['DB1_B', 'DB2_B']
        train_ratio: 训练集人数比例
        batch_size: batch大小
        seed: 随机种子

    Returns:
        train_loader, test_loader, num_classes
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    db_dirs = [os.path.join(data_root, db) for db in db_names]

    # 先扫描所有人的样本
    person_samples = {}  # key -> [(path, key)]
    for db_dir in db_dirs:
        db_path = Path(db_dir)
        for f in sorted(db_path.glob("*.bmp")):
            person_id = f.stem.split("_")[0]
            key = f"{db_dir}/{person_id}"
            if key not in person_samples:
                person_samples[key] = []
            person_samples[key].append(str(f))

    all_keys = sorted(person_samples.keys())
    random.shuffle(all_keys)

    n_train = max(1, int(len(all_keys) * train_ratio))
    train_keys = set(all_keys[:n_train])
    test_keys = set(all_keys[n_train:])

    # 构建全局标签映射
    label_map = {k: i for i, k in enumerate(all_keys)}
    num_classes = len(all_keys)

    train_samples = []
    test_samples = []
    for key in all_keys:
        label = label_map[key]
        for path in person_samples[key]:
            if key in train_keys:
                train_samples.append((path, label))
            else:
                test_samples.append((path, label))

    print(f"训练集: {len(train_keys)} 人, {len(train_samples)} 张")
    print(f"测试集: {len(test_keys)} 人, {len(test_samples)} 张")

    train_dataset = _ListDataset(train_samples, get_transforms(train=True))
    test_dataset = _ListDataset(test_samples, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, test_loader, num_classes


class _ListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
