"""
分析脚本：
1. 绘制完整 G-S 曲线（K 从小到大，展示 GAR 从100%下降的完整过程）
2. 分析每个 bit 位置的翻转率，找出稳定比特 vs 不稳定比特
3. 为改进 CTM 提供数据支撑
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
import matplotlib.pyplot as plt

from dataset import build_dataloaders, get_transforms
from model import FingerprintHashNet
from ctm import CTM
from sstm import SSTM


DATA_ROOT = "/root/autodl-tmp/FVC2004"
DB_NAMES = [
    "DB1_A/image", "DB1_B/image", 
    "DB2_A/image", "DB2_B/image", 
    "DB3_A/image", "DB3_B/image"
    ]
OUTPUT_DIR = "results"


def load_model_and_data(model_path=None):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8
    )
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024, pretrained=False)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型: {model_path}")
    model = model.to(device)
    model.set_beta(32)
    return model, train_loader, test_loader, num_classes, device


def extract_codes(model, loader, device):
    """提取所有样本的二值哈希码"""
    model.eval()
    all_codes, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, _, binary = model(imgs)
            all_codes.append(binary.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_codes), np.concatenate(all_labels)


# ============================================================
# 分析一：完整 G-S 曲线
# ============================================================
def plot_full_gs_curve(codes, labels, G=512, output_dir=OUTPUT_DIR):
    """
    绘制完整 G-S 曲线：K 从 7 到 N-1，展示 GAR 从 100% 下降到 0% 的完整过程
    这是展示"模型不足"的关键图
    """
    os.makedirs(output_dir, exist_ok=True)
    N = G // 8
    ctm = CTM(hash_dim=1024, G=G)

    # 用较稀疏的 K 采样，避免计算太慢
    K_list = list(range(7, N, 2))  # 步长2
    gars_sk = []
    print(f"\n计算完整 G-S 曲线 (G={G}, N={N}, K从7到{K_list[-1]})...")

    for K in K_list:
        sstm = SSTM(G=G, K=K)
        # 只算 stolen key 场景的 genuine GAR
        unique_ids = np.unique(labels)
        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm.enroll(codes[idx[0]])
            stored_hash, _ = sstm.enroll(re)
            for i in idx[1:]:
                rp = ctm.authenticate(codes[i], ke)
                is_genuine, _ = sstm.authenticate(rp, stored_hash)
                pass_count += int(is_genuine)
                total += 1
        gar = pass_count / total * 100 if total > 0 else 0
        gars_sk.append(gar)
        k_bits = K * 8
        print(f"  K={K:3d} (k={k_bits:4d} bits): GAR={gar:.1f}%")

    k_bits_list = [K * 8 for K in K_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_bits_list, gars_sk, 'b-o', linewidth=2, markersize=4)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('安全性 k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'完整 G-S 曲线 (G={G} bits) — 随机比特选择（基线）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # 标注 GAR=50% 对应的 k 值
    for i, (k, gar) in enumerate(zip(k_bits_list, gars_sk)):
        if gar < 50 and i > 0 and gars_sk[i-1] >= 50:
            ax.axvline(x=k, color='red', linestyle=':', alpha=0.7)
            ax.annotate(f'k={k}bits\nGAR开始<50%',
                        xy=(k, 50), xytext=(k+20, 60),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        color='red', fontsize=9)

    save_path = os.path.join(output_dir, f"gs_curve_G{G}_baseline_full.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"完整G-S曲线已保存: {save_path}")
    plt.show()
    return K_list, gars_sk


# ============================================================
# 分析二：每个 bit 位置的翻转率
# ============================================================
def analyze_bit_flip_rates(codes, labels, output_dir=OUTPUT_DIR):
    """
    分析每个 bit 位置的翻转率（稳定性）

    翻转率计算方法：
    对于每个身份 uid，取其所有样本的二值码，
    计算每个 bit 位置上"与该用户众数不同"的概率
    → 翻转率越低，该位置越稳定
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_ids = np.unique(labels)
    hash_dim = codes.shape[1]

    # 收集每个 bit 位置的翻转统计
    flip_counts = np.zeros(hash_dim)
    total_pairs = np.zeros(hash_dim)

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        user_codes = codes[idx]  # (n_samples, hash_dim)
        user_codes_01 = (user_codes > 0).astype(float)

        # 每个位置的众数（0 或 1）
        majority = (user_codes_01.mean(axis=0) >= 0.5).astype(float)

        # 每个样本与众数不同的次数
        for code in user_codes_01:
            flip_counts += (code != majority)
            total_pairs += 1

    # 每个位置的翻转率
    flip_rate = flip_counts / np.maximum(total_pairs, 1)

    print(f"\n=== Bit 翻转率统计 ===")
    print(f"总 bit 数: {hash_dim}")
    print(f"翻转率均值: {flip_rate.mean():.4f} ({flip_rate.mean()*100:.2f}%)")
    print(f"翻转率中位数: {np.median(flip_rate):.4f}")
    print(f"翻转率 < 5%  的 bit 数: {(flip_rate < 0.05).sum()} ({(flip_rate < 0.05).mean()*100:.1f}%)")
    print(f"翻转率 < 10% 的 bit 数: {(flip_rate < 0.10).sum()} ({(flip_rate < 0.10).mean()*100:.1f}%)")
    print(f"翻转率 < 20% 的 bit 数: {(flip_rate < 0.20).sum()} ({(flip_rate < 0.20).mean()*100:.1f}%)")
    print(f"翻转率 > 40% 的 bit 数: {(flip_rate > 0.40).sum()} ({(flip_rate > 0.40).mean()*100:.1f}%)")

    # 绘制翻转率分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：翻转率直方图
    axes[0].hist(flip_rate, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=flip_rate.mean(), color='red', linestyle='--',
                    label=f'均值={flip_rate.mean()*100:.1f}%')
    axes[0].axvline(x=0.05, color='green', linestyle=':', label='5% 阈值')
    axes[0].set_xlabel('翻转率')
    axes[0].set_ylabel('bit 数量')
    axes[0].set_title('各 bit 位置翻转率分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：按翻转率排序的 bit 位置（稳定性图）
    sorted_flip = np.sort(flip_rate)
    axes[1].plot(range(hash_dim), sorted_flip, 'b-', linewidth=1)
    axes[1].axhline(y=0.05, color='green', linestyle='--', label='5% 阈值')
    axes[1].axhline(y=0.10, color='orange', linestyle='--', label='10% 阈值')
    axes[1].axhline(y=flip_rate.mean(), color='red', linestyle='--',
                    label=f'均值={flip_rate.mean()*100:.1f}%')
    n_stable = (flip_rate < 0.10).sum()
    axes[1].axvline(x=n_stable, color='orange', linestyle=':',
                    label=f'前{n_stable}个bit翻转率<10%')
    axes[1].set_xlabel('bit 位置（按翻转率排序）')
    axes[1].set_ylabel('翻转率')
    axes[1].set_title('bit 稳定性排序（改进CTM的依据）')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "bit_flip_rate_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"翻转率分析图已保存: {save_path}")
    plt.show()

    return flip_rate


# ============================================================
# 分析三：稳定比特 vs 随机比特 的 Genuine 汉明距离对比
# ============================================================
def compare_stable_vs_random(codes, labels, flip_rate, G=512, stable_ratio=0.8, output_dir=OUTPUT_DIR):
    """
    对比两种比特选择策略下的 Genuine 汉明距离分布：
    1. 随机选 G 个 bit（论文基线 CTM）
    2. 从稳定池（翻转率最低的 stable_ratio 比例）中随机选 G 个 bit（StableCTM，保留可撤销性）

    目的：展示稳定比特选择能降低 Genuine 翻转数，同时保持可撤销性
    """
    os.makedirs(output_dir, exist_ok=True)

    hash_dim = len(flip_rate)
    rng = np.random.default_rng(42)

    # 策略1：从全部 hash_dim 个 bit 中随机选 G 个（普通 CTM）
    random_indices = rng.choice(hash_dim, size=G, replace=False)
    random_indices = np.sort(random_indices)

    # 策略2：从稳定池（翻转率最低的 stable_ratio 比例）中随机选 G 个（StableCTM，保留可撤销性）
    n_stable = max(int(hash_dim * stable_ratio), G + 1)
    stable_pool = np.argsort(flip_rate)[:n_stable]
    stable_indices = rng.choice(stable_pool, size=G, replace=False)
    stable_indices = np.sort(stable_indices)

    print(f"\n=== 比特选择策略对比 (G={G}, stable_ratio={stable_ratio}) ===")
    print(f"随机选择（CTM）:    稳定池大小={hash_dim}, 平均翻转率={flip_rate[random_indices].mean()*100:.2f}%")
    print(f"稳定位选择（StableCTM）: 稳定池大小={n_stable}, 平均翻转率={flip_rate[stable_indices].mean()*100:.2f}%")

    # 计算两种策略下的 Genuine 汉明距离
    def compute_genuine_dists(indices):
        dists = []
        unique_ids = np.unique(labels)
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            ref = codes[idx[0]][indices]
            for i in idx[1:]:
                probe = codes[i][indices]
                d = np.sum((ref > 0) != (probe > 0)) / len(indices)
                dists.append(d)
        return np.array(dists)

    dists_random = compute_genuine_dists(random_indices)
    dists_stable = compute_genuine_dists(stable_indices)

    print(f"\nGenuine 汉明距离（归一化）:")
    print(f"  随机选择（CTM）:         均值={dists_random.mean():.4f} ({dists_random.mean()*100:.1f}%), "
          f"std={dists_random.std():.4f}")
    print(f"  稳定位选择（StableCTM）: 均值={dists_stable.mean():.4f} ({dists_stable.mean()*100:.1f}%), "
          f"std={dists_stable.std():.4f}")
    print(f"  改善幅度: {(dists_random.mean() - dists_stable.mean())*100:.1f}% 翻转率降低（保留可撤销性）")

    # 绘制对比图
    from scipy.stats import norm as scipy_norm
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 0.6, 300)

    g_mean, g_std = dists_random.mean(), dists_random.std()
    s_mean, s_std = dists_stable.mean(), dists_stable.std()

    ax.plot(x, scipy_norm.pdf(x, g_mean, g_std), 'r-', linewidth=2,
            label=f'CTM 随机选位（基线）: 均值={g_mean*100:.1f}%')
    ax.plot(x, scipy_norm.pdf(x, s_mean, s_std), 'b-', linewidth=2,
            label=f'StableCTM 稳定位选择（保留可撤销性）: 均值={s_mean*100:.1f}%')
    ax.fill_between(x, scipy_norm.pdf(x, g_mean, g_std), alpha=0.15, color='red')
    ax.fill_between(x, scipy_norm.pdf(x, s_mean, s_std), alpha=0.15, color='blue')

    ax.set_xlabel('Genuine 归一化汉明距离（翻转率）')
    ax.set_ylabel('概率密度')
    ax.set_title(f'比特选择策略对比 (G={G}) — Genuine 翻转率分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"compare_selection_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {save_path}")
    plt.show()

    return dists_random, dists_stable, stable_indices


# ============================================================
# 分析四：基线 vs 改进 G-S 曲线对比（核心对比实验）
# ============================================================
def compare_gs_curves(test_codes, test_labels, flip_rate, G=512, output_dir=OUTPUT_DIR):
    """
    对比基线（随机比特选择）和改进（稳定比特选择）的完整 G-S 曲线

    K 范围需要覆盖两条曲线的下降区间：
      - 基线拐点约在 K=39 (k≈312 bits)
      - 改进拐点约在 K=55+ (k≈440+ bits)
    所以 K 范围取 7~63，步长 2
    """
    os.makedirs(output_dir, exist_ok=True)
    N = G // 8  # 64 symbols for G=512
    K_list = list(range(7, N, 2))  # step=2, covers K=7..63

    # --- 基线 CTM（随机选择）---
    ctm_baseline = CTM(hash_dim=1024, G=G)
    gars_baseline = []
    print(f"\nComputing baseline G-S curve (G={G}, K from 7 to {K_list[-1]})...")
    for K in K_list:
        sstm = SSTM(G=G, K=K)
        unique_ids = np.unique(test_labels)
        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(test_labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm_baseline.enroll(test_codes[idx[0]])
            stored_hash, _ = sstm.enroll(re)
            for i in idx[1:]:
                rp = ctm_baseline.authenticate(test_codes[i], ke)
                is_genuine, _ = sstm.authenticate(rp, stored_hash)
                pass_count += int(is_genuine)
                total += 1
        gar = pass_count / total * 100 if total > 0 else 0
        gars_baseline.append(gar)
        print(f"  Baseline  K={K:3d} (k={K*8:4d} bits): GAR={gar:.1f}%")

    # --- 改进 CTM（稳定比特选择）---
    ctm_stable = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=0.3)
    gars_stable = []
    print(f"\nComputing improved G-S curve (Stable-Bit CTM)...")
    for K in K_list:
        sstm = SSTM(G=G, K=K)
        unique_ids = np.unique(test_labels)
        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(test_labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm_stable.enroll(test_codes[idx[0]])
            stored_hash, _ = sstm.enroll(re)
            for i in idx[1:]:
                rp = ctm_stable.authenticate(test_codes[i], ke)
                is_genuine, _ = sstm.authenticate(rp, stored_hash)
                pass_count += int(is_genuine)
                total += 1
        gar = pass_count / total * 100 if total > 0 else 0
        gars_stable.append(gar)
        print(f"  Improved  K={K:3d} (k={K*8:4d} bits): GAR={gar:.1f}%")

    # --- 绘制对比图 ---
    k_bits_list = [K * 8 for K in K_list]
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(k_bits_list, gars_baseline, 'r-o', linewidth=2, markersize=4,
            label='Baseline (Random Bit Selection)')
    ax.plot(k_bits_list, gars_stable, 'b-s', linewidth=2, markersize=4,
            label='Improved (Stable Bit Selection)')

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S Curve Comparison: Baseline vs. Improved CTM (G={G} bits)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    # 标注两条曲线的拐点
    for label, gars, color in [('Baseline', gars_baseline, 'red'),
                                ('Improved', gars_stable, 'blue')]:
        for i, (k, gar) in enumerate(zip(k_bits_list, gars)):
            if gar < 50 and i > 0 and gars[i-1] >= 50:
                ax.axvline(x=k, color=color, linestyle=':', alpha=0.6)
                offset = 15 if label == 'Baseline' else -80
                ax.annotate(f'{label}\nk={k}b',
                            xy=(k, 50), xytext=(k + offset, 62),
                            arrowprops=dict(arrowstyle='->', color=color),
                            color=color, fontsize=8)
                break

    save_path = os.path.join(output_dir, f"gs_comparison_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison G-S curve saved: {save_path}")
    plt.show()
    return K_list, gars_baseline, gars_stable


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型和数据
    model_path = "checkpoints/final_model.pth"
    model, train_loader, test_loader, num_classes, device = load_model_and_data(model_path)

    # 提取训练集和测试集的二值码
    print("\nExtracting training set binary codes (for flip rate analysis)...")
    train_codes, train_labels = extract_codes(model, train_loader, device)
    print(f"Training set: {train_codes.shape}")

    print("\nExtracting test set binary codes (for evaluation)...")
    test_codes, test_labels = extract_codes(model, test_loader, device)
    print(f"Test set: {test_codes.shape}")

    # 分析一：bit 翻转率分析（为改进提供依据）
    print("\n" + "="*50)
    print("Analysis 1: Bit Flip Rate Analysis")
    flip_rate = analyze_bit_flip_rates(train_codes, train_labels)

    # 分析二：稳定比特 vs 随机比特 Genuine 汉明距离对比
    print("\n" + "="*50)
    print("Analysis 2: Stable Bit vs. Random Bit Selection")
    dists_random, dists_stable, stable_indices = compare_stable_vs_random(
        test_codes, test_labels, flip_rate, G=512
    )

    # 分析三：核心对比实验 — 完整 G-S 曲线（基线 vs 改进）
    print("\n" + "="*50)
    print("Analysis 3: G-S Curve Comparison (Baseline vs. Improved CTM)")
    K_list, gars_baseline, gars_stable = compare_gs_curves(
        test_codes, test_labels, flip_rate, G=512
    )

    print("\n" + "="*50)
    print("Analysis complete! Results saved to results/ directory")
    print("Generated figures:")
    print("  - bit_flip_rate_analysis.png   Bit flip rate analysis")
    print("  - compare_selection_G512.png   Genuine flip rate distribution comparison")
    print("  - gs_comparison_G512.png       G-S curve: Baseline vs. Improved (KEY FIGURE)")
