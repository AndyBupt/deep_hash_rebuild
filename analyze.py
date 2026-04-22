"""
Analysis Script:
1. Plot full G-S curve (K from small to large, showing GAR dropping from 100%)
2. Analyze per-bit flip rates to identify stable vs. unstable bits
3. Provide data support for CTM improvement
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import build_dataloaders, get_transforms
from model import FingerprintHashNet
from ctm import CTM, StableCTM
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
    print(f"Device: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8
    )
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024, pretrained=False)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    model = model.to(device)
    model.set_beta(32)
    return model, train_loader, test_loader, num_classes, device


def extract_codes(model, loader, device):
    """Extract binary hash codes for all samples."""
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
# Analysis 1: Full G-S Curve
# ============================================================
def plot_full_gs_curve(codes, labels, G=512, output_dir=OUTPUT_DIR):
    """
    Plot full G-S curve: K from 7 to N-1, showing GAR dropping from 100% to 0%.
    Key figure showing the limitation of the baseline system.
    """
    os.makedirs(output_dir, exist_ok=True)
    N = G // 8
    ctm = CTM(hash_dim=1024, G=G)

    K_list = list(range(7, N, 2))  # step=2
    gars_sk = []
    print(f"\nComputing full G-S curve (G={G}, N={N}, K from 7 to {K_list[-1]})...")

    for K in K_list:
        sstm = SSTM(G=G, K=K)
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
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'Full G-S Curve (G={G} bits) - Baseline CTM (Random Bit Selection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    for i, (k, gar) in enumerate(zip(k_bits_list, gars_sk)):
        if gar < 50 and i > 0 and gars_sk[i-1] >= 50:
            ax.axvline(x=k, color='red', linestyle=':', alpha=0.7)
            ax.annotate(f'k={k} bits\nGAR<50%',
                        xy=(k, 50), xytext=(k+20, 60),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        color='red', fontsize=9)

    save_path = os.path.join(output_dir, f"gs_curve_G{G}_baseline_full.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Full G-S curve saved: {save_path}")
    plt.close()
    return K_list, gars_sk


# ============================================================
# Analysis 2: Per-bit Flip Rate
# ============================================================
def analyze_bit_flip_rates(codes, labels, output_dir=OUTPUT_DIR):
    """
    Analyze per-bit flip rate (stability).

    Flip rate calculation:
    For each identity uid, take all sample binary codes,
    compute the probability that each bit position differs from the majority vote.
    Lower flip rate = more stable bit position.
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_ids = np.unique(labels)
    hash_dim = codes.shape[1]

    flip_counts = np.zeros(hash_dim)
    total_pairs = np.zeros(hash_dim)

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        user_codes = codes[idx]
        user_codes_01 = (user_codes > 0).astype(float)

        majority = (user_codes_01.mean(axis=0) >= 0.5).astype(float)

        for code in user_codes_01:
            flip_counts += (code != majority)
            total_pairs += 1

    flip_rate = flip_counts / np.maximum(total_pairs, 1)

    print(f"\n=== Bit Flip Rate Statistics ===")
    print(f"Total bits: {hash_dim}")
    print(f"Mean flip rate: {flip_rate.mean():.4f} ({flip_rate.mean()*100:.2f}%)")
    print(f"Median flip rate: {np.median(flip_rate):.4f}")
    print(f"Bits with flip rate < 5%:  {(flip_rate < 0.05).sum()} ({(flip_rate < 0.05).mean()*100:.1f}%)")
    print(f"Bits with flip rate < 10%: {(flip_rate < 0.10).sum()} ({(flip_rate < 0.10).mean()*100:.1f}%)")
    print(f"Bits with flip rate < 20%: {(flip_rate < 0.20).sum()} ({(flip_rate < 0.20).mean()*100:.1f}%)")
    print(f"Bits with flip rate > 40%: {(flip_rate > 0.40).sum()} ({(flip_rate > 0.40).mean()*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(flip_rate, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=flip_rate.mean(), color='red', linestyle='--',
                    label=f'Mean={flip_rate.mean()*100:.1f}%')
    axes[0].axvline(x=0.05, color='green', linestyle=':', label='5% threshold')
    axes[0].set_xlabel('Flip Rate')
    axes[0].set_ylabel('Number of Bits')
    axes[0].set_title('Per-bit Flip Rate Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sorted_flip = np.sort(flip_rate)
    axes[1].plot(range(hash_dim), sorted_flip, 'b-', linewidth=1)
    axes[1].axhline(y=0.05, color='green', linestyle='--', label='5% threshold')
    axes[1].axhline(y=0.10, color='orange', linestyle='--', label='10% threshold')
    axes[1].axhline(y=flip_rate.mean(), color='red', linestyle='--',
                    label=f'Mean={flip_rate.mean()*100:.1f}%')
    n_stable = (flip_rate < 0.10).sum()
    axes[1].axvline(x=n_stable, color='orange', linestyle=':',
                    label=f'Top {n_stable} bits with flip rate < 10%')
    axes[1].set_xlabel('Bit Position (sorted by flip rate)')
    axes[1].set_ylabel('Flip Rate')
    axes[1].set_title('Bit Stability Ranking (basis for StableCTM)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "bit_flip_rate_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Flip rate analysis saved: {save_path}")
    plt.close()

    return flip_rate


# ============================================================
# Analysis 3: Stable vs. Random Bit Selection - Genuine Distance
# ============================================================
def compare_stable_vs_random(codes, labels, flip_rate, G=512, stable_ratio=0.8, output_dir=OUTPUT_DIR):
    """
    Compare Genuine Hamming distance distributions under two bit selection strategies:
    1. Random selection from all bits (baseline CTM)
    2. Random selection from stable pool (StableCTM, preserving cancelability)

    Goal: show that stable bit selection reduces Genuine flip rate while maintaining cancelability.
    """
    os.makedirs(output_dir, exist_ok=True)

    hash_dim = len(flip_rate)
    rng = np.random.default_rng(42)

    # Strategy 1: random selection from all hash_dim bits (CTM)
    random_indices = rng.choice(hash_dim, size=G, replace=False)
    random_indices = np.sort(random_indices)

    # Strategy 2: random selection from stable pool (StableCTM, preserving cancelability)
    n_stable = max(int(hash_dim * stable_ratio), G + 1)
    stable_pool = np.argsort(flip_rate)[:n_stable]
    stable_indices = rng.choice(stable_pool, size=G, replace=False)
    stable_indices = np.sort(stable_indices)

    print(f"\n=== Bit Selection Strategy Comparison (G={G}, stable_ratio={stable_ratio}) ===")
    print(f"CTM (random):    pool size={hash_dim}, mean flip rate={flip_rate[random_indices].mean()*100:.2f}%")
    print(f"StableCTM:       pool size={n_stable}, mean flip rate={flip_rate[stable_indices].mean()*100:.2f}%")

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

    print(f"\nGenuine Hamming Distance (normalized):")
    print(f"  CTM (random):  mean={dists_random.mean():.4f} ({dists_random.mean()*100:.1f}%), "
          f"std={dists_random.std():.4f}")
    print(f"  StableCTM:     mean={dists_stable.mean():.4f} ({dists_stable.mean()*100:.1f}%), "
          f"std={dists_stable.std():.4f}")
    print(f"  Improvement: {(dists_random.mean() - dists_stable.mean())*100:.1f}% flip rate reduction (cancelability preserved)")

    from scipy.stats import norm as scipy_norm
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 0.6, 300)

    g_mean, g_std = dists_random.mean(), dists_random.std()
    s_mean, s_std = dists_stable.mean(), dists_stable.std()

    ax.plot(x, scipy_norm.pdf(x, g_mean, g_std), 'r-', linewidth=2,
            label=f'CTM - Random Selection (Baseline): mean={g_mean*100:.1f}%')
    ax.plot(x, scipy_norm.pdf(x, s_mean, s_std), 'b-', linewidth=2,
            label=f'StableCTM - Stable Pool Selection: mean={s_mean*100:.1f}%')
    ax.fill_between(x, scipy_norm.pdf(x, g_mean, g_std), alpha=0.15, color='red')
    ax.fill_between(x, scipy_norm.pdf(x, s_mean, s_std), alpha=0.15, color='blue')

    ax.set_xlabel('Normalized Genuine Hamming Distance (Flip Rate)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Bit Selection Strategy Comparison (G={G}) - Genuine Flip Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"compare_selection_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved: {save_path}")
    plt.close()

    return dists_random, dists_stable, stable_indices


# ============================================================
# Analysis 4: Baseline vs. Improved G-S Curve Comparison
# ============================================================
def compare_gs_curves(test_codes, test_labels, flip_rate, G=512, output_dir=OUTPUT_DIR):
    """
    Compare full G-S curves: baseline (random bit selection) vs. improved (stable bit selection).
    """
    os.makedirs(output_dir, exist_ok=True)
    N = G // 8
    K_list = list(range(7, N, 2))

    # --- Baseline CTM (random selection) ---
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

    # --- Improved CTM (stable bit selection) ---
    ctm_stable = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=0.8)
    gars_stable = []
    print(f"\nComputing improved G-S curve (StableCTM)...")
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

    # --- Plot ---
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
    plt.close()
    return K_list, gars_baseline, gars_stable


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = "checkpoints/final_model.pth"
    model, train_loader, test_loader, num_classes, device = load_model_and_data(model_path)

    print("\nExtracting training set binary codes (for flip rate analysis)...")
    train_codes, train_labels = extract_codes(model, train_loader, device)
    print(f"Training set: {train_codes.shape}")

    print("\nExtracting test set binary codes (for evaluation)...")
    test_codes, test_labels = extract_codes(model, test_loader, device)
    print(f"Test set: {test_codes.shape}")

    print("\n" + "="*50)
    print("Analysis 1: Bit Flip Rate Analysis")
    flip_rate = analyze_bit_flip_rates(train_codes, train_labels)

    print("\n" + "="*50)
    print("Analysis 2: Stable Bit vs. Random Bit Selection")
    dists_random, dists_stable, stable_indices = compare_stable_vs_random(
        test_codes, test_labels, flip_rate, G=512
    )

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
