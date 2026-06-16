"""
evaluate_entropy.py — Entropy Analysis of RGSS-Selected Reliable Bits

Addresses the TBIOM reviewer question:
  "RGSS selects high-confidence bits — do these bits have lower entropy?
   If so, an attacker could predict/guess them."

Key claim to prove:
  "reliable does not necessarily mean predictable"
  → RGSS selects bits that are stable for a specific user,
    but their absolute values (0 or 1) are NOT biased at the population level.

Analysis:
  1. For each user, record which bits RGSS selects as reliable (top-k by |tanh|).
  2. For each position, compute p = P(bit=1) across the user population.
  3. Shannon entropy H_i = -p_i*log2(p_i) - (1-p_i)*log2(1-p_i)
  4. Min-entropy = -log2(max(p_i, 1-p_i))
  5. Compare RGSS-selected bits vs randomly-selected bits.
  6. Report: average entropy, fraction of bits with H > 0.9 (near-uniform).
  7. Also measure pairwise correlation among selected bits.

Output: results_entropy/
  entropy_comparison.png    — entropy histogram: RGSS vs random
  bit_bias.png              — p(bit=1) distribution for selected bits
  entropy_analysis.json     — numerical results

Usage:
  python evaluate_entropy.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import StableCTM


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_entropy"
G            = 512
STABLE_RATIO = 0.8
K_BITS       = 264      # RGSS key length (k₅₀ inflection)
N_SEEDS      = 5        # random seeds for CTM to average over


# ──────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────

def extract_codes_with_embed(model, loader, device):
    model.eval()
    all_binary, all_hash, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, hash_c, binary_c = model(imgs.to(device))
            all_binary.append(binary_c.cpu().numpy())
            all_hash.append(hash_c.cpu().numpy())
            all_labels.append(lbs.numpy())
    return (np.vstack(all_binary),
            np.vstack(all_hash),
            np.concatenate(all_labels))


# ──────────────────────────────────────────────
# Entropy helpers
# ──────────────────────────────────────────────

def shannon_entropy(p):
    """H(p) = -p*log2(p) - (1-p)*log2(1-p)  per bit."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def min_entropy(p):
    """H_inf = -log2(max(p, 1-p))"""
    p = np.clip(p, 0, 1)
    return -np.log2(np.maximum(p, 1 - p))


def compute_entropy_stats(bit_values):
    """
    bit_values: (N_users, k_bits) array of {0,1} values
    Returns per-bit p(bit=1), Shannon entropy, min-entropy.
    """
    p = bit_values.mean(axis=0)          # (k_bits,) fraction of 1s per bit
    h_shannon = shannon_entropy(p)        # (k_bits,)
    h_min     = min_entropy(p)            # (k_bits,)
    return p, h_shannon, h_min


def get_rgss_selected_bit_values(binary_codes, hash_codes, labels, ctm, k_bits):
    """
    For each user (one enrollment sample), determine RGSS-selected positions
    via argsort(-|tanh|), then record the bit values (0 or 1) at those positions.

    Returns: (N_users, k_bits) array of bit values.
    """
    unique_ids = np.unique(labels)
    bit_value_matrix = []

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]

        # Step 1: CTM selects G positions (ke)
        _, ke = ctm.enroll(binary_codes[enroll_idx])

        # Step 2: RGSS selects top-k by |tanh| within those G positions
        embed = hash_codes[enroll_idx][ke]   # (G,) tanh values
        perm  = np.argsort(-np.abs(embed))   # sorted by reliability
        reliable_ke = ke[perm[:k_bits]]      # top-k indices in original 1024-dim space

        # Step 3: Record bit values (0 or 1)
        bits = (binary_codes[enroll_idx][reliable_ke] > 0).astype(np.uint8)
        bit_value_matrix.append(bits)

    return np.vstack(bit_value_matrix)   # (N_users, k_bits)


def get_random_selected_bit_values(binary_codes, labels, ctm, k_bits, rng):
    """
    For each user, randomly select k_bits positions (ignoring reliability).
    Used as control group.
    """
    unique_ids = np.unique(labels)
    bit_value_matrix = []

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]

        _, ke = ctm.enroll(binary_codes[enroll_idx])
        random_positions = rng.choice(ke, size=k_bits, replace=False)
        bits = (binary_codes[enroll_idx][random_positions] > 0).astype(np.uint8)
        bit_value_matrix.append(bits)

    return np.vstack(bit_value_matrix)


def compute_pairwise_correlation(bit_values):
    """
    Average absolute pairwise correlation among selected bits.
    Lower = more independent = more entropy preserved.
    """
    corr_matrix = np.corrcoef(bit_values.T)   # (k_bits, k_bits)
    # Average upper triangle (excluding diagonal)
    n = corr_matrix.shape[0]
    upper = corr_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(np.abs(upper)))


def compute_effective_entropy(bit_values):
    """
    Effective entropy accounting for correlations:
    H_eff = H_bits - MI_penalty (approximate using correlation).
    Here we use a simple approximation: sum of individual entropies
    as an upper bound, and note correlation as a qualifier.
    """
    p = bit_values.mean(axis=0)
    h = shannon_entropy(p)
    return float(h.sum())   # bits of entropy (upper bound)


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_entropy_comparison(p_rgss, h_rgss, p_random, h_random, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of p(bit=1)
    ax = axes[0]
    bins = np.linspace(0, 1, 31)
    ax.hist(p_rgss,   bins=bins, alpha=0.7, color='#1f77b4', label='RGSS selected bits')
    ax.hist(p_random, bins=bins, alpha=0.7, color='#d62728', label='Random selected bits')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='p=0.5 (unbiased)')
    ax.set_xlabel('P(bit = 1) across users')
    ax.set_ylabel('Number of bits')
    ax.set_title('Bit Bias: RGSS-selected vs Random-selected')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Histogram of Shannon entropy
    ax2 = axes[1]
    bins_h = np.linspace(0, 1, 31)
    ax2.hist(h_rgss,   bins=bins_h, alpha=0.7, color='#1f77b4',
             label=f'RGSS: mean H={h_rgss.mean():.3f} bits')
    ax2.hist(h_random, bins=bins_h, alpha=0.7, color='#d62728',
             label=f'Random: mean H={h_random.mean():.3f} bits')
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='H=1 (uniform)')
    ax2.set_xlabel('Shannon entropy per bit (bits)')
    ax2.set_ylabel('Number of bits')
    ax2.set_title('Per-bit Shannon Entropy: RGSS vs Random\n'
                  '"reliable ≠ predictable" if RGSS entropy ≈ random')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "entropy_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8
    )

    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024, pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded: {MODEL_PATH}")
    model = model.to(device)
    model.set_beta(32)

    print("\nExtracting training codes...")
    train_binary, train_hash, train_labels = extract_codes_with_embed(
        model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)

    print("Extracting test codes (with tanh embed)...")
    test_binary, test_hash, test_labels = extract_codes_with_embed(
        model, test_loader, device)
    unique_ids = np.unique(test_labels)
    print(f"Test set: {test_binary.shape}, users: {len(unique_ids)}")

    ctm = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    rng = np.random.default_rng(42)

    print(f"\n{'='*60}")
    print(f"Entropy Analysis  (k_bits={K_BITS}, G={G})")
    print("="*60)

    # ── RGSS selected bits ─────────────────────
    print("\n  [RGSS selected bits]...")
    bv_rgss = get_rgss_selected_bit_values(
        test_binary, test_hash, test_labels, ctm, K_BITS
    )
    p_rgss, h_rgss, hmin_rgss = compute_entropy_stats(bv_rgss)
    corr_rgss  = compute_pairwise_correlation(bv_rgss)
    heff_rgss  = compute_effective_entropy(bv_rgss)

    print(f"    N_users × k_bits = {bv_rgss.shape}")
    print(f"    Mean p(bit=1)     = {p_rgss.mean():.4f}  (0.5 = unbiased)")
    print(f"    Mean Shannon H    = {h_rgss.mean():.4f} bits  (1.0 = max)")
    print(f"    Mean min-entropy  = {hmin_rgss.mean():.4f} bits")
    print(f"    Bits with H>0.9   = {(h_rgss > 0.9).sum()}/{len(h_rgss)} "
          f"({(h_rgss > 0.9).mean()*100:.1f}%)")
    print(f"    Avg pairwise corr = {corr_rgss:.4f}")
    print(f"    Sum H (upper bnd) = {heff_rgss:.1f} bits")

    # ── Random selected bits ───────────────────
    print("\n  [Random selected bits (control)]...")
    bv_random = get_random_selected_bit_values(
        test_binary, test_labels, ctm, K_BITS, rng
    )
    p_random, h_random, hmin_random = compute_entropy_stats(bv_random)
    corr_random  = compute_pairwise_correlation(bv_random)
    heff_random  = compute_effective_entropy(bv_random)

    print(f"    Mean p(bit=1)     = {p_random.mean():.4f}")
    print(f"    Mean Shannon H    = {h_random.mean():.4f} bits")
    print(f"    Mean min-entropy  = {hmin_random.mean():.4f} bits")
    print(f"    Bits with H>0.9   = {(h_random > 0.9).sum()}/{len(h_random)} "
          f"({(h_random > 0.9).mean()*100:.1f}%)")
    print(f"    Avg pairwise corr = {corr_random:.4f}")

    print("\n  [Comparison]")
    h_diff = h_rgss.mean() - h_random.mean()
    print(f"    RGSS - Random entropy difference = {h_diff:+.4f} bits")
    if abs(h_diff) < 0.05:
        verdict = "RGSS entropy ≈ random → 'reliable ≠ predictable' CONFIRMED ✓"
    elif h_diff < 0:
        verdict = "RGSS entropy < random → some bias in selected bits (investigate)"
    else:
        verdict = "RGSS entropy > random → selected bits are MORE uniform (good)"
    print(f"    Verdict: {verdict}")

    plot_entropy_comparison(p_rgss, h_rgss, p_random, h_random, OUTPUT_DIR)

    # ── Save JSON ──────────────────────────────
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G, "k_bits": K_BITS,
        "n_users": int(len(unique_ids)),
        "RGSS_selected": {
            "mean_p_bit1":        round(float(p_rgss.mean()), 4),
            "std_p_bit1":         round(float(p_rgss.std()),  4),
            "mean_shannon_H":     round(float(h_rgss.mean()),    4),
            "min_shannon_H":      round(float(h_rgss.min()),     4),
            "mean_min_entropy":   round(float(hmin_rgss.mean()), 4),
            "fraction_H_gt_0.9":  round(float((h_rgss > 0.9).mean()), 4),
            "avg_pairwise_corr":  round(corr_rgss, 4),
            "sum_H_upper_bound":  round(heff_rgss, 2),
        },
        "Random_selected": {
            "mean_p_bit1":        round(float(p_random.mean()), 4),
            "std_p_bit1":         round(float(p_random.std()),  4),
            "mean_shannon_H":     round(float(h_random.mean()),    4),
            "mean_min_entropy":   round(float(hmin_random.mean()), 4),
            "fraction_H_gt_0.9":  round(float((h_random > 0.9).mean()), 4),
            "avg_pairwise_corr":  round(corr_random, 4),
            "sum_H_upper_bound":  round(heff_random, 2),
        },
        "entropy_difference_RGSS_minus_Random": round(h_diff, 4),
        "verdict": verdict,
    }
    json_path = os.path.join(OUTPUT_DIR, "entropy_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
