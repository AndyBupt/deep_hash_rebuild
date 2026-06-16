"""
evaluate_unlinkability.py — Formal Unlinkability & Multiple-Revocation Evaluation

Extends evaluate_cancelability.py with TBIOM-grade unlinkability metrics:

1. Multiple Revocation
   Each user is assigned N_KEYS independent keys (simulating N_KEYS revocations).
   All cross-key template pairs for the same user should be indistinguishable from
   cross-user pairs → proves repeated revocation is safe.

2. Formal Unlinkability Metrics (Gomez-Barrero et al.)
   - Mated scores:     same user, different keys  → should look random (~50%)
   - Non-mated scores: different users, diff keys → random (~50%)
   If the two distributions overlap, the system is unlinkable.

   Reported metrics:
     a) Linkability EER — if EER ≈ 50%, attacker cannot link templates
     b) Dsys            — area between mated / non-mated CDFs; 0 = perfect unlinkability
     c) Distribution plot with local unlinkability D(s) curve

Output: results_unlinkability/
  revocation_distances.png         — distance distributions for all revocation rounds
  mated_vs_nonmated.png            — formal mated / non-mated score distributions
  unlinkability_summary.json       — numerical results

Usage:
  python evaluate_unlinkability.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve

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
OUTPUT_DIR   = "results_unlinkability"
G            = 512
STABLE_RATIO = 0.8
N_KEYS       = 5        # revocations per user
N_NON_MATED  = 3000     # non-mated pairs to sample


# ──────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────

def extract_codes(model, loader, device):
    model.eval()
    all_binary, all_labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, _, binary_c = model(imgs.to(device))
            all_binary.append(binary_c.cpu().numpy())
            all_labels.append(lbs.numpy())
    return np.vstack(all_binary), np.concatenate(all_labels)


# ──────────────────────────────────────────────
# Hamming distance helper
# ──────────────────────────────────────────────

def hamming_dist(re1, re2):
    b1 = (re1 > 0).astype(np.uint8)
    b2 = (re2 > 0).astype(np.uint8)
    return np.sum(b1 != b2) / len(b1)


# ──────────────────────────────────────────────
# Formal unlinkability metrics
# ──────────────────────────────────────────────

def compute_linkability_eer(mated_dists, non_mated_dists):
    """
    Linkability EER: how well can an attacker classify 'same user, diff key' (mated)
    vs 'diff user, diff key' (non-mated) pairs?

    EER = 50% → attacker cannot distinguish → perfect unlinkability
    EER = 0%  → perfectly linkable

    Uses distance as feature (lower dist → more similar → more likely mated).
    We negate distances so higher score = more similar.
    """
    scores = np.concatenate([-mated_dists, -non_mated_dists])
    labels = np.concatenate([np.ones(len(mated_dists)), np.zeros(len(non_mated_dists))])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    return eer


def compute_dsys(mated_dists, non_mated_dists, n_bins=200):
    """
    Dsys: global unlinkability measure (simplified from Gomez-Barrero et al.).

    Dsys = (1/2) * ∫ |F_mated(s) - F_non_mated(s)| ds  (area between CDFs)

    Range: [0, 1]
      0 → perfect unlinkability (identical distributions)
      1 → perfect linkability (fully separated)

    Also returns the local D(s) curve for plotting.
    """
    all_dists = np.concatenate([mated_dists, non_mated_dists])
    s_min, s_max = all_dists.min(), all_dists.max()
    s_vals = np.linspace(s_min, s_max, n_bins)

    f_mated     = np.array([np.mean(mated_dists     <= s) for s in s_vals])
    f_non_mated = np.array([np.mean(non_mated_dists  <= s) for s in s_vals])

    # Local linkability D(s): probability that a pair with score s is mated minus chance
    # D(s) = 2 * max(0, f_mated(s) / (f_mated(s) + f_non_mated(s)) - 0.5)
    eps = 1e-12
    denominator = f_mated + f_non_mated + eps
    d_local = 2.0 * np.maximum(0, f_mated / denominator - 0.5)

    # Global Dsys = area between CDFs normalized
    ds = s_vals[1] - s_vals[0]
    dsys = float(0.5 * np.sum(np.abs(f_mated - f_non_mated)) * ds / (s_max - s_min))

    return dsys, s_vals, d_local, f_mated, f_non_mated


# ──────────────────────────────────────────────
# Multiple revocation experiment
# ──────────────────────────────────────────────

def compute_revocation_distances(binary_codes, labels, ctm, n_keys, rng):
    """
    For each user, generate n_keys independent keys and compute:
      - Cross-key distances (same user, different keys) for all key pairs
      - This simulates n_keys successive revocations and checks whether
        past templates can be linked to new ones.

    Returns arrays of distances split by revocation round.
    """
    unique_ids = np.unique(labels)
    print(f"  Users: {len(unique_ids)}, Keys per user: {n_keys}")

    # cross_key_dists[j] = distances between templates made with key j and key j+1
    # We store all cross-key pairs (not just consecutive)
    all_cross_key = []          # all same-user, different-key distances
    round_dists   = [[] for _ in range(n_keys - 1)]  # consecutive round pairs

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue

        # Pick one enrollment sample per user
        enroll_idx = idx[0]

        # Generate n_keys independent templates
        templates = []
        for k in range(n_keys):
            re, _ = ctm.enroll(binary_codes[enroll_idx])
            templates.append(re)

        # All unique key pairs
        for i in range(n_keys):
            for j in range(i + 1, n_keys):
                d = hamming_dist(templates[i], templates[j])
                all_cross_key.append(d)
                # Store consecutive round (round index = i)
                if j == i + 1:
                    round_dists[i].append(d)

    return np.array(all_cross_key), [np.array(rd) for rd in round_dists]


def compute_non_mated_diff_key(binary_codes, labels, ctm, n_pairs, rng):
    """
    Non-mated distances: different users, each with their own independent key.
    Used as reference for unlinkability.
    """
    unique_ids = np.unique(labels)
    dists = []
    for _ in range(n_pairs):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        re1, _ = ctm.enroll(binary_codes[idx1])
        re2, _ = ctm.enroll(binary_codes[idx2])
        dists.append(hamming_dist(re1, re2))
    return np.array(dists)


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_revocation_distributions(cross_key_dists, round_dists, non_mated_dists,
                                   output_dir):
    """Plot distance distributions for each revocation round + non-mated reference."""
    x = np.linspace(0, 0.8, 500)
    fig, ax = plt.subplots(figsize=(11, 6))

    # Non-mated reference
    kde_nm = gaussian_kde(non_mated_dists, bw_method=0.08)
    ax.plot(x, kde_nm(x), 'k--', linewidth=2,
            label=f'Non-mated (diff user, diff key)  μ={non_mated_dists.mean()*100:.1f}%')
    ax.fill_between(x, kde_nm(x), alpha=0.06, color='black')

    # Each revocation round
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(round_dists)))
    for i, (rd, color) in enumerate(zip(round_dists, colors)):
        if len(rd) == 0:
            continue
        kde = gaussian_kde(rd, bw_method=0.08)
        ax.plot(x, kde(x), color=color, linewidth=1.8,
                label=f'Revocation round {i}→{i+1}  μ={rd.mean()*100:.1f}%')

    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='50% (random)')
    ax.set_xlabel('Normalised Hamming Distance')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Multiple Revocation: Cross-Key Distance Distributions\n'
                 f'({len(round_dists) + 1} keys / user, G={G}, FVC2004)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.8)
    save_path = os.path.join(output_dir, "revocation_distances.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_mated_vs_nonmated(mated_dists, non_mated_dists, s_vals, d_local,
                            f_mated, f_non_mated, dsys, link_eer, output_dir):
    """Formal unlinkability plot: mated vs non-mated + Dsys curve."""
    x = np.linspace(0, 0.8, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PDF distributions
    ax = axes[0]
    kde_m  = gaussian_kde(mated_dists,     bw_method=0.08)
    kde_nm = gaussian_kde(non_mated_dists, bw_method=0.08)
    ax.plot(x, kde_m(x),  'b-',  linewidth=2,
            label=f'Mated (same user, diff key)  μ={mated_dists.mean()*100:.1f}%')
    ax.plot(x, kde_nm(x), 'r--', linewidth=2,
            label=f'Non-mated (diff user, diff key)  μ={non_mated_dists.mean()*100:.1f}%')
    ax.fill_between(x, kde_m(x),  alpha=0.12, color='blue')
    ax.fill_between(x, kde_nm(x), alpha=0.12, color='red')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Normalised Hamming Distance')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Formal Unlinkability: Mated vs Non-mated\n'
                 f'Linkability EER = {link_eer*100:.1f}%  |  Dsys = {dsys:.4f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.8)

    # Right: Local unlinkability D(s) and CDFs
    ax2 = axes[1]
    ax2.plot(s_vals, f_mated,     'b-',  linewidth=1.5, label='CDF mated')
    ax2.plot(s_vals, f_non_mated, 'r--', linewidth=1.5, label='CDF non-mated')
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(s_vals, d_local, alpha=0.25, color='green')
    ax2_twin.plot(s_vals, d_local, 'g-', linewidth=1.5, label='Local D(s)')
    ax2_twin.set_ylabel('Local linkability D(s)', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2_twin.set_ylim(0, 1)
    ax2.set_xlabel('Normalised Hamming Distance')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title(f'CDFs and Local Unlinkability Curve\n'
                  f'Dsys = {dsys:.4f}  (0 = perfect unlinkability)')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "mated_vs_nonmated.png")
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
    else:
        print("WARNING: using random model weights")
    model = model.to(device)
    model.set_beta(32)

    print("\nExtracting training codes...")
    train_binary, train_labels = extract_codes(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    print(f"Training flip rate mean: {flip_rate.mean()*100:.2f}%")

    print("Extracting test codes...")
    test_binary, test_labels = extract_codes(model, test_loader, device)
    unique_ids = np.unique(test_labels)
    print(f"Test set: {test_binary.shape}, users: {len(unique_ids)}")

    ctm = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    rng = np.random.default_rng(42)

    # ── 1. Multiple Revocation ─────────────────
    print(f"\n{'='*60}")
    print(f"Multiple Revocation Experiment  ({N_KEYS} keys/user)")
    print("="*60)
    cross_key_dists, round_dists = compute_revocation_distances(
        test_binary, test_labels, ctm, N_KEYS, rng
    )
    print(f"  All cross-key pairs: n={len(cross_key_dists)}, "
          f"mean={cross_key_dists.mean()*100:.2f}%, std={cross_key_dists.std()*100:.2f}%")
    for i, rd in enumerate(round_dists):
        print(f"  Round {i}→{i+1}: n={len(rd)}, "
              f"mean={rd.mean()*100:.2f}%, std={rd.std()*100:.2f}%")

    # Non-mated reference
    print("\n  Computing non-mated distances...")
    non_mated_dists = compute_non_mated_diff_key(
        test_binary, test_labels, ctm, N_NON_MATED, rng
    )
    print(f"  Non-mated: n={len(non_mated_dists)}, "
          f"mean={non_mated_dists.mean()*100:.2f}%, std={non_mated_dists.std()*100:.2f}%")

    plot_revocation_distributions(cross_key_dists, round_dists, non_mated_dists, OUTPUT_DIR)

    # ── 2. Formal Unlinkability Metrics ────────
    print(f"\n{'='*60}")
    print("Formal Unlinkability Metrics")
    print("="*60)

    # Mated = all cross-key pairs (same user, different keys)
    mated_dists = cross_key_dists

    link_eer = compute_linkability_eer(mated_dists, non_mated_dists)
    dsys, s_vals, d_local, f_mated, f_non_mated = compute_dsys(
        mated_dists, non_mated_dists
    )

    print(f"\n  Linkability EER     = {link_eer*100:.2f}%  "
          f"(50% = perfect unlinkability, 0% = fully linkable)")
    print(f"  Dsys                = {dsys:.4f}  "
          f"(0 = perfect unlinkability, 1 = fully linkable)")
    print(f"  Mated  dist: mean={mated_dists.mean()*100:.2f}%  std={mated_dists.std()*100:.2f}%")
    print(f"  Non-mated dist: mean={non_mated_dists.mean()*100:.2f}%  std={non_mated_dists.std()*100:.2f}%")

    plot_mated_vs_nonmated(
        mated_dists, non_mated_dists, s_vals, d_local,
        f_mated, f_non_mated, dsys, link_eer, OUTPUT_DIR
    )

    # ── Save JSON ──────────────────────────────
    results = {
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G":              G,
        "n_keys_per_user": N_KEYS,
        "multiple_revocation": {
            "all_cross_key_mean_%":  round(float(cross_key_dists.mean() * 100), 3),
            "all_cross_key_std_%":   round(float(cross_key_dists.std()  * 100), 3),
            "n_pairs":               int(len(cross_key_dists)),
            "per_round": [
                {"round": f"{i}->{i+1}",
                 "mean_%": round(float(rd.mean() * 100), 3),
                 "std_%":  round(float(rd.std()  * 100), 3),
                 "n":      int(len(rd))}
                for i, rd in enumerate(round_dists)
            ],
        },
        "formal_unlinkability": {
            "linkability_EER_%":   round(link_eer * 100, 3),
            "Dsys":                round(dsys, 6),
            "mated_mean_%":        round(float(mated_dists.mean()     * 100), 3),
            "mated_std_%":         round(float(mated_dists.std()      * 100), 3),
            "non_mated_mean_%":    round(float(non_mated_dists.mean() * 100), 3),
            "non_mated_std_%":     round(float(non_mated_dists.std()  * 100), 3),
            "interpretation": (
                "EER=50% → perfect unlinkability (random). "
                "Dsys=0 → distributions identical → perfectly unlinkable."
            ),
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "unlinkability_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
