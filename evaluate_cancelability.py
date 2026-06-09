"""
evaluate_cancelability.py — Cancelable Template Property Evaluation

Tests the four distance scenarios required to verify cancelability and
unlinkability of the template protection scheme:

  1. Genuine   same-key:       same finger, same key       → distance LOW  (~20%)
  2. Impostor  same-key:       diff finger, same key       → distance HIGH (~50%)
  3. Genuine   different-key:  same finger, NEW key        → distance ~50% (cancelability!)
  4. Impostor  different-key:  diff finger, diff key       → distance HIGH (~50%)

Expected result:
  - Genuine same-key forms a tight cluster at ~genuine_flip_rate
  - The other three distributions all cluster near 50% (random)
  - In particular, Genuine different-key ≈ Impostor → proves cancelability
  - Impostor different-key ≈ Impostor same-key → proves unlinkability

Output: results_cancelability/
  distance_distributions.png   — overlapping PDF of all 4 distributions
  cancelability_results.json   — numerical statistics

Usage:
  python evaluate_cancelability.py
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

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import CTM, StableCTM


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_cancelability"
G            = 512
STABLE_RATIO = 0.8
N_IMPOSTOR   = 2000   # number of impostor pairs to sample


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
    """Normalised Hamming distance between two {-1,+1} or {0,1} binary vectors."""
    b1 = (re1 > 0).astype(np.uint8)
    b2 = (re2 > 0).astype(np.uint8)
    return np.sum(b1 != b2) / len(b1)


# ──────────────────────────────────────────────
# Four-scenario distance computation
# ──────────────────────────────────────────────

def compute_distributions(binary_codes, labels, ctm, rng):
    unique_ids = np.unique(labels)

    dist_genuine_same   = []   # Genuine  same-key
    dist_impostor_same  = []   # Impostor same-key
    dist_genuine_diff   = []   # Genuine  different-key  (cancelability)
    dist_impostor_diff  = []   # Impostor different-key  (unlinkability)

    # ── 1. Genuine same-key ────────────────────────────────────────────
    # Same user, same key ke, different sample pairs.
    print("  Computing Genuine same-key distances...")
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re0, ke = ctm.enroll(binary_codes[idx[0]])
        for i in idx[1:]:
            re_i = ctm.authenticate(binary_codes[i], ke)
            dist_genuine_same.append(hamming_dist(re0, re_i))

    # ── 2. Impostor same-key ───────────────────────────────────────────
    # Different user, but impostor's sample is mapped through the genuine
    # user's key ke.  Models the "stolen-key" threat.
    print("  Computing Impostor same-key distances...")
    for _ in range(N_IMPOSTOR):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        re1, ke1 = ctm.enroll(binary_codes[idx1])
        re2      = ctm.authenticate(binary_codes[idx2], ke1)   # impostor uses ke1
        dist_impostor_same.append(hamming_dist(re1, re2))

    # ── 3. Genuine different-key  (CANCELABILITY) ──────────────────────
    # Same user, but two DIFFERENT random keys.
    # This simulates: old template revoked, new template issued.
    # Distance should be ~50% → old and new templates are unlinkable.
    print("  Computing Genuine different-key distances (cancelability)...")
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        # Enroll same user with two independently drawn keys
        re1, _  = ctm.enroll(binary_codes[idx[0]])       # key drawn from CTM internally
        re2, _  = ctm.enroll(binary_codes[idx[1]])       # different sample → different key
        dist_genuine_diff.append(hamming_dist(re1, re2))

    # ── 4. Impostor different-key  (UNLINKABILITY) ─────────────────────
    # Different users, each mapped with their own independent key.
    # Models cross-application comparison.
    print("  Computing Impostor different-key distances (unlinkability)...")
    for _ in range(N_IMPOSTOR):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        re1, _ = ctm.enroll(binary_codes[idx1])   # key1 for user1
        re2, _ = ctm.enroll(binary_codes[idx2])   # key2 for user2 (independent)
        dist_impostor_diff.append(hamming_dist(re1, re2))

    return (np.array(dist_genuine_same),
            np.array(dist_impostor_same),
            np.array(dist_genuine_diff),
            np.array(dist_impostor_diff))


# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────

def plot_distributions(d_gs, d_is, d_gd, d_id, output_dir):
    x = np.linspace(0, 0.8, 500)

    configs = [
        (d_gs, '#1f77b4', '-',  'Genuine  same-key      (normal auth)'),
        (d_is, '#d62728', '--', 'Impostor same-key       (stolen-key attack)'),
        (d_gd, '#2ca02c', '-',  'Genuine  different-key  (cancelability ✓)'),
        (d_id, '#ff7f0e', ':',  'Impostor different-key  (unlinkability ✓)'),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    for data, color, ls, label in configs:
        kde = gaussian_kde(data, bw_method=0.08)
        ax.plot(x, kde(x), color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(x, kde(x), alpha=0.08, color=color)

    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='50% (random)')
    ax.set_xlabel('Normalised Hamming Distance')
    ax.set_ylabel('Probability Density')
    ax.set_title('Cancelable Template Distance Distributions\n'
                 '(G=512, StableCTM, FVC2004)')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.8)

    save_path = os.path.join(output_dir, "distance_distributions.png")
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

    # Load data
    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8
    )

    # Load model
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024,
                               pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print("WARNING: using random model weights")
    model = model.to(device)
    model.set_beta(32)

    # Compute flip_rate from training set (needed for StableCTM)
    print("\nExtracting training codes...")
    train_binary, train_labels = extract_codes(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)

    # Extract test codes
    print("Extracting test codes...")
    test_binary, test_labels = extract_codes(model, test_loader, device)
    print(f"Test set: {test_binary.shape}, users: {len(np.unique(test_labels))}")

    # CTM with stable channel selection
    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO)

    rng = np.random.default_rng(42)

    print("\n" + "="*60)
    print("Computing four distance distributions...")
    print("="*60)
    d_gs, d_is, d_gd, d_id = compute_distributions(
        test_binary, test_labels, ctm, rng
    )

    # Statistics
    print("\n" + "="*60)
    print("RESULTS — Distance Statistics")
    print(f"{'Scenario':<35}  {'Mean':>7}  {'Std':>7}")
    print("-"*55)
    for name, data in [
        ("Genuine  same-key   (normal auth)", d_gs),
        ("Impostor same-key   (stolen-key)",  d_is),
        ("Genuine  diff-key   (cancelability)", d_gd),
        ("Impostor diff-key   (unlinkability)", d_id),
    ]:
        print(f"  {name:<35}  {data.mean()*100:>6.1f}%  {data.std()*100:>6.1f}%")

    print()
    # Interpretation
    gap_cancel = abs(d_gd.mean() - d_is.mean()) * 100
    gap_unlink = abs(d_id.mean() - d_is.mean()) * 100
    print(f"Cancelability check:   |Genuine diff-key  − Impostor same-key| = {gap_cancel:.2f}%")
    print(f"Unlinkability check:   |Impostor diff-key − Impostor same-key| = {gap_unlink:.2f}%")
    print("(Both should be near 0% to confirm the properties.)")

    # Plot
    plot_distributions(d_gs, d_is, d_gd, d_id, OUTPUT_DIR)

    # Save JSON
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "n_impostor_pairs": N_IMPOSTOR,
        "Genuine_same_key":      {"mean_%": round(d_gs.mean()*100, 2), "std_%": round(d_gs.std()*100, 2), "n": len(d_gs)},
        "Impostor_same_key":     {"mean_%": round(d_is.mean()*100, 2), "std_%": round(d_is.std()*100, 2), "n": len(d_is)},
        "Genuine_diff_key":      {"mean_%": round(d_gd.mean()*100, 2), "std_%": round(d_gd.std()*100, 2), "n": len(d_gd)},
        "Impostor_diff_key":     {"mean_%": round(d_id.mean()*100, 2), "std_%": round(d_id.std()*100, 2), "n": len(d_id)},
        "cancelability_gap_%":   round(gap_cancel, 3),
        "unlinkability_gap_%":   round(gap_unlink, 3),
    }
    json_path = os.path.join(OUTPUT_DIR, "cancelability_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
