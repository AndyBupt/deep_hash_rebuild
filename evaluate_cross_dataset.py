"""
evaluate_cross_dataset.py — Cross-Dataset Generalization Evaluation

Uses the FVC2004-trained model to evaluate on other FVC datasets
(FVC2002, FVC2006, etc.) WITHOUT retraining.

Key question: Does RGSS consistently outperform BCH on unseen datasets?

Output: results_cross_dataset/
  gs_{DATASET_NAME}.png      — BCH vs RGSS G-S curve for each dataset
  cross_dataset_results.json — Numerical results
  cross_dataset_summary.png  — Bar chart comparing all datasets

Usage:
  python evaluate_cross_dataset.py

Configure DATASETS below to point to your available FVC datasets.
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bchlib

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import StableCTM
from sstm_bch import SSTM_BCH
from sstm_polar_embed import SSTM_PolarEmbed


# ──────────────────────────────────────────────
# Config — add / remove datasets as available
# ──────────────────────────────────────────────
MODEL_PATH = "checkpoints/final_model.pth"
OUTPUT_DIR = "results_cross_dataset"
G            = 512
STABLE_RATIO = 0.8

# Each entry: (display_name, DATA_ROOT, DB_NAMES)
# Uncomment / add lines for datasets available on your server.
DATASETS = [
    (
        "FVC2004",
        "/root/autodl-tmp/FVC2004",
        ["DB1_A/image", "DB1_B/image",
         "DB2_A/image", "DB2_B/image",
         "DB3_A/image", "DB3_B/image"],
    ),
    # ── FVC2002 ────────────────────────────────────────────────────────
    (
        "FVC2002",
        "/root/autodl-tmp/FVC2002",
        ["DB1_A/image", "DB1_B/image",
         "DB2_A/image", "DB2_B/image",
         "DB3_A/image", "DB3_B/image"],
    ),
    # ── FVC2006 (only A variants available) ───────────────────────────
    (
        "FVC2006",
        "/root/autodl-tmp/FVC2006",
        ["DB1_A/image", "DB2_A/image", "DB3_A/image"],
    ),
]


# ──────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────

def extract_codes(model, loader, device):
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
# BCH parameter helper
# ──────────────────────────────────────────────

def get_bch_params(t_min=None, step=1):
    """Return BCH(m=9) params where 40 <= k_bits < G."""
    params = []
    for t in range(1, 57):
        try:
            b = bchlib.BCH(t=t, m=9)
            k_bytes = (b.n - b.ecc_bits) // 8
            k_bits = k_bytes * 8
            if 40 <= k_bits < G:
                params.append((9, b.t, k_bits))
        except Exception:
            break
    seen_k = {}
    for m, t, k in params:
        if k not in seen_k or t > seen_k[k][1]:
            seen_k[k] = (m, t, k)
    all_params = sorted(seen_k.values(), key=lambda x: x[2])
    if t_min is not None:
        all_params = [(m, t, k) for m, t, k in all_params if t >= t_min]
    return all_params[::step]


# ──────────────────────────────────────────────
# GAR computation
# ──────────────────────────────────────────────

def compute_gar_bch(binary_codes, labels, ctm, m, t):
    sstm = SSTM_BCH(G=G, m=m, t=t)
    unique_ids = np.unique(labels)
    pass_count = total = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(binary_codes[idx[0]])
        stored, _ = sstm.enroll(re)
        for i in idx[1:]:
            rp = ctm.authenticate(binary_codes[i], ke)
            ok, _ = sstm.authenticate(rp, stored)
            pass_count += int(ok)
            total += 1
    return pass_count / total if total > 0 else 0.0


def compute_gar_rgss(binary_codes, hash_codes, labels, ctm, k_bits, m, t):
    try:
        sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
    except AssertionError:
        return None
    unique_ids = np.unique(labels)
    pass_count = total = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(binary_codes[idx[0]])
        embed_e = hash_codes[idx[0]][ke]
        stored, _ = sstm.enroll(re, embed_e)
        for i in idx[1:]:
            rp = ctm.authenticate(binary_codes[i], ke)
            ok, _ = sstm.authenticate(rp, stored)
            pass_count += int(ok)
            total += 1
    return pass_count / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# Per-dataset evaluation
# ──────────────────────────────────────────────

def evaluate_dataset(name, data_root, db_names, model, device, flip_rate_src,
                     bch_params, output_dir):
    """Run BCH vs RGSS G-S curve evaluation on one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    # Load this dataset's test split
    _, test_loader, num_cls = build_dataloaders(
        data_root, db_names, train_ratio=0.7, batch_size=8
    )
    print(f"  Loaded: {num_cls} identities")

    # Extract codes using the FVC2004-trained model (no fine-tuning)
    print("  Extracting codes (cross-dataset, no retraining)...")
    binary_codes, hash_codes, labels = extract_codes(model, test_loader, device)
    print(f"  Test set: {binary_codes.shape}")

    # CTM uses flip_rate from FVC2004 training set (source domain)
    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate_src, stable_ratio=STABLE_RATIO)

    # Genuine flip rate on this dataset
    unique_ids = np.unique(labels)
    dists = []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(binary_codes[idx[0]])
        for i in idx[1:]:
            rp = ctm.authenticate(binary_codes[i], ke)
            d = ctm.hamming_distance(re, rp) / G
            dists.append(d)
    flip_rate_test = np.mean(dists) if dists else 0.0
    print(f"  Genuine flip rate: {flip_rate_test*100:.1f}%")

    # BCH G-S curve
    k_bits_bch, gars_bch = [], []
    print("\n  [BCH]...")
    for m, t, k_bits in bch_params:
        gar = compute_gar_bch(binary_codes, labels, ctm, m, t)
        gars_bch.append(gar * 100)
        k_bits_bch.append(k_bits)
        print(f"    k={k_bits:4d} bits  GAR={gar*100:.1f}%")

    # RGSS G-S curve
    k_bits_rgss, gars_rgss = [], []
    print("\n  [RGSS]...")
    for m, t, k_bits in bch_params:
        gar = compute_gar_rgss(binary_codes, hash_codes, labels, ctm, k_bits, m, t)
        if gar is None:
            continue
        gars_rgss.append(gar * 100)
        k_bits_rgss.append(k_bits)
        print(f"    k={k_bits:4d} bits  GAR={gar*100:.1f}%")

    # Inflection points
    bch_inf  = next((k for k, g in zip(k_bits_bch,  gars_bch)  if g < 50), None)
    rgss_inf = next((k for k, g in zip(k_bits_rgss, gars_rgss) if g < 50), None)
    bch_k50  = k_bits_bch[k_bits_bch.index(bch_inf)  - 1] if bch_inf and k_bits_bch.index(bch_inf)  > 0 else (k_bits_bch[-1]  if gars_bch[-1]  >= 50 else None)
    rgss_k50 = k_bits_rgss[k_bits_rgss.index(rgss_inf) - 1] if rgss_inf and k_bits_rgss.index(rgss_inf) > 0 else (k_bits_rgss[-1] if gars_rgss[-1] >= 50 else None)

    print(f"\n  BCH  GAR=50% inflection: k={bch_k50}  bits")
    print(f"  RGSS GAR=50% inflection: k={rgss_k50} bits")

    # Plot
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_bits_bch,  gars_bch,  'r-o', linewidth=2, markersize=4,
            label='BCH (Baseline)')
    ax.plot(k_bits_rgss, gars_rgss, 'b-s', linewidth=2, markersize=4,
            label='RGSS — Reliability-Guided (Proposed)')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S Curve: BCH vs RGSS — {name}\n'
                 f'(G={G}, StableCTM, FVC2004-trained model, genuine flip={flip_rate_test*100:.1f}%)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)
    save_path = os.path.join(output_dir, f"gs_{name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return {
        "genuine_flip_rate (%)": round(flip_rate_test * 100, 2),
        "BCH":  {"k_bits": k_bits_bch,  "GAR (%)": [round(g, 2) for g in gars_bch],
                 "inflection_k": bch_k50},
        "RGSS": {"k_bits": k_bits_rgss, "GAR (%)": [round(g, 2) for g in gars_rgss],
                 "inflection_k": rgss_k50},
    }


# ──────────────────────────────────────────────
# Summary bar chart
# ──────────────────────────────────────────────

def plot_summary(all_results, output_dir):
    names = list(all_results.keys())
    bch_k50s  = [all_results[n]["BCH"]["inflection_k"]  or 0 for n in names]
    rgss_k50s = [all_results[n]["RGSS"]["inflection_k"] or 0 for n in names]

    x = np.arange(len(names))
    w = 0.35
    _, ax = plt.subplots(figsize=(max(8, len(names) * 3), 6))

    bars_bch  = ax.bar(x - w/2, bch_k50s,  w, label='BCH (Baseline)',
                       color='#d62728', alpha=0.85)
    bars_rgss = ax.bar(x + w/2, rgss_k50s, w, label='RGSS (Proposed)',
                       color='#1f77b4', alpha=0.85)

    for bar in bars_bch:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 2,
                    f'{int(h)}b', ha='center', va='bottom', fontsize=9)
    for bar in bars_rgss:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 2,
                    f'{int(h)}b', ha='center', va='bottom', fontsize=9,
                    color='#1f77b4', fontweight='bold')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Security Level at GAR=50% (bits)')
    ax.set_title('Cross-Dataset Comparison: BCH vs RGSS\n'
                 '(FVC2004-trained model, G=512 bits)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(rgss_k50s + [1]) * 1.25)

    save_path = os.path.join(output_dir, "cross_dataset_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model (FVC2004-trained, NOT retrained on new datasets)
    # num_classes from FVC2004 training — needed only for classifier head,
    # but we only use the hash layer so it doesn't matter for cross-dataset eval.
    _, _, num_classes_src = build_dataloaders(
        DATASETS[0][1], DATASETS[0][2], train_ratio=0.7, batch_size=8
    )
    model = FingerprintHashNet(num_classes=num_classes_src, hash_dim=1024,
                               pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded: {MODEL_PATH}  (FVC2004-trained, used as-is)")
    else:
        print("WARNING: using random model weights")
    model = model.to(device)
    model.set_beta(32)

    # Compute flip_rate from FVC2004 TRAINING set (used by StableCTM for all datasets)
    print("\nComputing flip_rate from FVC2004 training set...")
    train_loader, _, _ = build_dataloaders(
        DATASETS[0][1], DATASETS[0][2], train_ratio=0.7, batch_size=8
    )
    model.eval()
    all_bin, all_lbl = [], []
    with torch.no_grad():
        for imgs, lbs in train_loader:
            _, _, bc = model(imgs.to(device))
            all_bin.append(bc.cpu().numpy())
            all_lbl.append(lbs.numpy())
    train_binary = np.vstack(all_bin)
    train_labels = np.concatenate(all_lbl)
    flip_rate_src = StableCTM.compute_flip_rate(train_binary, train_labels)
    print(f"FVC2004 training flip_rate mean: {flip_rate_src.mean()*100:.2f}%")

    # BCH params (same for all datasets)
    bch_params = get_bch_params()

    # Evaluate each dataset
    all_results = {}
    for name, data_root, db_names in DATASETS:
        result = evaluate_dataset(
            name, data_root, db_names,
            model, device, flip_rate_src, bch_params, OUTPUT_DIR
        )
        all_results[name] = result

    # Summary
    plot_summary(all_results, OUTPUT_DIR)

    print("\n" + "="*60)
    print("CROSS-DATASET SUMMARY")
    print(f"{'Dataset':<12}  {'Flip%':>6}  {'BCH k50':>8}  {'RGSS k50':>9}  {'Delta':>6}")
    print("-"*50)
    for name, r in all_results.items():
        flip = r["genuine_flip_rate (%)"]
        bk   = r["BCH"]["inflection_k"]  or 0
        rk   = r["RGSS"]["inflection_k"] or 0
        print(f"  {name:<10}  {flip:>5.1f}%  {bk:>7} b  {rk:>8} b  {rk-bk:>+5} b")

    # Save JSON
    all_results["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_path = os.path.join(OUTPUT_DIR, "cross_dataset_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
