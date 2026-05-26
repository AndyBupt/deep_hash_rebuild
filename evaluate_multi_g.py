"""
evaluate_multi_g.py — Multi-G Template Length Evaluation

Compare BCH vs RGSS across G ∈ {128, 256, 512} bits.

Key question:
  As G decreases, BCH's usable k range shrinks (partial-fill ratio k/G
  drops), while RGSS concentrates key bits in reliable channels and
  maintains a better GAR-k tradeoff.

Output: results_multi_g/
  gs_G128.png / gs_G256.png / gs_G512.png  — per-G curves
  gs_summary.png                            — inflection comparison bar chart
  multi_g_results.json

Usage:
  python evaluate_multi_g.py
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
from ctm import CTM, StableCTM
from sstm_bch import SSTM_BCH
from sstm_polar_embed import SSTM_PolarEmbed


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_multi_g"
G_VALUES     = [128, 256, 512]
STABLE_RATIO = 0.8


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
# BCH parameter helpers
# ──────────────────────────────────────────────

def get_bch_params_for_g(G, t_min=None, step=1):
    """Return BCH params (m=8 or m=9) where 40 <= k_bits < G.
    Tries both m=8 (n=255) and m=9 (n=511) to cover small G values.
    For each unique k, keeps the entry with the highest t (best error correction).
    """
    seen_k = {}

    for m in (8, 9):
        t_limit = 57 if m == 9 else 35
        for t in range(1, t_limit):
            try:
                b = bchlib.BCH(t=t, m=m)
                k_bytes = (b.n - b.ecc_bits) // 8
                k_bits = k_bytes * 8
                if 40 <= k_bits < G:
                    # Keep the (m, t) pair with the highest t for each k
                    if k_bits not in seen_k or t > seen_k[k_bits][1]:
                        seen_k[k_bits] = (m, t, k_bits)
            except Exception:
                break

    all_params = sorted(seen_k.values(), key=lambda x: x[2])
    if t_min is not None:
        all_params = [(m, t, k) for m, t, k in all_params if t >= t_min]
    return all_params[::step]


# ──────────────────────────────────────────────
# Core GAR computation helpers
# ──────────────────────────────────────────────

def compute_gar_bch(binary_codes, labels, ctm, G, m, t):
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


def compute_gar_rgss(binary_codes, hash_codes, labels, ctm, G, k_bits, m=9, t=56):
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
# Per-G experiment
# ──────────────────────────────────────────────

def run_gs_for_g(binary_codes, hash_codes, labels, flip_rate, G, output_dir):
    print(f"\n{'='*60}")
    print(f"G = {G} bits")
    print(f"{'='*60}")

    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    bch_params = get_bch_params_for_g(G)

    if not bch_params:
        print(f"  WARNING: No valid BCH params for G={G}, skipping.")
        return {}

    print(f"  BCH valid k range: {bch_params[0][2]}–{bch_params[-1][2]} bits "
          f"({len(bch_params)} points)")

    # ── BCH baseline ────────────────────────────
    k_bits_bch, gars_bch = [], []
    print("\n  [BCH]...")
    for m_b, t_b, k_b in bch_params:
        gar = compute_gar_bch(binary_codes, labels, ctm, G, m_b, t_b)
        gars_bch.append(gar * 100)
        k_bits_bch.append(k_b)
        print(f"    t={t_b:3d}  k={k_b:4d} bits  GAR={gar*100:.1f}%")

    # ── RGSS (full k range valid for this G) ───
    k_bits_rgss, gars_rgss = [], []
    print("\n  [RGSS]...")
    for m_b, t_b, k_b in bch_params:
        gar = compute_gar_rgss(binary_codes, hash_codes, labels, ctm,
                               G=G, k_bits=k_b, m=m_b, t=t_b)
        if gar is None:
            print(f"    t={t_b:3d}  k={k_b:4d} bits  SKIP")
            continue
        gars_rgss.append(gar * 100)
        k_bits_rgss.append(k_b)
        print(f"    t={t_b:3d}  k={k_b:4d} bits  GAR={gar*100:.1f}%")

    # ── Plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_bits_bch,  gars_bch,  'r-o', linewidth=2, markersize=4,
            label='BCH (Baseline)')
    ax.plot(k_bits_rgss, gars_rgss, 'b-s', linewidth=2, markersize=4,
            label='RGSS — Reliability-Guided (Proposed)')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')

    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S Curve: BCH vs RGSS  (G={G} bits, StableCTM)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)
    save_path = os.path.join(output_dir, f"gs_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    return {
        "BCH":  {"k_bits": k_bits_bch,  "GAR (%)": [round(g, 2) for g in gars_bch]},
        "RGSS": {"k_bits": k_bits_rgss, "GAR (%)": [round(g, 2) for g in gars_rgss]},
    }


# ──────────────────────────────────────────────
# Summary bar chart
# ──────────────────────────────────────────────

def plot_summary(all_g_results, output_dir):
    """Bar chart: GAR=50% inflection vs G for BCH and RGSS."""
    g_vals, bch_inflects, rgss_inflects = [], [], []

    for G, res in sorted(all_g_results.items()):
        g_vals.append(G)
        for name, key_list in [('BCH', bch_inflects), ('RGSS', rgss_inflects)]:
            r = res.get(name, {})
            pts = [(k, g) for k, g in zip(r.get("k_bits", []),
                                           r.get("GAR (%)", [])) if g >= 50]
            key_list.append(pts[-1][0] if pts else 0)

    x = np.arange(len(g_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_bch  = ax.bar(x - width/2, bch_inflects,  width, label='BCH (Baseline)',
                       color='#d62728', alpha=0.85)
    bars_rgss = ax.bar(x + width/2, rgss_inflects, width, label='RGSS (Proposed)',
                       color='#1f77b4', alpha=0.85)

    # value labels
    for bar in bars_bch:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 3,
                f'{int(h)}b', ha='center', va='bottom', fontsize=9)
    for bar in bars_rgss:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 3,
                f'{int(h)}b', ha='center', va='bottom', fontsize=9,
                color='#1f77b4', fontweight='bold')

    ax.set_xlabel('Template Length G (bits)')
    ax.set_ylabel('Security Level at GAR=50% (bits)')
    ax.set_title('GAR=50% Inflection vs Template Length G\n'
                 'BCH vs RGSS (FVC2004, StableCTM)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G={g}' for g in g_vals])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(rgss_inflects + [1]) * 1.2)

    save_path = os.path.join(output_dir, "gs_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {save_path}")


def plot_combined(all_g_results, output_dir):
    """3-panel figure: one column per G value."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, G in zip(axes, sorted(all_g_results.keys())):
        res = all_g_results[G]
        for name, color, marker in [('BCH', 'red', 'o'), ('RGSS', 'blue', 's')]:
            r = res.get(name, {})
            if not r.get("k_bits"):
                continue
            label = 'BCH (Baseline)' if name == 'BCH' else 'RGSS (Proposed)'
            ax.plot(r["k_bits"], r["GAR (%)"],
                    color=color, marker=marker, linewidth=2, markersize=4,
                    label=label)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'G = {G} bits', fontsize=12)
        ax.set_xlabel('Security Level k (bits)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 108)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('GAR (%)')
    fig.suptitle('BCH vs RGSS G-S Curves Across Template Lengths (FVC2004)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "gs_combined.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined: {save_path}")


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

    # Extract codes once (G-independent: always 1024-dim)
    print("\nExtracting training codes...")
    train_binary, _, train_labels = extract_codes_with_embed(
        model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    print(f"Flip rate mean: {flip_rate.mean()*100:.2f}%")

    print("\nExtracting test codes...")
    test_binary, test_hash, test_labels = extract_codes_with_embed(
        model, test_loader, device)
    print(f"Test set: {test_binary.shape}")

    # Run for each G
    all_g_results = {}
    for G in G_VALUES:
        res = run_gs_for_g(test_binary, test_hash, test_labels,
                           flip_rate, G, OUTPUT_DIR)
        all_g_results[G] = res

    # Summary plots
    plot_summary(all_g_results, OUTPUT_DIR)
    plot_combined(all_g_results, OUTPUT_DIR)

    # Print table
    print("\n" + "="*60)
    print("SUMMARY — GAR=50% inflection point")
    print(f"{'G':>8}  {'BCH k (bits)':>14}  {'RGSS k (bits)':>14}  {'Delta':>8}")
    print("-"*50)
    for G in sorted(all_g_results.keys()):
        res = all_g_results[G]
        bch_inf = rgss_inf = 0
        for name, var in [('BCH', 'bch_inf'), ('RGSS', 'rgss_inf')]:
            r = res.get(name, {})
            pts = [(k, g) for k, g in zip(r.get("k_bits", []),
                                           r.get("GAR (%)", [])) if g >= 50]
            val = pts[-1][0] if pts else 0
            if name == 'BCH':
                bch_inf = val
            else:
                rgss_inf = val
        delta = rgss_inf - bch_inf
        print(f"  G={G:3d}  BCH={bch_inf:4d} bits  RGSS={rgss_inf:4d} bits  "
              f"Δ={delta:+d} bits")

    # Save JSON
    serializable = {str(G): v for G, v in all_g_results.items()}
    serializable["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_path = os.path.join(OUTPUT_DIR, "multi_g_results.json")
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
