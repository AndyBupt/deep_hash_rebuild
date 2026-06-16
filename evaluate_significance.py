"""
evaluate_significance.py — Statistical Significance of BCH vs RGSS

Addresses the reviewer requirement:
  "Report mean ± std across multiple experimental runs to confirm
   the performance advantage is statistically significant."

Strategy:
  The neural network model is fixed (already trained).
  Variance is introduced by different train/test PERSON splits:
    - Different seeds → different subsets of users in test set
    - Each seed gives independent k₅₀ estimates for BCH and RGSS
  5 seeds → 5 paired observations → paired t-test

For each seed:
  1. Split users with that seed (train 70% / test 30%)
  2. Compute training-set flip rate (for stable pool selection)
  3. Run BCH G-S curve  → k₅₀_BCH
  4. Run RGSS G-S curve → k₅₀_RGSS
  5. Record advantage = k₅₀_RGSS − k₅₀_BCH

Output: results_significance/
  significance_boxplot.png       — box plots of k₅₀ per method + per-seed dots
  significance_results.json      — mean, std, t-stat, p-value

Usage:
  python evaluate_significance.py
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
from scipy import stats

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import StableCTM
from sstm_polar_embed import SSTM_PolarEmbed


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_significance"
G            = 512
STABLE_RATIO = 0.8
SEEDS        = [42, 123, 456, 789, 1024]   # 5 independent splits


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
# BCH parameter enumeration
# ──────────────────────────────────────────────

def get_bch_params(G, m=9):
    """Return sorted list of (m, t, k_bits) for BCH(m) with 40 <= k_bits < G."""
    params = []
    for t in range(1, 65):
        try:
            b = bchlib.BCH(t=t, m=m)
            k_bytes = (b.n - b.ecc_bits) // 8
            k_bits  = k_bytes * 8
            if 40 <= k_bits < G:
                params.append((m, b.t, k_bits))
        except Exception:
            break
    # Deduplicate on k_bits, keep max t
    seen = {}
    for entry in params:
        k = entry[2]
        if k not in seen or entry[1] > seen[k][1]:
            seen[k] = entry
    return sorted(seen.values(), key=lambda x: x[2])


# ──────────────────────────────────────────────
# GAR for one (method, k_bits)
# ──────────────────────────────────────────────

def compute_gar(binary_codes, hash_codes, labels, ctm,
                G, k_bits, m, t, use_tanh):
    """
    GAR for a single operating point.
    use_tanh=True  → RGSS (embed = tanh of enrollment sample)
    use_tanh=False → BCH baseline (embed = None, random position)
    """
    try:
        sstm_factory = lambda: SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
        _ = sstm_factory()   # validate params
    except (AssertionError, Exception):
        return None

    unique_ids = np.unique(labels)
    pass_cnt = total = 0

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue

        re, ke = ctm.enroll(binary_codes[idx[0]])
        sstm = sstm_factory()

        embed_e = hash_codes[idx[0]][ke].astype(np.float32) if use_tanh else None
        try:
            stored, _ = sstm.enroll(re, embed_e)
        except Exception:
            continue

        for i in idx[1:]:
            rp = ctm.authenticate(binary_codes[i], ke)
            ok, _ = sstm.authenticate(rp, stored)
            pass_cnt += int(ok)
            total += 1

    return pass_cnt / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# k₅₀ inflection point
# ──────────────────────────────────────────────

def find_k50(k_list, gar_list):
    """
    Last k where GAR >= 50%.
    Returns None if no such point exists.
    """
    candidates = [k for k, g in zip(k_list, gar_list) if g >= 50.0]
    return candidates[-1] if candidates else None


# ──────────────────────────────────────────────
# G-S curve for one split
# ──────────────────────────────────────────────

def run_gs_curve(binary_codes, hash_codes, labels, ctm, G, bch_params, use_tanh, name):
    k_list, gar_list = [], []
    print(f"      [{name}]", end="", flush=True)
    for m, t, k_bits in bch_params:
        gar = compute_gar(binary_codes, hash_codes, labels, ctm,
                          G, k_bits, m, t, use_tanh)
        if gar is None:
            continue
        k_list.append(k_bits)
        gar_list.append(gar * 100)
        print(f"  k={k_bits}:{gar*100:.0f}%", end="", flush=True)
    print()
    return k_list, gar_list


# ──────────────────────────────────────────────
# Per-seed experiment
# ──────────────────────────────────────────────

def run_one_seed(seed, model, device, bch_params):
    print(f"\n  ── Seed {seed} ──")
    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8, seed=seed
    )

    print(f"    Extracting codes (train)...")
    train_binary, _, train_labels = extract_codes_with_embed(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)

    print(f"    Extracting codes (test)...")
    test_binary, test_hash, test_labels = extract_codes_with_embed(
        model, test_loader, device)
    n_users = len(np.unique(test_labels))
    print(f"    Test users: {n_users}")

    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO)

    k_bch,  gar_bch  = run_gs_curve(test_binary, test_hash, test_labels,
                                     ctm, G, bch_params, False, "BCH")
    k_rgss, gar_rgss = run_gs_curve(test_binary, test_hash, test_labels,
                                     ctm, G, bch_params, True,  "RGSS")

    k50_bch  = find_k50(k_bch,  gar_bch)
    k50_rgss = find_k50(k_rgss, gar_rgss)

    print(f"    k₅₀ BCH={k50_bch}  RGSS={k50_rgss}  "
          f"Δ={None if k50_bch is None or k50_rgss is None else k50_rgss - k50_bch}")

    return {
        "seed": seed,
        "n_test_users": n_users,
        "k50_BCH":  k50_bch,
        "k50_RGSS": k50_rgss,
        "advantage": (k50_rgss - k50_bch) if (k50_bch and k50_rgss) else None,
        "curve_BCH":  {"k": k_bch,  "gar": gar_bch},
        "curve_RGSS": {"k": k_rgss, "gar": gar_rgss},
    }


# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────

def plot_results(seed_results, output_dir):
    valid = [r for r in seed_results if r["k50_BCH"] and r["k50_RGSS"]]
    k50_bch  = [r["k50_BCH"]  for r in valid]
    k50_rgss = [r["k50_RGSS"] for r in valid]
    seeds    = [r["seed"]     for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: box plot + dots
    ax = axes[0]
    bp = ax.boxplot([k50_bch, k50_rgss],
                    labels=["BCH (baseline)", "RGSS (proposed)"],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    bp['boxes'][1].set_facecolor('lightgreen')

    # Individual seed dots
    jitter = np.array([-0.1, 0.0, 0.1, -0.05, 0.05])[:len(seeds)]
    for i, (kb, kr, s, j) in enumerate(zip(k50_bch, k50_rgss, seeds, jitter)):
        ax.plot(1 + j, kb, 'o', color='steelblue',  markersize=8, zorder=5)
        ax.plot(2 + j, kr, 'o', color='darkgreen',  markersize=8, zorder=5)
        ax.plot([1 + j, 2 + j], [kb, kr], '-', color='gray', alpha=0.4, zorder=3)

    ax.set_ylabel('k₅₀ inflection point (bits)')
    ax.set_title(f'k₅₀ across {len(seeds)} random splits\n'
                 f'BCH: {np.mean(k50_bch):.1f}±{np.std(k50_bch):.1f}   '
                 f'RGSS: {np.mean(k50_rgss):.1f}±{np.std(k50_rgss):.1f} bits')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=max(0, min(k50_bch + k50_rgss) - 30))

    # Right: advantage per seed + mean±std band
    ax2 = axes[1]
    advantages = [kr - kb for kb, kr in zip(k50_bch, k50_rgss)]
    x = np.arange(len(seeds))
    ax2.bar(x, advantages, color=['#2ca02c' if a > 0 else '#d62728' for a in advantages],
            alpha=0.75, edgecolor='black')
    ax2.axhline(y=np.mean(advantages), color='navy', linestyle='--', linewidth=2,
                label=f'Mean Δ = {np.mean(advantages):.1f} bits')
    ax2.axhspan(np.mean(advantages) - np.std(advantages),
                np.mean(advantages) + np.std(advantages),
                alpha=0.15, color='navy', label='±1 std')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seed={s}" for s in seeds], rotation=15)
    ax2.set_ylabel('RGSS − BCH advantage (bits)')
    ax2.set_title(f'k₅₀ advantage: RGSS − BCH\n'
                  f'Mean={np.mean(advantages):.1f} bits, Std={np.std(advantages):.1f} bits')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "significance_boxplot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_curves(seed_results, output_dir):
    """Plot G-S curves for all seeds, BCH vs RGSS."""
    colors_bch  = plt.cm.Blues(np.linspace(0.4, 0.9, len(seed_results)))
    colors_rgss = plt.cm.Greens(np.linspace(0.4, 0.9, len(seed_results)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(seed_results):
        if not r["curve_BCH"]["k"]:
            continue
        ax.plot(r["curve_BCH"]["k"],  r["curve_BCH"]["gar"],
                color=colors_bch[i],  linestyle='--', linewidth=1.5,
                label=f'BCH  seed={r["seed"]}')
        ax.plot(r["curve_RGSS"]["k"], r["curve_RGSS"]["gar"],
                color=colors_rgss[i], linestyle='-',  linewidth=1.5,
                label=f'RGSS seed={r["seed"]}')

    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
               label='GAR = 50% (k₅₀ threshold)')
    ax.set_xlabel('Security parameter k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S Curves across {len(seed_results)} random splits (G={G})\n'
                 f'Dashed = BCH, Solid = RGSS')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 102)

    save_path = os.path.join(output_dir, "gs_curves_all_seeds.png")
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

    # Load model once — shared across all seed runs
    # Use seed=42 just to determine num_classes (any seed works)
    _, _, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8, seed=42
    )
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024, pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print("WARNING: using random model weights")
    model = model.to(device)
    model.set_beta(32)

    # Pre-compute BCH params once
    bch_params = get_bch_params(G, m=9)
    print(f"\nBCH params: {len(bch_params)} operating points, "
          f"k ∈ [{bch_params[0][2]}, {bch_params[-1][2]}] bits")

    # ── Run all seeds ──────────────────────────
    seed_results = []
    for seed in SEEDS:
        result = run_one_seed(seed, model, device, bch_params)
        seed_results.append(result)

    # ── Statistics ────────────────────────────
    valid = [r for r in seed_results if r["k50_BCH"] and r["k50_RGSS"]]
    k50_bch  = np.array([r["k50_BCH"]  for r in valid])
    k50_rgss = np.array([r["k50_RGSS"] for r in valid])
    advantages = k50_rgss - k50_bch

    # Paired t-test: H0: mean advantage = 0
    t_stat, p_value = stats.ttest_rel(k50_rgss, k50_bch)

    print(f"\n{'='*60}")
    print(f"Statistical Significance Summary  (N={len(valid)} seeds)")
    print("="*60)
    print(f"  BCH  k₅₀: {k50_bch.mean():.1f} ± {k50_bch.std():.1f} bits")
    print(f"  RGSS k₅₀: {k50_rgss.mean():.1f} ± {k50_rgss.std():.1f} bits")
    print(f"  Advantage: {advantages.mean():.1f} ± {advantages.std():.1f} bits")
    print(f"  Paired t-test: t={t_stat:.3f},  p={p_value:.4f}")
    if p_value < 0.05:
        print(f"  → STATISTICALLY SIGNIFICANT (p<0.05) ✓")
    elif p_value < 0.10:
        print(f"  → Marginally significant (p<0.10)")
    else:
        print(f"  → Not significant at p<0.05 (investigate)")

    # ── Plots ─────────────────────────────────
    plot_results(seed_results, OUTPUT_DIR)
    plot_all_curves(seed_results, OUTPUT_DIR)

    # ── Save JSON ─────────────────────────────
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G, "seeds": SEEDS,
        "n_valid_seeds": len(valid),
        "BCH": {
            "k50_per_seed": k50_bch.tolist(),
            "mean_k50":     round(float(k50_bch.mean()), 2),
            "std_k50":      round(float(k50_bch.std()),  2),
        },
        "RGSS": {
            "k50_per_seed": k50_rgss.tolist(),
            "mean_k50":     round(float(k50_rgss.mean()), 2),
            "std_k50":      round(float(k50_rgss.std()),  2),
        },
        "advantage_RGSS_minus_BCH": {
            "per_seed": advantages.tolist(),
            "mean":     round(float(advantages.mean()), 2),
            "std":      round(float(advantages.std()),  2),
        },
        "paired_ttest": {
            "t_statistic": round(float(t_stat), 4),
            "p_value":     round(float(p_value), 6),
            "significant_at_0.05": bool(p_value < 0.05),
        },
        "per_seed_detail": [
            {k: v for k, v in r.items() if k != "curve_BCH" and k != "curve_RGSS"}
            for r in seed_results
        ],
    }
    json_path = os.path.join(OUTPUT_DIR, "significance_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
