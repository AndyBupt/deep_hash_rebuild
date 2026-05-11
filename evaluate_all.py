"""
evaluate_all.py — Complete Evaluation Pipeline

Experiments:
  Exp 1: RS Code vs BCH Code (G-S Curve) — proving RS fails, BCH works
  Exp 2: ROC / EER curves — model quality
  Exp 3: Genuine/Impostor distance distribution
  Exp 4: Ablation (CTM vs StableCTM vs BioHashing + BCH) — frontend comparison
  Exp 5: BCH vs RGSS vs SCL+CRC (G-S Curve) — SSTM method comparison
         RGSS = Reliability-Guided Secure Sketch (aka PolarEmbed)

All figures use English labels.
Output: results_all/

Usage:
  python evaluate_all.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import bchlib

from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import CTM, StableCTM
from biohashing import BioHashing
from sstm import SSTM
from sstm_bch import SSTM_BCH
from sstm_polar_embed import SSTM_PolarEmbed
from sstm_polar_scl import SSTM_PolarSCL


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_all"
G            = 512
STABLE_RATIO = 0.8
SCL_L        = 8
SCL_CRC      = 8


# ──────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────

def extract_codes(model, loader, device):
    """Extract binary codes only."""
    model.eval()
    all_binary, all_labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, _, binary_c = model(imgs.to(device))
            all_binary.append(binary_c.cpu().numpy())
            all_labels.append(lbs.numpy())
    return np.vstack(all_binary), np.concatenate(all_labels)


def extract_codes_with_embed(model, loader, device):
    """Extract binary codes + tanh continuous values (confidence)."""
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

def get_bch_params(t_min=None, step=1):
    """Return sorted (m, t, k_bits) tuples for BCH(m=9)."""
    params = []
    for t in range(1, 57):
        try:
            b = bchlib.BCH(t=t, m=9)
            k_bytes = (b.n - b.ecc_bits) // 8
            k_bits = k_bytes * 8
            if k_bits >= 40:
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
# Experiment 1: RS vs BCH (G-S Curve)
# ──────────────────────────────────────────────

def run_exp1_rs_vs_bch(binary_codes, labels, ctm, output_dir):
    """
    Exp 1: RS Code vs BCH Code G-S Curve.
    Demonstrates RS code failure (symbol error rate ~84%) and BCH solution.
    """
    print("\n" + "="*60)
    print("Exp 1: RS Code vs BCH Code (G-S Curve)")
    print("="*60)

    bch_params = get_bch_params()
    unique_ids = np.unique(labels)

    # RS curve — using K values corresponding to BCH k_bits
    k_bits_rs, gars_rs = [], []
    N = G // 8
    K_values = list(range(7, N, 2))
    print("\n[RS Code]...")
    for K in K_values:
        if G // 8 <= K:
            continue
        sstm = SSTM(G=G, K=K)
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
        gar = pass_count / total if total > 0 else 0.0
        gars_rs.append(gar * 100)
        k_bits_rs.append(K * 8)
        print(f"  K={K:3d}  k={K*8:4d} bits  GAR={gar*100:.1f}%")

    # BCH curve
    k_bits_bch, gars_bch = [], []
    print("\n[BCH Code]...")
    for m, t, k_bits in bch_params:
        sstm = SSTM_BCH(G=G, m=m, t=t)
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
        gar = pass_count / total if total > 0 else 0.0
        gars_bch.append(gar * 100)
        k_bits_bch.append(k_bits)
        print(f"  t={t:3d}  k={k_bits:4d} bits  GAR={gar*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_bits_rs, gars_rs, 'r-o', linewidth=2, markersize=4,
            label='RS Code (Symbol-level ECC, fails)')
    ax.plot(k_bits_bch, gars_bch, 'b-s', linewidth=2, markersize=4,
            label='BCH Code (Bit-level ECC)')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'Exp 1: RS vs BCH Code (G={G} bits, StableCTM)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)
    save_path = os.path.join(output_dir, "exp1_rs_vs_bch.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return {
        "RS":  {"k_bits": k_bits_rs,  "GAR (%)": [round(g,2) for g in gars_rs]},
        "BCH": {"k_bits": k_bits_bch, "GAR (%)": [round(g,2) for g in gars_bch]},
    }


# ──────────────────────────────────────────────
# Experiment 2: ROC / EER
# ──────────────────────────────────────────────

def run_exp2_roc(binary_codes, labels, ctm, output_dir):
    """
    Exp 2: ROC curves and EER for the base system (StableCTM, no SSTM).
    Proves model discriminative quality.
    """
    print("\n" + "="*60)
    print("Exp 2: ROC / EER Curves")
    print("="*60)

    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)

    genuine_dists, impostor_sk_dists, impostor_uk_dists = [], [], []

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(binary_codes[idx[0]])
        for i in idx[1:]:
            rp = ctm.authenticate(binary_codes[i], ke)
            d = ctm.hamming_distance(re, rp) / ctm.G
            genuine_dists.append(d)

    n_imp = min(len(genuine_dists) * 5, 2000)
    for _ in range(n_imp):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        re, ke = ctm.enroll(binary_codes[idx1])
        rp_sk = ctm.authenticate(binary_codes[idx2], ke)
        d_sk = ctm.hamming_distance(re, rp_sk) / ctm.G
        impostor_sk_dists.append(d_sk)
        _, ke_rand = ctm.enroll(binary_codes[idx2],
                                seed=int(rng.integers(0, 99999)))
        rp_uk = ctm.authenticate(binary_codes[idx2], ke_rand)
        d_uk = ctm.hamming_distance(re, rp_uk) / ctm.G
        impostor_uk_dists.append(d_uk)

    genuine_dists  = np.array(genuine_dists)
    impostor_sk_dists = np.array(impostor_sk_dists)
    impostor_uk_dists = np.array(impostor_uk_dists)

    def compute_eer(gen, imp):
        scores = np.concatenate([-gen, -imp])
        y = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
        fpr, tpr, _ = roc_curve(y, scores)
        fnr = 1 - tpr
        idx = np.argmin(np.abs(fpr - fnr))
        return (fpr[idx]+fnr[idx])/2, fpr, tpr

    eer_sk, fpr_sk, tpr_sk = compute_eer(genuine_dists, impostor_sk_dists)
    eer_uk, fpr_uk, tpr_uk = compute_eer(genuine_dists, impostor_uk_dists)
    auc_sk = auc(fpr_sk, tpr_sk)
    auc_uk = auc(fpr_uk, tpr_uk)

    print(f"EER (Stolen Key):  {eer_sk*100:.2f}%  AUC={auc_sk:.4f}")
    print(f"EER (Unknown Key): {eer_uk*100:.2f}%  AUC={auc_uk:.4f}")
    print(f"Genuine flip rate: mean={genuine_dists.mean()*100:.1f}%  "
          f"std={genuine_dists.std()*100:.1f}%")

    # ROC plot (log-scale x-axis)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.semilogx(fpr_sk*100, tpr_sk*100, 'r--', linewidth=2,
                label=f'Stolen Key (AUC={auc_sk:.3f}, EER={eer_sk*100:.2f}%)')
    ax.semilogx(fpr_uk*100, tpr_uk*100, 'b-', linewidth=2,
                label=f'Unknown Key (AUC={auc_uk:.3f}, EER={eer_uk*100:.2f}%)')
    ax.scatter([eer_sk*100], [100-eer_sk*100], color='red', s=60, zorder=5)
    ax.scatter([eer_uk*100], [100-eer_uk*100], color='blue', s=60, zorder=5)
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(0, 105)
    ax.set_xlabel('FAR (%)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'ROC Curve (G={G}, StableCTM)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Distance distribution plot
    ax2 = axes[1]
    x = np.linspace(0, 1, 300)
    ax2.plot(x, norm.pdf(x, genuine_dists.mean(), genuine_dists.std()),
             'b-', linewidth=2, label=f'Genuine (mean={genuine_dists.mean()*100:.1f}%)')
    ax2.plot(x, norm.pdf(x, impostor_sk_dists.mean(), impostor_sk_dists.std()),
             'r--', linewidth=2,
             label=f'Impostor Stolen Key (mean={impostor_sk_dists.mean()*100:.1f}%)')
    ax2.plot(x, norm.pdf(x, impostor_uk_dists.mean(), impostor_uk_dists.std()),
             'g-.', linewidth=2,
             label=f'Impostor Unknown Key (mean={impostor_uk_dists.mean()*100:.1f}%)')
    ax2.fill_between(x, norm.pdf(x, genuine_dists.mean(), genuine_dists.std()),
                     alpha=0.15, color='blue')
    ax2.fill_between(x, norm.pdf(x, impostor_sk_dists.mean(), impostor_sk_dists.std()),
                     alpha=0.15, color='red')
    ax2.set_xlabel('Normalized Hamming Distance')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Genuine / Impostor Distance Distribution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp2_roc_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return {
        "EER_stolen_key (%)":  round(eer_sk*100, 2),
        "EER_unknown_key (%)": round(eer_uk*100, 2),
        "AUC_stolen_key":      round(auc_sk, 4),
        "AUC_unknown_key":     round(auc_uk, 4),
        "genuine_flip_rate (%)": round(genuine_dists.mean()*100, 2),
    }


# ──────────────────────────────────────────────
# Experiment 3: Bit flip rate analysis + CTM comparison
# ──────────────────────────────────────────────

def run_exp3_ctm_analysis(train_binary, train_labels,
                           test_binary, test_labels,
                           flip_rate, output_dir):
    """
    Exp 3: Bit flip rate distribution + CTM vs StableCTM comparison.
    """
    print("\n" + "="*60)
    print("Exp 3: Bit Flip Rate Analysis & CTM Comparison")
    print("="*60)

    hash_dim = train_binary.shape[1]
    rng = np.random.default_rng(42)

    # Flip rate distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(flip_rate, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=flip_rate.mean(), color='red', linestyle='--',
               label=f'Mean={flip_rate.mean()*100:.1f}%')
    ax.axvline(x=0.05, color='green', linestyle=':', label='5% threshold')
    ax.set_xlabel('Flip Rate')
    ax.set_ylabel('Number of Bits')
    ax.set_title('Per-bit Flip Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    sorted_flip = np.sort(flip_rate)
    ax2 = axes[1]
    ax2.plot(range(hash_dim), sorted_flip, 'b-', linewidth=1)
    ax2.axhline(y=0.05, color='green', linestyle='--', label='5% threshold')
    ax2.axhline(y=0.10, color='orange', linestyle='--', label='10% threshold')
    ax2.axhline(y=flip_rate.mean(), color='red', linestyle='--',
                label=f'Mean={flip_rate.mean()*100:.1f}%')
    n_stable = (flip_rate < 0.10).sum()
    ax2.axvline(x=n_stable, color='orange', linestyle=':',
                label=f'Top {n_stable} bits < 10%')
    ax2.set_xlabel('Bit Position (sorted by flip rate)')
    ax2.set_ylabel('Flip Rate')
    ax2.set_title('Bit Stability Ranking (basis for StableCTM)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "exp3_flip_rate_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # CTM vs StableCTM genuine distance comparison
    n_stable_pool = max(int(hash_dim * 0.8), G + 1)
    stable_pool = np.argsort(flip_rate)[:n_stable_pool]

    random_indices = rng.choice(hash_dim, size=G, replace=False)
    random_indices = np.sort(random_indices)
    stable_indices = rng.choice(stable_pool, size=G, replace=False)
    stable_indices = np.sort(stable_indices)

    def compute_genuine_dists(indices):
        dists = []
        unique_ids = np.unique(test_labels)
        for uid in unique_ids:
            idx = np.where(test_labels == uid)[0]
            if len(idx) < 2:
                continue
            ref = test_binary[idx[0]][indices]
            for i in idx[1:]:
                probe = test_binary[i][indices]
                d = np.sum((ref > 0) != (probe > 0)) / len(indices)
                dists.append(d)
        return np.array(dists)

    dists_random = compute_genuine_dists(random_indices)
    dists_stable = compute_genuine_dists(stable_indices)

    print(f"CTM (random):   mean={dists_random.mean()*100:.1f}%  "
          f"std={dists_random.std()*100:.1f}%")
    print(f"StableCTM:      mean={dists_stable.mean()*100:.1f}%  "
          f"std={dists_stable.std()*100:.1f}%")

    from scipy.stats import norm as scipy_norm
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 0.6, 300)
    g_mean, g_std = dists_random.mean(), dists_random.std()
    s_mean, s_std = dists_stable.mean(), dists_stable.std()
    ax.plot(x, scipy_norm.pdf(x, g_mean, g_std), 'r-', linewidth=2,
            label=f'CTM - Random Selection (mean={g_mean*100:.1f}%)')
    ax.plot(x, scipy_norm.pdf(x, s_mean, s_std), 'b-', linewidth=2,
            label=f'StableCTM - Stable Pool Selection (mean={s_mean*100:.1f}%)')
    ax.fill_between(x, scipy_norm.pdf(x, g_mean, g_std), alpha=0.15, color='red')
    ax.fill_between(x, scipy_norm.pdf(x, s_mean, s_std), alpha=0.15, color='blue')
    ax.set_xlabel('Normalized Genuine Hamming Distance (Flip Rate)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'CTM vs StableCTM: Genuine Flip Rate Distribution (G={G})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path2 = os.path.join(output_dir, "exp3_ctm_vs_stablectm.png")
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path} and {save_path2}")

    return {
        "flip_rate_mean (%)": round(flip_rate.mean()*100, 2),
        "CTM_genuine_flip (%)":       round(dists_random.mean()*100, 2),
        "StableCTM_genuine_flip (%)": round(dists_stable.mean()*100, 2),
    }


# ──────────────────────────────────────────────
# Experiment 4: Ablation — frontend methods
# ──────────────────────────────────────────────

def run_exp4_ablation(binary_codes, hash_codes, labels,
                       flip_rate, output_dir):
    """
    Exp 4: Ablation study — CTM vs StableCTM vs BioHashing + BCH SSTM.
    All use the same BCH SSTM (m=9, t from full range).
    """
    print("\n" + "="*60)
    print("Exp 4: Ablation — Frontend Methods (+ BCH SSTM)")
    print("="*60)

    import bchlib as _bchlib
    bch_params = []
    for t in range(1, 57):
        try:
            b = _bchlib.BCH(t=t, m=9)
            k_bytes = (b.n - b.ecc_bits) // 8
            k_bits = k_bytes * 8
            if k_bits >= 40:
                bch_params.append((9, b.t, k_bits))
        except Exception:
            break
    seen_k = {}
    for m, t, k in bch_params:
        if k not in seen_k or t > seen_k[k][1]:
            seen_k[k] = (m, t, k)
    bch_params = sorted(seen_k.values(), key=lambda x: x[2])

    unique_ids = np.unique(labels)

    methods = {
        "CTM (Random)":  CTM(hash_dim=1024, G=G),
        "StableCTM":     StableCTM(hash_dim=1024, G=G,
                                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO),
        "BioHashing":    BioHashing(hash_dim=1024, G=G),
    }
    colors  = {"CTM (Random)": "green", "StableCTM": "blue", "BioHashing": "red"}
    markers = {"CTM (Random)": "^",     "StableCTM": "s",    "BioHashing": "o"}

    results = {}
    for name, ctm_method in methods.items():
        print(f"\n[{name}]...")
        # Genuine flip rate
        genuine_dists = []
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm_method.enroll(binary_codes[idx[0]])
            for i in idx[1:]:
                rp = ctm_method.authenticate(binary_codes[i], ke)
                d = ctm_method.hamming_distance(re, rp) / ctm_method.G
                genuine_dists.append(d)
        flip = float(np.mean(genuine_dists))

        # G-S curve
        k_bits_list, gars = [], []
        for m, t, k_bits in bch_params:
            sstm = SSTM_BCH(G=G, m=m, t=t)
            pass_count = total = 0
            for uid in unique_ids:
                idx = np.where(labels == uid)[0]
                if len(idx) < 2:
                    continue
                re, ke = ctm_method.enroll(binary_codes[idx[0]])
                stored, _ = sstm.enroll(re)
                for i in idx[1:]:
                    rp = ctm_method.authenticate(binary_codes[i], ke)
                    ok, _ = sstm.authenticate(rp, stored)
                    pass_count += int(ok)
                    total += 1
            gar = pass_count / total if total > 0 else 0.0
            gars.append(gar * 100)
            k_bits_list.append(k_bits)

        inflect = [(k, g) for k, g in zip(k_bits_list, gars) if g >= 50]
        inflect_str = f"k={inflect[-1][0]} bits" if inflect else "N/A"
        print(f"  Genuine flip={flip*100:.1f}%  GAR=50% inflection: {inflect_str}")

        results[name] = {
            "genuine_flip_rate (%)": round(flip*100, 2),
            "k_bits": k_bits_list,
            "GAR (%)": [round(g, 2) for g in gars],
        }

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in results.items():
        ax.plot(r["k_bits"], r["GAR (%)"],
                color=colors[name], marker=markers[name],
                linewidth=2, markersize=4,
                label=f"{name} (flip={r['genuine_flip_rate (%)']:.1f}%)")
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'Exp 4: Ablation — Frontend Methods + BCH SSTM (G={G})')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)
    save_path = os.path.join(output_dir, "exp4_ablation_frontend.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    return results


# ──────────────────────────────────────────────
# Experiment 5: SSTM method comparison
# BCH vs RGSS vs SCL+CRC
# ──────────────────────────────────────────────

def _compute_gar_for_sstm(binary_codes, hash_codes, labels, ctm,
                           sstm_fn, use_embed=False):
    """Helper: compute GAR for a given SSTM factory function."""
    unique_ids = np.unique(labels)
    pass_count = total = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(binary_codes[idx[0]])
        if use_embed:
            embed_e = hash_codes[idx[0]][ke]
            stored, _ = sstm_fn().enroll(re, embed_e)
        else:
            stored, _ = sstm_fn().enroll(re)
        for i in idx[1:]:
            rp = ctm.authenticate(binary_codes[i], ke)
            if use_embed:
                embed_p = hash_codes[i][ke]
                ok, _ = sstm_fn().authenticate(rp, stored, embed_p)
            else:
                ok, _ = sstm_fn().authenticate(rp, stored)
            pass_count += int(ok)
            total += 1
    return pass_count / total if total > 0 else 0.0


def run_exp5_sstm_comparison(binary_codes, hash_codes, labels,
                              ctm, flip_prob, output_dir):
    """
    Exp 5: SSTM Method Comparison (G-S Curve).
      - BCH: baseline
      - RGSS (Reliability-Guided Secure Sketch): proposed method
      - SCL L=1 no-CRC: equivalent to SC decoding (sanity check)
      - SCL L=8 CRC-8: full polar code with list decoding
    BCH and RGSS: full sampling.
    SCL: sparse sampling (t>=30, step=4) for speed.
    """
    print("\n" + "="*60)
    print("Exp 5: SSTM Method Comparison (G-S Curve)")
    print("="*60)

    bch_params_full   = get_bch_params()
    bch_params_sparse = get_bch_params(t_min=30, step=4)
    unique_ids = np.unique(labels)

    results = {}

    # ── BCH (baseline) ──────────────────────────
    k_bits_bch, gars_bch = [], []
    print("\n[BCH] full sampling...")
    for m, t, k_bits in bch_params_full:
        gar = _compute_gar_for_sstm(
            binary_codes, hash_codes, labels, ctm,
            sstm_fn=lambda m=m, t=t: SSTM_BCH(G=G, m=m, t=t),
            use_embed=False
        )
        gars_bch.append(gar * 100)
        k_bits_bch.append(k_bits)
        print(f"  t={t:3d}  k={k_bits:4d} bits  GAR={gar*100:.1f}%")
    results["BCH"] = {"k_bits": k_bits_bch,
                      "GAR (%)": [round(g,2) for g in gars_bch]}

    # ── RGSS (PolarEmbed) ────────────────────────
    k_bits_rgss, gars_rgss = [], []
    print("\n[RGSS - Reliability-Guided Secure Sketch] full sampling...")
    for m, t, k_bits in bch_params_full:
        try:
            gar = _compute_gar_for_sstm(
                binary_codes, hash_codes, labels, ctm,
                sstm_fn=lambda m=m, t=t, k=k_bits: SSTM_PolarEmbed(G=G, k_bits=k, m=m, t=t),
                use_embed=True
            )
        except Exception as e:
            print(f"  skip k={k_bits}: {e}")
            continue
        gars_rgss.append(gar * 100)
        k_bits_rgss.append(k_bits)
        print(f"  t={t:3d}  k={k_bits:4d} bits  GAR={gar*100:.1f}%")
    results["RGSS"] = {"k_bits": k_bits_rgss,
                       "GAR (%)": [round(g,2) for g in gars_rgss]}

    # ── SCL L=1 no-CRC (= SC, sanity check) ──────
    k_bits_sc, gars_sc = [], []
    print("\n[SCL L=1 no-CRC (≡ SC)] sparse sampling...")
    for m, t, k_bits in bch_params_sparse:
        k_total = k_bits  # no CRC
        if k_total >= G:
            continue
        try:
            gar = _compute_gar_for_sstm(
                binary_codes, hash_codes, labels, ctm,
                sstm_fn=lambda k=k_bits: SSTM_PolarSCL(
                    G=G, k=k, flip_prob=flip_prob, L=1, crc_bits=0),
                use_embed=True
            )
        except Exception as e:
            print(f"  skip k={k_bits}: {e}")
            continue
        gars_sc.append(gar * 100)
        k_bits_sc.append(k_bits)
        print(f"  t={t:3d}  k={k_bits:4d} bits  GAR={gar*100:.1f}%")
    results["SCL_L1_noCRC"] = {
        "k_bits": k_bits_sc,
        "GAR (%)": [round(g,2) for g in gars_sc],
        "note": "L=1 no-CRC ≡ SC decoding (sanity check)"
    }

    # ── SCL L=8 CRC-8 ────────────────────────────
    k_bits_scl, gars_scl = [], []
    print(f"\n[SCL L={SCL_L} CRC-{SCL_CRC}] sparse sampling...")
    for m, t, k_bits in bch_params_sparse:
        k_total = k_bits + SCL_CRC
        if k_total >= G:
            continue
        try:
            gar = _compute_gar_for_sstm(
                binary_codes, hash_codes, labels, ctm,
                sstm_fn=lambda k=k_bits: SSTM_PolarSCL(
                    G=G, k=k, flip_prob=flip_prob, L=SCL_L, crc_bits=SCL_CRC),
                use_embed=True
            )
        except Exception as e:
            print(f"  skip k={k_bits}: {e}")
            continue
        gars_scl.append(gar * 100)
        k_bits_scl.append(k_bits)
        print(f"  t={t:3d}  k={k_bits:4d} bits  GAR={gar*100:.1f}%")
    results["SCL_CRC"] = {
        "k_bits": k_bits_scl,
        "GAR (%)": [round(g,2) for g in gars_scl],
        "note": f"SCL L={SCL_L} CRC-{SCL_CRC}"
    }

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        "BCH":          ("r-o",  "BCH Code (Baseline)"),
        "RGSS":         ("b-s",  "RGSS - Reliability-Guided Secure Sketch (Proposed)"),
        "SCL_L1_noCRC": ("k--",  "Polar SC (L=1, no CRC, sanity check)"),
        "SCL_CRC":      ("m-^",  f"Polar SCL (L={SCL_L}, CRC-{SCL_CRC})"),
    }
    for name, (style, label) in styles.items():
        if name not in results:
            continue
        r = results[name]
        ax.plot(r["k_bits"], r["GAR (%)"],
                style, linewidth=2, markersize=4, label=label)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'Exp 5: SSTM Method Comparison (G={G}, StableCTM)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)
    save_path = os.path.join(output_dir, "exp5_sstm_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    return results


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
        print("WARNING: using random model")
    model = model.to(device)
    model.set_beta(32)

    # Extract codes
    print("\nExtracting training codes...")
    train_binary, train_hash, train_labels = extract_codes_with_embed(
        model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    flip_prob_mean = float(flip_rate.mean())
    print(f"Per-bit flip rate: mean={flip_prob_mean*100:.2f}%")

    print("\nExtracting test codes...")
    test_binary, test_hash, test_labels = extract_codes_with_embed(
        model, test_loader, device)
    print(f"Test: binary={test_binary.shape}, hash={test_hash.shape}")

    # CTM (used across experiments)
    ctm_stable = StableCTM(hash_dim=1024, G=G,
                            flip_rate=flip_rate, stable_ratio=STABLE_RATIO)

    all_results = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "G": G, "flip_prob_mean": round(flip_prob_mean, 4)}

    # Run experiments
    all_results["exp1_rs_vs_bch"] = run_exp1_rs_vs_bch(
        test_binary, test_labels, ctm_stable, OUTPUT_DIR)

    all_results["exp2_roc"] = run_exp2_roc(
        test_binary, test_labels, ctm_stable, OUTPUT_DIR)

    all_results["exp3_ctm_analysis"] = run_exp3_ctm_analysis(
        train_binary, train_labels,
        test_binary, test_labels,
        flip_rate, OUTPUT_DIR)

    all_results["exp4_ablation"] = run_exp4_ablation(
        test_binary, test_hash, test_labels,
        flip_rate, OUTPUT_DIR)

    all_results["exp5_sstm_comparison"] = run_exp5_sstm_comparison(
        test_binary, test_hash, test_labels,
        ctm_stable, flip_prob_mean, OUTPUT_DIR)

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved: {json_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    r2 = all_results.get("exp2_roc", {})
    print(f"Model EER (Stolen Key):  {r2.get('EER_stolen_key (%)', 'N/A')}%")
    print(f"Model EER (Unknown Key): {r2.get('EER_unknown_key (%)', 'N/A')}%")
    print(f"Genuine flip rate:       {r2.get('genuine_flip_rate (%)', 'N/A')}%")

    print("\nG-S inflection (GAR=50%):")
    for name, r in all_results.get("exp5_sstm_comparison", {}).items():
        if not isinstance(r, dict) or "k_bits" not in r:
            continue
        inflect = [(k, g) for k, g in zip(r["k_bits"], r["GAR (%)"]) if g >= 50]
        if inflect:
            print(f"  {name:<25}: k={inflect[-1][0]} bits (GAR={inflect[-1][1]:.1f}%)")
        else:
            print(f"  {name:<25}: GAR never reaches 50%")

    print("\nGenerated figures:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith(".png"):
            print(f"  {OUTPUT_DIR}/{fname}")


if __name__ == "__main__":
    main()
