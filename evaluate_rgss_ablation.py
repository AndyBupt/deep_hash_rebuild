"""
evaluate_rgss_ablation.py — RGSS Channel-Selection Strategy Ablation

Ablates the reliability signal used to select k "key-embedding" positions.
All variants share the same BCH back-end ECC and G=512 template.

5 strategies:
  1. Random          — random position selection (equivalent to BCH partial-fill baseline)
  2. Flip-rate       — select k positions with LOWEST training-set flip rate
  3. Tanh-confidence — select k positions with HIGHEST |tanh| output  ← proposed RGSS
  4. Worst-channel   — select k positions with LOWEST |tanh| (adversarial control)
  5. Oracle          — select k positions with LOWEST actual test-set flip rate (upper bound)

Output: results_ablation/
  ablation_gs.png        — G-S curves for all 5 strategies
  ablation_results.json

Usage:
  python evaluate_rgss_ablation.py
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
from sstm_polar_embed import SSTM_PolarEmbed


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_ablation"
G            = 512
STABLE_RATIO = 0.8


# Strategy display config: name → (strategy_key, color, linestyle+marker)
STRATEGY_STYLES = {
    "Random":          ("random",    "#7f7f7f", "--x"),
    "Flip-rate":       ("flip_rate", "#ff7f0e", "-^"),
    "Tanh-confidence": ("tanh",      "#1f77b4", "-s"),
    "Worst-channel":   ("worst",     "#d62728", "-v"),
    "Oracle":          ("oracle",    "#2ca02c", "-D"),
}


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
# Per-position test-set flip rate (oracle)
# ──────────────────────────────────────────────

def compute_per_position_flip_rate(binary_codes, labels):
    """
    Compute per-position genuine flip rate from test set genuine pairs.
    Returns array of shape (hash_dim,).
    This is the oracle signal: using ground-truth stability from test data.
    """
    unique_ids = np.unique(labels)
    flip_counts = np.zeros(binary_codes.shape[1])
    total_pairs = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        ref = (binary_codes[idx[0]] > 0).astype(np.float32)
        for i in idx[1:]:
            probe = (binary_codes[i] > 0).astype(np.float32)
            flip_counts += (ref != probe)
            total_pairs += 1
    return flip_counts / max(total_pairs, 1)


# ──────────────────────────────────────────────
# BCH parameter helpers
# ──────────────────────────────────────────────

def get_bch_params_for_g(G, t_min=None, step=1):
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
# Embed computation per strategy
# ──────────────────────────────────────────────

def make_embed(strategy, hash_vec_1024, ke, flip_rate_1024, oracle_flip_1024):
    """
    Compute the embed array (G,) passed to SSTM_PolarEmbed.enroll/authenticate.

    SSTM_PolarEmbed selects the k positions with HIGHEST |embed| as
    reliable channels. Each strategy encodes a different reliability signal.

    Args:
        strategy:          one of the strategy keys defined in STRATEGY_STYLES
        hash_vec_1024:     (1024,) tanh continuous values for this sample
        ke:                (G,) indices selected by CTM for this user
        flip_rate_1024:    (1024,) training-set per-bit flip rates
        oracle_flip_1024:  (1024,) test-set per-bit flip rates (oracle only)

    Returns:
        embed: (G,) float array, or None for random strategy
    """
    if strategy == "random":
        # No reliability guidance → random permutation inside SSTM_PolarEmbed
        return None

    elif strategy == "flip_rate":
        # Reliability ∝ (1 - training flip rate): lower flip = more stable = reliable
        return (1.0 - flip_rate_1024[ke]).astype(np.float32)

    elif strategy == "tanh":
        # Standard RGSS: |tanh| as reliability
        return hash_vec_1024[ke].astype(np.float32)

    elif strategy == "worst":
        # Adversarial control: invert |tanh| to select LEAST confident positions
        # embed = 1/(|tanh| + eps) → large |embed| when |tanh| is small
        tanh_at_ke = np.abs(hash_vec_1024[ke]).astype(np.float32)
        return (1.0 / (tanh_at_ke + 1e-6))

    elif strategy == "oracle":
        # Upper bound: reliability ∝ (1 - test-set flip rate)
        return (1.0 - oracle_flip_1024[ke]).astype(np.float32)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ──────────────────────────────────────────────
# GAR computation for one strategy × one k_bits
# ──────────────────────────────────────────────

def compute_gar_ablation(binary_codes, hash_codes, labels, ctm,
                          G, k_bits, m, t, strategy,
                          flip_rate_1024, oracle_flip_1024):
    """
    Compute GAR for a given channel-selection strategy.
    All strategies use the same SSTM_PolarEmbed with BCH(m, t).
    """
    try:
        sstm_factory = lambda: SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
    except AssertionError:
        return None

    unique_ids = np.unique(labels)
    pass_count = total = 0

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue

        re, ke = ctm.enroll(binary_codes[idx[0]])
        sstm = sstm_factory()

        # Embed for enrollment (based on enrollment sample's tanh)
        embed_e = make_embed(strategy,
                             hash_codes[idx[0]],   # tanh values of enrollment sample
                             ke,
                             flip_rate_1024,
                             oracle_flip_1024)
        try:
            stored, _ = sstm.enroll(re, embed_e)
        except Exception:
            continue

        for i in idx[1:]:
            rp = ctm.authenticate(binary_codes[i], ke)
            # For tanh strategy authentication uses probe's own tanh;
            # for other strategies the embed is not used in authenticate
            # (perm is stored in the template from enrollment)
            ok, _ = sstm.authenticate(rp, stored)
            pass_count += int(ok)
            total += 1

    return pass_count / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_ablation(binary_codes, hash_codes, labels,
                 flip_rate_1024, oracle_flip_1024, output_dir):
    print("\n" + "="*60)
    print("RGSS Channel-Selection Strategy Ablation (G=512)")
    print("="*60)

    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate_1024, stable_ratio=STABLE_RATIO)
    bch_params = get_bch_params_for_g(G)

    results = {}

    for display_name, (strategy, color, style) in STRATEGY_STYLES.items():
        print(f"\n  [{display_name}]...")
        k_bits_list, gars = [], []
        for m, t, k_bits in bch_params:
            gar = compute_gar_ablation(
                binary_codes, hash_codes, labels, ctm,
                G, k_bits, m, t, strategy,
                flip_rate_1024, oracle_flip_1024
            )
            if gar is None:
                continue
            gars.append(gar * 100)
            k_bits_list.append(k_bits)
            print(f"    t={t:3d}  k={k_bits:4d} bits  GAR={gar*100:.1f}%")

        inflect = [(k, g) for k, g in zip(k_bits_list, gars) if g >= 50]
        inflect_str = f"k={inflect[-1][0]} bits" if inflect else "N/A"
        print(f"  → GAR=50% inflection: {inflect_str}")

        results[display_name] = {
            "strategy":         strategy,
            "k_bits":           k_bits_list,
            "GAR (%)":          [round(g, 2) for g in gars],
            "inflection_k_bits": inflect[-1][0] if inflect else None,
        }

    # ── Plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    for display_name, (strategy, color, style) in STRATEGY_STYLES.items():
        r = results.get(display_name, {})
        if not r.get("k_bits"):
            continue
        marker = style[-1]
        linestyle = style[:-1]
        label = display_name
        if r.get("inflection_k_bits"):
            label += f"  (k₅₀={r['inflection_k_bits']} bits)"
        ax.plot(r["k_bits"], r["GAR (%)"],
                color=color, linestyle=linestyle, marker=marker,
                linewidth=2, markersize=4, label=label)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Key Length k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'RGSS Ablation: Channel-Selection Strategy Comparison\n'
                 f'G={G} bits, BCH back-end ECC, StableCTM (FVC2004)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    save_path = os.path.join(output_dir, "ablation_gs.png")
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
        print("WARNING: using random model weights")
    model = model.to(device)
    model.set_beta(32)

    # Extract codes
    print("\nExtracting training codes (for flip-rate strategy)...")
    train_binary, _, train_labels = extract_codes_with_embed(
        model, train_loader, device)
    flip_rate_1024 = StableCTM.compute_flip_rate(train_binary, train_labels)
    print(f"Training flip rate mean: {flip_rate_1024.mean()*100:.2f}%")

    print("\nExtracting test codes...")
    test_binary, test_hash, test_labels = extract_codes_with_embed(
        model, test_loader, device)
    print(f"Test set: {test_binary.shape}")

    # Compute oracle flip rate from test set genuine pairs
    print("\nComputing oracle (test-set) per-position flip rate...")
    oracle_flip_1024 = compute_per_position_flip_rate(test_binary, test_labels)
    print(f"Oracle flip rate mean: {oracle_flip_1024.mean()*100:.2f}%")

    # Run ablation
    results = run_ablation(test_binary, test_hash, test_labels,
                           flip_rate_1024, oracle_flip_1024, OUTPUT_DIR)

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY — RGSS Ablation")
    print(f"{'Strategy':<20}  {'GAR=50% k (bits)':>18}")
    print("-"*42)
    for name, r in results.items():
        inf = r.get("inflection_k_bits")
        print(f"  {name:<18}  {str(inf) if inf else 'N/A':>18}")

    # Save JSON
    out = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "G": G,
           "strategies": results}
    json_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
