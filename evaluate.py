"""
Evaluation Script
Corresponds to Paper Section V and Section VI

Metrics:
  - EER (Equal Error Rate)
  - GAR@FAR (Genuine Accept Rate at given False Accept Rate)
  - ROC Curve
  - Genuine/Impostor Hamming Distance Distribution
  - G-S Curve (GAR vs. Security Level in bits)
  - Comparison: Baseline CTM vs. Improved StableCTM
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import CTM, StableCTM
from sstm import SSTM


def extract_all_binary_codes(model, loader, device):
    """
    Extract binary hash codes and labels from all test samples.

    Returns:
        codes: (N, hash_dim) numpy array, values in {-1, +1}
        labels: (N,) numpy array
    """
    model.eval()
    all_codes = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, _, binary = model(imgs)
            all_codes.append(binary.cpu().numpy())
            all_labels.append(labels.numpy())

    codes = np.vstack(all_codes)
    labels = np.concatenate(all_labels)
    return codes, labels


def compute_genuine_impostor_distances(codes, labels, ctm, scenario="unknown_key"):
    """
    Compute Hamming distances for all genuine pairs and impostor pairs.

    Corresponds to Paper Section V-B, two scenarios:
      unknown_key: impostor uses a random key + own biometrics
      stolen_key:  impostor uses the genuine user's key + own biometrics

    Args:
        scenario: "unknown_key" or "stolen_key"
    """
    genuine_dists = []
    impostor_dists = []
    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)

    # Genuine pairs: enroll with first sample, authenticate with the rest
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(codes[idx[0]])
        for i in idx[1:]:
            rp = ctm.authenticate(codes[i], ke)
            d = ctm.hamming_distance(re, rp)
            genuine_dists.append(d / ctm.G)

    n_impostors = min(len(genuine_dists) * 5, 2000)

    if scenario == "unknown_key":
        # Unknown key: impostor uses random indices for bit selection
        for _ in range(n_impostors):
            id1, id2 = rng.choice(unique_ids, size=2, replace=False)
            idx1 = rng.choice(np.where(labels == id1)[0])
            idx2 = rng.choice(np.where(labels == id2)[0])
            re, ke_genuine = ctm.enroll(codes[idx1], seed=int(idx1))
            _, ke_random = ctm.enroll(codes[idx2], seed=int(rng.integers(0, 99999)))
            rp = ctm.authenticate(codes[idx2], ke_random)
            d = ctm.hamming_distance(re, rp)
            impostor_dists.append(d / ctm.G)

    elif scenario == "stolen_key":
        # Stolen key: impostor knows the genuine user's key but uses own biometrics
        for _ in range(n_impostors):
            id1, id2 = rng.choice(unique_ids, size=2, replace=False)
            idx1 = rng.choice(np.where(labels == id1)[0])
            idx2 = rng.choice(np.where(labels == id2)[0])
            re, ke_genuine = ctm.enroll(codes[idx1])
            rp = ctm.authenticate(codes[idx2], ke_genuine)
            d = ctm.hamming_distance(re, rp)
            impostor_dists.append(d / ctm.G)

    else:
        raise ValueError(f"scenario must be 'unknown_key' or 'stolen_key', got: {scenario}")

    return np.array(genuine_dists), np.array(impostor_dists)


def compute_eer(genuine_dists, impostor_dists):
    """Compute Equal Error Rate (EER)."""
    scores = np.concatenate([-genuine_dists, -impostor_dists])
    y_true = np.concatenate([np.ones(len(genuine_dists)),
                              np.zeros(len(impostor_dists))])

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr

    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return eer, fpr, tpr, thresholds


def compute_gar_at_far(fpr, tpr, target_far=0.005):
    """Compute GAR at a given FAR."""
    idx = np.searchsorted(fpr, target_far)
    if idx >= len(tpr):
        return tpr[-1]
    return tpr[idx]


def plot_distributions(genuine_dists, impostor_uk_dists, impostor_sk_dists,
                       G, save_path=None):
    """
    Plot Genuine/Impostor Hamming distance distributions (Paper Fig.4).
    Shows both unknown key and stolen key impostor distributions.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 1, 300)

    g_mean, g_std = genuine_dists.mean(), genuine_dists.std()
    uk_mean, uk_std = impostor_uk_dists.mean(), impostor_uk_dists.std()
    sk_mean, sk_std = impostor_sk_dists.mean(), impostor_sk_dists.std()

    ax.plot(x, norm.pdf(x, g_mean, g_std),  'b-', linewidth=2, label='Genuine')
    ax.plot(x, norm.pdf(x, uk_mean, uk_std), 'r-', linewidth=2,
            label='Impostor (Unknown Key)')
    ax.plot(x, norm.pdf(x, sk_mean, sk_std), 'g--', linewidth=2,
            label='Impostor (Stolen Key)')
    ax.fill_between(x, norm.pdf(x, g_mean, g_std),  alpha=0.15, color='blue')
    ax.fill_between(x, norm.pdf(x, uk_mean, uk_std), alpha=0.15, color='red')
    ax.fill_between(x, norm.pdf(x, sk_mean, sk_std), alpha=0.15, color='green')

    ax.set_xlabel('Normalized Hamming Distance')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Genuine/Impostor Distribution (G={G})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved: {save_path}")
    plt.show()


def plot_roc(fpr_uk, tpr_uk, eer_uk, fpr_sk, tpr_sk, eer_sk,
             G, save_path=None):
    """
    Plot ROC curves (Paper Fig.7/8).
    Shows both unknown key and stolen key scenarios.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    auc_uk = auc(fpr_uk, tpr_uk)
    auc_sk = auc(fpr_sk, tpr_sk)
    ax.plot(fpr_uk, tpr_uk, 'b-', linewidth=2,
            label=f'Unknown Key (AUC={auc_uk:.3f}, EER={eer_uk*100:.2f}%)')
    ax.plot(fpr_sk, tpr_sk, 'r--', linewidth=2,
            label=f'Stolen Key  (AUC={auc_sk:.3f}, EER={eer_sk*100:.2f}%)')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FAR (False Accept Rate)')
    ax.set_ylabel('GAR (Genuine Accept Rate)')
    ax.set_title(f'ROC Curve (G={G})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved: {save_path}")
    plt.show()


def _compute_gar_with_sstm(codes, labels, ctm, sstm, scenario):
    """Helper: compute GAR using SSTM for a given scenario."""
    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)
    pass_count = total = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(codes[idx[0]])
        stored_hash, _ = sstm.enroll(re)
        for i in idx[1:]:
            if scenario == "unknown_key":
                _, ke_rand = ctm.enroll(codes[i],
                                        seed=int(rng.integers(0, 99999)))
                rp = ctm.authenticate(codes[i], ke_rand)
            else:
                rp = ctm.authenticate(codes[i], ke)
            is_genuine, _ = sstm.authenticate(rp, stored_hash)
            pass_count += int(is_genuine)
            total += 1
    return pass_count / total if total > 0 else 0


def plot_gs_curve_comparison(codes, labels, ctm_baseline, ctm_improved,
                              K_values, G, save_path=None):
    """
    Plot G-S curves comparing Baseline CTM vs. Improved StableCTM.
    K range should cover both inflection points.

    Args:
        ctm_baseline: original CTM (random bit selection)
        ctm_improved: StableCTM (stable bit selection), or None to skip
        K_values: list of RS symbol counts, security = K*8 bits
    """
    k_bits_list = [K * 8 for K in K_values]
    gars_baseline = []
    gars_improved = [] if ctm_improved is not None else None

    print(f"\nComputing Baseline G-S curve (G={G})...")
    for K in K_values:
        if G % 8 != 0 or G // 8 <= K:
            gars_baseline.append(0)
            if gars_improved is not None:
                gars_improved.append(0)
            continue
        sstm = SSTM(G=G, K=K)
        gar = _compute_gar_with_sstm(codes, labels, ctm_baseline, sstm,
                                      scenario="stolen_key")
        gars_baseline.append(gar * 100)
        print(f"  Baseline  K={K:3d} (k={K*8:4d} bits): GAR={gar*100:.1f}%")

    if ctm_improved is not None:
        print(f"\nComputing Improved G-S curve (StableCTM)...")
        for K in K_values:
            if G % 8 != 0 or G // 8 <= K:
                gars_improved.append(0)
                continue
            sstm = SSTM(G=G, K=K)
            gar = _compute_gar_with_sstm(codes, labels, ctm_improved, sstm,
                                          scenario="stolen_key")
            gars_improved.append(gar * 100)
            print(f"  Improved  K={K:3d} (k={K*8:4d} bits): GAR={gar*100:.1f}%")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(k_bits_list, gars_baseline, 'r-o', linewidth=2, markersize=4,
            label='Baseline (Random Bit Selection)')
    if gars_improved is not None:
        ax.plot(k_bits_list, gars_improved, 'b-s', linewidth=2, markersize=4,
                label='Improved (Stable Bit Selection)')

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    title = f'G-S Curve: Baseline vs. Improved CTM (G={G} bits)'
    if ctm_improved is None:
        title = f'G-S Curve: Baseline CTM (G={G} bits)'
    ax.set_title(title)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    # Annotate inflection points
    for label_str, gars, color in [('Baseline', gars_baseline, 'red'),
                                    ('Improved', gars_improved or [], 'blue')]:
        for i, (k, gar) in enumerate(zip(k_bits_list, gars)):
            if gar < 50 and i > 0 and gars[i-1] >= 50:
                ax.axvline(x=k, color=color, linestyle=':', alpha=0.6)
                offset = 15 if label_str == 'Baseline' else -90
                ax.annotate(f'{label_str}\nk={k}b',
                            xy=(k, 50), xytext=(k + offset, 62),
                            arrowprops=dict(arrowstyle='->', color=color),
                            color=color, fontsize=8)
                break

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"G-S comparison curve saved: {save_path}")
    plt.show()
    return k_bits_list, gars_baseline, gars_improved


def run_evaluation(model_path=None, G_values=None, data_root="fingerprints",
                   db_names=None, output_dir="results",
                   run_comparison=True):
    """
    Full evaluation pipeline.

    Args:
        model_path:      path to model weights; None = random init (for testing)
        G_values:        list of bit counts to evaluate, e.g. [128, 256, 512]
        data_root:       data root directory
        db_names:        list of DB names to use
        output_dir:      directory to save results
        run_comparison:  if True, run Baseline vs. Improved G-S comparison
    """
    if G_values is None:
        G_values = [128, 256, 512]
    if db_names is None:
        db_names = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_loader, test_loader, num_classes = build_dataloaders(
        data_root, db_names, train_ratio=0.7, batch_size=8
    )

    # Load model
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024, pretrained=False)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    else:
        print("Using randomly initialized model (for pipeline testing only)")
    model = model.to(device)
    model.set_beta(32)

    # Extract binary codes
    print("\nExtracting test set binary codes...")
    codes, labels = extract_all_binary_codes(model, test_loader, device)
    print(f"Extracted: {codes.shape}")

    # --- Part 1: Standard evaluation (Baseline CTM) ---
    results = {}
    for G in G_values:
        print(f"\n{'='*45}")
        print(f"Evaluating G={G} bits")
        ctm = CTM(hash_dim=1024, G=G)

        genuine_dists, imp_uk = compute_genuine_impostor_distances(
            codes, labels, ctm, scenario="unknown_key"
        )
        _, imp_sk = compute_genuine_impostor_distances(
            codes, labels, ctm, scenario="stolen_key"
        )

        print(f"  Genuine:       mean={genuine_dists.mean():.3f}, "
              f"std={genuine_dists.std():.3f}")
        print(f"  Impostor(UK):  mean={imp_uk.mean():.3f}, "
              f"std={imp_uk.std():.3f}")
        print(f"  Impostor(SK):  mean={imp_sk.mean():.3f}, "
              f"std={imp_sk.std():.3f}")

        eer_uk, fpr_uk, tpr_uk, _ = compute_eer(genuine_dists, imp_uk)
        eer_sk, fpr_sk, tpr_sk, _ = compute_eer(genuine_dists, imp_sk)
        gar_uk = compute_gar_at_far(fpr_uk, tpr_uk, target_far=0.005)
        gar_sk = compute_gar_at_far(fpr_sk, tpr_sk, target_far=0.005)

        print(f"  EER (Unknown Key): {eer_uk*100:.2f}%  "
              f"GAR@FAR=0.5%: {gar_uk*100:.2f}%")
        print(f"  EER (Stolen Key):  {eer_sk*100:.2f}%  "
              f"GAR@FAR=0.5%: {gar_sk*100:.2f}%")

        results[G] = {
            'eer_uk': eer_uk, 'gar_uk': gar_uk,
            'eer_sk': eer_sk, 'gar_sk': gar_sk,
            'genuine_dists': genuine_dists,
            'imp_uk': imp_uk, 'imp_sk': imp_sk,
            'fpr_uk': fpr_uk, 'tpr_uk': tpr_uk,
            'fpr_sk': fpr_sk, 'tpr_sk': tpr_sk,
        }

        plot_distributions(genuine_dists, imp_uk, imp_sk, G,
                           save_path=os.path.join(output_dir, f"dist_G{G}.png"))
        plot_roc(fpr_uk, tpr_uk, eer_uk, fpr_sk, tpr_sk, eer_sk, G,
                 save_path=os.path.join(output_dir, f"roc_G{G}.png"))

    # Summary table (Paper Table I format)
    print("\n" + "="*60)
    print("Summary Results (Paper Table I format):")
    print(f"{'G':>6} | {'EER_UK(%)':>10} | {'EER_SK(%)':>10} | "
          f"{'GAR_UK@0.5%':>12} | {'GAR_SK@0.5%':>12}")
    print("-" * 60)
    for G in G_values:
        r = results[G]
        print(f"{G:>6} | {r['eer_uk']*100:>10.2f} | {r['eer_sk']*100:>10.2f} | "
              f"{r['gar_uk']*100:>12.2f} | {r['gar_sk']*100:>12.2f}")

    # --- Part 2: G-S Curve Comparison (Baseline vs. Improved) ---
    best_G = max([G for G in G_values if G % 8 == 0], default=G_values[-1])
    N = best_G // 8  # = 64 for G=512

    # K range: step=2, from 7 to N-1, covers both inflection points
    # Baseline inflection ~K=39 (k≈312 bits)
    # Improved inflection ~K=55 (k≈440 bits)
    K_values = list(range(7, N, 2))

    if run_comparison:
        # Build flip rate from training set for StableCTM
        print("\nExtracting training set binary codes (for flip rate)...")
        train_codes, train_labels = extract_all_binary_codes(model, train_loader, device)
        flip_rate = StableCTM.compute_flip_rate(train_codes, train_labels)
        print(f"Flip rate computed: mean={flip_rate.mean()*100:.1f}%")

        ctm_baseline = CTM(hash_dim=1024, G=best_G)
        ctm_improved = StableCTM(hash_dim=1024, G=best_G,
                                  flip_rate=flip_rate, stable_ratio=0.3)

        print(f"\nPlotting G-S comparison (G={best_G}, K from 7 to {K_values[-1]})...")
        plot_gs_curve_comparison(
            codes, labels,
            ctm_baseline=ctm_baseline,
            ctm_improved=ctm_improved,
            K_values=K_values,
            G=best_G,
            save_path=os.path.join(output_dir, f"gs_comparison_G{best_G}.png")
        )
    else:
        # Baseline only
        ctm_baseline = CTM(hash_dim=1024, G=best_G)
        print(f"\nPlotting Baseline G-S curve (G={best_G}, K from 7 to {K_values[-1]})...")
        plot_gs_curve_comparison(
            codes, labels,
            ctm_baseline=ctm_baseline,
            ctm_improved=None,
            K_values=K_values,
            G=best_G,
            save_path=os.path.join(output_dir, f"gs_curve_G{best_G}_full.png")
        )

    return results


if __name__ == "__main__":
    model_path = "checkpoints/final_model.pth"
    run_evaluation(
        model_path=model_path if os.path.exists(model_path) else None,
        G_values=[128, 256, 512],
        output_dir="results",
        run_comparison=True  # Set to False to skip StableCTM comparison
    )
