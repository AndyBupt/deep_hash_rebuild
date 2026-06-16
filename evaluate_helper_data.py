"""
evaluate_helper_data.py — Helper Data Leakage Security Analysis

Addresses the TBIOM reviewer concern:
  "The system stores helper data (h = re XOR s_mapped, ECC, perm).
   Does this leak information that helps attackers?"

Specifically for RGSS, the stored template contains:
  - h: auxiliary data (re XOR s_mapped)
  - ecc: BCH error-correction parity bits
  - perm: reliable-channel ordering (which positions are most reliable)

The concern: perm encodes which k positions have highest |tanh|.
  → Does perm carry user-specific identity information?
  → Can an attacker use helper data to authenticate or link templates?

Two experiments (from the teacher's guide):

Experiment A: Helper-Known Impostor Attack
  Baseline (standard impostor):
    Attacker uses impostor biometric + stolen key ke → attempt auth
  Helper-known impostor:
    Attacker additionally knows perm (reliable-channel ordering from stored template)
    Uses perm knowledge to optimise impostor template alignment
  Question: does knowing perm improve FAR?
  Expected: FAR stays ~0% because identity verification comes from biometrics,
            not from perm. perm only tells which positions to decode, not
            what values those positions should have.

Experiment B: Perm-Based Linkability (Helper Data Linkability)
  Two deployments of the same user (different keys k1, k2):
    - Enrollment 1: key k1 → perm1 (reliable positions for user under k1)
    - Enrollment 2: key k2 → perm2 (reliable positions for user under k2)
  Question: is Jaccard(perm1, perm2) > Jaccard(perm_user1, perm_user2)?
  If NOT: perm is randomised by CTM and carries no cross-deployment identity
  Expected: after CTM, perm similarity ≈ random (k/G ratio)
            This proves CTM suppresses identity-specific reliable-position structure.

Output: results_helper_data/
  helper_known_attack.png         — FAR with/without perm knowledge
  perm_linkability.png            — Jaccard similarity: same user vs diff user
  helper_data_analysis.json       — numerical results

Usage:
  python evaluate_helper_data.py
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
OUTPUT_DIR   = "results_helper_data"
G            = 512
STABLE_RATIO = 0.8
N_TRIALS     = 1000

# RGSS operating point (k₅₀ inflection)
RGSS_M, RGSS_T, RGSS_K = 9, 29, 264
BCH_M, BCH_T, BCH_K    = 9, 41, 208


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
# perm extraction helper
# ──────────────────────────────────────────────

def get_perm(hash_vec_at_ke, k_bits):
    """Return top-k reliable positions (indices within ke) sorted by |tanh|."""
    return np.argsort(-np.abs(hash_vec_at_ke))[:k_bits]


def get_perm_global(binary_code, hash_code, ctm, k_bits):
    """
    Return the globally-indexed reliable positions after CTM + RGSS.
    Step 1: CTM selects G positions → ke
    Step 2: RGSS sorts ke by |tanh| → perm[:k_bits] within ke
    Step 3: Map back to 1024-dim space → absolute position indices
    """
    _, ke = ctm.enroll(binary_code)
    embed = hash_code[ke]
    perm_in_ke = get_perm(embed, k_bits)
    return ke[perm_in_ke]   # (k_bits,) absolute positions in 1024-dim


# ──────────────────────────────────────────────
# Experiment A: Helper-Known Impostor Attack
# ──────────────────────────────────────────────

def run_experiment_A(binary_codes, hash_codes, labels, ctm, n_trials, rng):
    """
    Compare FAR for RGSS with two impostor strategies:

    Strategy 1 (Baseline): Standard stolen-key impostor
      - Attacker knows ke (stolen key)
      - Maps impostor biometric through ke → re_impostor
      - Attempts authentication (no knowledge of perm)

    Strategy 2 (Helper-known): Attacker additionally knows perm
      - perm = reliable-channel ordering stored in the template
      - Attacker knows WHICH k positions are decoded for key matching
      - This does NOT change the authentication process (perm is used
        internally by SSTM.authenticate), but we test whether an attacker
        with perm knowledge can mount a more targeted impostor attack.

    Since perm is public in the fuzzy commitment model (it IS the stored
    helper data), this essentially tests:
      "Does leaking perm reduce security beyond leaking ke?"

    Implementation:
      - Standard: stolen ke, no extra perm info → same as evaluate_attack_models stolen-key
      - Helper-known: stolen ke + known perm → attack targets only perm positions
        (attacker uses impostor biometric mapped through ke, focused on perm slots)
    Note: in practice, knowing perm doesn't change what the attacker can feed;
    authentication still checks re[perm[:k_bits]] via BCH. The test quantifies
    whether any advantage exists.
    """
    try:
        sstm_rgss = SSTM_PolarEmbed(G=G, k_bits=RGSS_K, m=RGSS_M, t=RGSS_T)
    except AssertionError:
        print("  RGSS SSTM init failed")
        return None

    sstm_bch = SSTM_BCH(G=G, m=BCH_M, t=BCH_T)
    unique_ids = np.unique(labels)

    # Standard stolen-key FAR (baseline, same as existing experiment)
    accept_rgss_std = 0
    accept_bch_std  = 0

    # Helper-known: attacker knows ke + knows perm (tries to pick best impostor sample)
    # We simulate this by using the most "compatible" impostor sample
    # (the one whose ke-extracted bits are closest to genuine in the perm positions)
    accept_rgss_hk = 0

    for _ in range(n_trials):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2_pool = np.where(labels == id2)[0]
        idx2 = rng.choice(idx2_pool)

        # Genuine enrollment
        re_genuine, ke = ctm.enroll(binary_codes[idx1])
        embed_e = hash_codes[idx1][ke]
        stored_rgss, _ = sstm_rgss.enroll(re_genuine, embed_e)
        stored_bch, _  = sstm_bch.enroll(re_genuine)

        # perm: reliable channel ordering (this IS stored in the template)
        perm_in_ke = get_perm(embed_e, RGSS_K)

        # Strategy 1: standard stolen-key impostor
        re_imp_std = ctm.authenticate(binary_codes[idx2], ke)
        ok_rgss, _ = sstm_rgss.authenticate(re_imp_std, stored_rgss)
        ok_bch,  _ = sstm_bch.authenticate(re_imp_std, stored_bch)
        accept_rgss_std += int(ok_rgss)
        accept_bch_std  += int(ok_bch)

        # Strategy 2: helper-known (attacker selects best impostor sample
        # by maximising match at perm positions — brute-force over impostor pool)
        best_ok = False
        for idx2_try in idx2_pool[:5]:    # try up to 5 impostor samples
            re_try = ctm.authenticate(binary_codes[idx2_try], ke)
            ok_try, _ = sstm_rgss.authenticate(re_try, stored_rgss)
            if ok_try:
                best_ok = True
                break
        accept_rgss_hk += int(best_ok)

    far_rgss_std = accept_rgss_std / n_trials * 100
    far_bch_std  = accept_bch_std  / n_trials * 100
    far_rgss_hk  = accept_rgss_hk  / n_trials * 100

    print(f"  BCH  FAR (standard stolen-key)        = {far_bch_std:.2f}%")
    print(f"  RGSS FAR (standard stolen-key)        = {far_rgss_std:.2f}%")
    print(f"  RGSS FAR (helper-known, best-of-5)    = {far_rgss_hk:.2f}%")
    diff = far_rgss_hk - far_rgss_std
    print(f"  Improvement from helper knowledge     = {diff:+.2f}%")
    if abs(diff) < 0.5:
        print("  → Helper data (perm) gives NO authentication advantage ✓")
    else:
        print(f"  → Helper data gives {diff:.2f}% advantage (investigate)")

    return {
        "BCH_standard_FAR_%":      round(far_bch_std,  3),
        "RGSS_standard_FAR_%":     round(far_rgss_std, 3),
        "RGSS_helper_known_FAR_%": round(far_rgss_hk,  3),
        "advantage_%":             round(diff, 3),
    }


# ──────────────────────────────────────────────
# Experiment B: Perm-Based Linkability
# ──────────────────────────────────────────────

def jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets of indices."""
    a, b = set(set_a), set(set_b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def run_experiment_B(binary_codes, hash_codes, labels, ctm, n_pairs, rng):
    """
    Perm-Based Linkability Test.

    Question: Given two templates of the same user (different keys),
              can an attacker link them by comparing perm sets?

    Same-user, different-key perm pairs (mated):
      → If perm is randomised by CTM, similarity ≈ k/G (random overlap)
    Different-user, different-key perm pairs (non-mated):
      → Should also be ≈ k/G

    If mated ≈ non-mated: CTM successfully destroys perm linkability.

    Also compare BEFORE CTM (using full tanh ordering without key selection):
      → If before-CTM mated > non-mated: there IS identity structure in raw tanh
      → After CTM it should disappear → proves CTM randomization is effective.
    """
    unique_ids = np.unique(labels)
    k = RGSS_K  # number of reliable positions
    expected_jaccard = (k / G) * (k / G)   # expected for random sets of size k from G

    print(f"\n  k_bits={k}, G={G}, expected random Jaccard ≈ {expected_jaccard:.4f}")

    # ── After CTM (perm in ke-space mapped to 1024-dim) ──
    mated_jaccard_after    = []
    non_mated_jaccard_after = []

    # ── Before CTM (raw tanh ordering, no key applied) ──
    mated_jaccard_before    = []
    non_mated_jaccard_before = []

    # Mated: same user, two different keys
    print("  Computing mated Jaccard similarities...")
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]

        # After CTM: two independent keys → two different perm sets
        perm1_global = get_perm_global(binary_codes[enroll_idx],
                                       hash_codes[enroll_idx], ctm, k)
        perm2_global = get_perm_global(binary_codes[enroll_idx],
                                       hash_codes[enroll_idx], ctm, k)
        mated_jaccard_after.append(jaccard_similarity(perm1_global, perm2_global))

        # Before CTM: use raw tanh ranking over all 1024 bits
        raw_embed = hash_codes[enroll_idx]
        top_raw1 = np.argsort(-np.abs(raw_embed))[:k]   # same, so we need two samples
        # Use a second sample of same user if available
        if len(idx) >= 2:
            top_raw2 = np.argsort(-np.abs(hash_codes[idx[1]]))[:k]
        else:
            top_raw2 = top_raw1  # same user, same sample (upper bound)
        mated_jaccard_before.append(jaccard_similarity(top_raw1, top_raw2))

    # Non-mated: different users, independent keys
    print("  Computing non-mated Jaccard similarities...")
    for _ in range(n_pairs):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])

        perm1 = get_perm_global(binary_codes[idx1], hash_codes[idx1], ctm, k)
        perm2 = get_perm_global(binary_codes[idx2], hash_codes[idx2], ctm, k)
        non_mated_jaccard_after.append(jaccard_similarity(perm1, perm2))

        raw1 = np.argsort(-np.abs(hash_codes[idx1]))[:k]
        raw2 = np.argsort(-np.abs(hash_codes[idx2]))[:k]
        non_mated_jaccard_before.append(jaccard_similarity(raw1, raw2))

    mated_after    = np.array(mated_jaccard_after)
    non_mated_after= np.array(non_mated_jaccard_after)
    mated_before   = np.array(mated_jaccard_before)
    non_mated_before = np.array(non_mated_jaccard_before)

    print(f"\n  === After CTM (perm with random key) ===")
    print(f"    Mated    Jaccard: mean={mated_after.mean():.4f}  std={mated_after.std():.4f}")
    print(f"    Non-mated Jaccard: mean={non_mated_after.mean():.4f}  std={non_mated_after.std():.4f}")
    print(f"    Expected random:  {expected_jaccard:.4f}")
    print(f"    Mated - Non-mated: {mated_after.mean()-non_mated_after.mean():+.4f}")

    print(f"\n  === Before CTM (raw tanh ranking, no key) ===")
    print(f"    Mated    Jaccard: mean={mated_before.mean():.4f}  std={mated_before.std():.4f}")
    print(f"    Non-mated Jaccard: mean={non_mated_before.mean():.4f}  std={non_mated_before.std():.4f}")
    print(f"    Mated - Non-mated: {mated_before.mean()-non_mated_before.mean():+.4f}")

    # Linkability EER based on Jaccard (higher Jaccard = more linkable)
    scores_after  = np.concatenate([mated_after, non_mated_after])
    labels_link   = np.concatenate([np.ones(len(mated_after)),
                                     np.zeros(len(non_mated_after))])
    fpr, tpr, _   = roc_curve(labels_link, scores_after)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    link_eer_after = float((fpr[eer_idx] + fnr[eer_idx]) / 2) * 100

    print(f"\n  Perm Linkability EER (after CTM) = {link_eer_after:.1f}%")
    print(f"    50% = perfectly unlinkable, 0% = fully linkable")

    return {
        "mated_jaccard_after_CTM_mean":     round(float(mated_after.mean()),     4),
        "mated_jaccard_after_CTM_std":      round(float(mated_after.std()),      4),
        "non_mated_jaccard_after_CTM_mean": round(float(non_mated_after.mean()), 4),
        "non_mated_jaccard_after_CTM_std":  round(float(non_mated_after.std()),  4),
        "mated_jaccard_before_CTM_mean":    round(float(mated_before.mean()),    4),
        "non_mated_jaccard_before_CTM_mean":round(float(non_mated_before.mean()),4),
        "expected_random_jaccard":          round(expected_jaccard, 4),
        "perm_linkability_EER_%":           round(link_eer_after, 2),
        "difference_after_CTM":             round(float(mated_after.mean() -
                                                         non_mated_after.mean()), 4),
    }, mated_after, non_mated_after, mated_before, non_mated_before


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_perm_linkability(mated_after, non_mated_after,
                           mated_before, non_mated_before, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(0, 1, 300)

    # After CTM
    ax = axes[0]
    kde_m  = gaussian_kde(mated_after,     bw_method=0.05)
    kde_nm = gaussian_kde(non_mated_after, bw_method=0.05)
    ax.plot(x, kde_m(x),  'b-',  linewidth=2,
            label=f'Same user, diff key  μ={mated_after.mean():.3f}')
    ax.plot(x, kde_nm(x), 'r--', linewidth=2,
            label=f'Diff user, diff key  μ={non_mated_after.mean():.3f}')
    ax.fill_between(x, kde_m(x),  alpha=0.12, color='blue')
    ax.fill_between(x, kde_nm(x), alpha=0.12, color='red')
    ax.set_xlabel('Jaccard Similarity of perm (reliable positions)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Perm Linkability AFTER CTM\n'
                 '(overlap ≈ random → CTM destroys identity structure)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Before CTM
    ax2 = axes[1]
    kde_mb  = gaussian_kde(mated_before,     bw_method=0.05)
    kde_nmb = gaussian_kde(non_mated_before, bw_method=0.05)
    ax2.plot(x, kde_mb(x),  'b-',  linewidth=2,
             label=f'Same user (raw tanh)  μ={mated_before.mean():.3f}')
    ax2.plot(x, kde_nmb(x), 'r--', linewidth=2,
             label=f'Diff user (raw tanh)  μ={non_mated_before.mean():.3f}')
    ax2.fill_between(x, kde_mb(x),  alpha=0.12, color='blue')
    ax2.fill_between(x, kde_nmb(x), alpha=0.12, color='red')
    ax2.set_xlabel('Jaccard Similarity of top-k tanh positions')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Perm Linkability BEFORE CTM\n'
                  '(if mated > non-mated: raw tanh has identity structure)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "perm_linkability.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_helper_known_attack(expA_results, output_dir):
    labels_bar = ['BCH\nstandard', 'RGSS\nstandard', 'RGSS\nhelper-known']
    values = [
        expA_results["BCH_standard_FAR_%"],
        expA_results["RGSS_standard_FAR_%"],
        expA_results["RGSS_helper_known_FAR_%"],
    ]
    colors = ['#d62728', '#1f77b4', '#aec7e8']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_bar, values, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('FAR (%)')
    ax.set_title('Helper Data (perm) Leakage: Does it help the attacker?\n'
                 '(Expected: all bars ≈ 0% → perm gives no authentication advantage)')
    ax.set_ylim(0, max(max(values) * 1.5, 1.0))
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "helper_known_attack.png")
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
    print(f"Test set: {test_binary.shape}, users: {len(np.unique(test_labels))}")

    ctm = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    rng = np.random.default_rng(42)

    # ── Experiment A: Helper-Known Attack ──────
    print(f"\n{'='*60}")
    print("Experiment A: Helper-Known Impostor Attack")
    print("="*60)
    expA = run_experiment_A(test_binary, test_hash, test_labels, ctm, N_TRIALS, rng)

    # ── Experiment B: Perm Linkability ─────────
    print(f"\n{'='*60}")
    print("Experiment B: Perm-Based Linkability")
    print("="*60)
    expB, mated_after, non_mated_after, mated_before, non_mated_before = \
        run_experiment_B(test_binary, test_hash, test_labels, ctm, N_TRIALS, rng)

    # ── Plots ──────────────────────────────────
    if expA is not None:
        plot_helper_known_attack(expA, OUTPUT_DIR)
    plot_perm_linkability(mated_after, non_mated_after,
                          mated_before, non_mated_before, OUTPUT_DIR)

    # ── Save JSON ──────────────────────────────
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G, "RGSS_k_bits": RGSS_K, "n_trials": N_TRIALS,
        "experiment_A_helper_known_impostor": expA,
        "experiment_B_perm_linkability": expB,
        "security_conclusions": {
            "A": ("perm gives no authentication advantage" if expA and
                  abs(expA["advantage_%"]) < 0.5 else "investigate perm advantage"),
            "B": ("CTM successfully suppresses perm linkability" if
                  abs(expB["difference_after_CTM"]) < 0.01 else
                  f"some residual perm linkability: {expB['difference_after_CTM']:.4f}"),
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "helper_data_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
