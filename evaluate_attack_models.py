"""
evaluate_attack_models.py — Security Evaluation under Three Attack Models

Extends the existing Unknown-key and Stolen-key attacks with
Partial Key Leakage attack, as suggested by the advisor.

Attack Models:
  1. Unknown-key attack  (leakage=0%):  attacker knows NOTHING about user's key ke
  2. Partial key leakage (leakage=p%):  attacker knows p% of user's key ke indices
  3. Stolen-key attack   (leakage=100%): attacker knows ke completely

The CTM key ke is the set of G bit-position indices (out of 1024) used to
construct the cancelable template.  More key leakage → easier to align the
impostor's biometric into the victim's feature space.

Compares how FAR rises with leakage level for BCH vs RGSS.

Output: results_attack_models/
  attack_far_curve.png         — FAR vs leakage for BCH and RGSS
  attack_models_results.json   — Numerical results

Usage:
  python evaluate_attack_models.py
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
OUTPUT_DIR   = "results_attack_models"
G            = 512
STABLE_RATIO = 0.8
N_TRIALS     = 1000   # impostor attempts per leakage level

# Leakage levels: 0% = unknown-key, 100% = stolen-key
LEAKAGE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# Fixed BCH/RGSS parameters (use the k₅₀ inflection operating point)
BCH_M, BCH_T, BCH_K   = 9, 41, 208   # BCH k=208 bits (GAR=50% inflection)
RGSS_M, RGSS_T, RGSS_K = 9, 29, 264   # RGSS k=264 bits (GAR=50% inflection)


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
# Partial key construction
# ──────────────────────────────────────────────

def make_leaked_key(ke_genuine, leakage_p, hash_dim, rng):
    """
    Construct a partially-leaked key.

    leakage_p fraction of ke_genuine is correct;
    the rest is replaced with random indices from the remaining pool.

    Args:
        ke_genuine: (G,) true key (sorted bit-position indices)
        leakage_p:  float in [0, 1]
        hash_dim:   total number of hash bits (1024)
        rng:        numpy Generator

    Returns:
        ke_leaked: (G,) partially-correct key, sorted
    """
    G = len(ke_genuine)
    n_known  = int(round(leakage_p * G))
    n_random = G - n_known

    if n_random == 0:
        return ke_genuine.copy()  # full leakage = stolen key

    # Known portion: first n_known entries of ke_genuine
    known_indices = ke_genuine[:n_known].copy()

    # Random portion: drawn from indices NOT in known_indices
    all_indices = np.arange(hash_dim)
    remaining   = np.setdiff1d(all_indices, known_indices)
    random_indices = rng.choice(remaining, size=n_random, replace=False)

    ke_leaked = np.concatenate([known_indices, random_indices])
    return np.sort(ke_leaked).astype(np.int64)


# ──────────────────────────────────────────────
# FAR computation under partial key leakage
# ──────────────────────────────────────────────

def compute_far_bch(binary_codes, labels, ctm, leakage_p, m, t, n_trials, rng):
    """FAR for BCH under leakage_p key leakage."""
    sstm    = SSTM_BCH(G=G, m=m, t=t)
    unique_ids = np.unique(labels)
    accept  = 0

    for _ in range(n_trials):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)

        # Genuine enrollment
        idx1 = rng.choice(np.where(labels == id1)[0])
        re_genuine, ke_genuine = ctm.enroll(binary_codes[idx1])
        stored, _ = sstm.enroll(re_genuine)

        # Impostor uses leaked key
        ke_leaked = make_leaked_key(ke_genuine, leakage_p, ctm.hash_dim, rng)
        idx2 = rng.choice(np.where(labels == id2)[0])
        re_impostor = ctm.authenticate(binary_codes[idx2], ke_leaked)

        ok, _ = sstm.authenticate(re_impostor, stored)
        accept += int(ok)

    return accept / n_trials


def compute_far_rgss(binary_codes, hash_codes, labels, ctm,
                     leakage_p, k_bits, m, t, n_trials, rng):
    """FAR for RGSS under leakage_p key leakage."""
    try:
        sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
    except AssertionError:
        return None
    unique_ids = np.unique(labels)
    accept = 0

    for _ in range(n_trials):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)

        # Genuine enrollment (uses ke_genuine and tanh embed)
        idx1 = rng.choice(np.where(labels == id1)[0])
        re_genuine, ke_genuine = ctm.enroll(binary_codes[idx1])
        embed_genuine = hash_codes[idx1][ke_genuine]
        stored, _ = sstm.enroll(re_genuine, embed_genuine)

        # Impostor uses leaked key
        # Note: perm (reliable-channel ordering) is stored inside 'stored'
        # and is used automatically during authenticate — no extra info needed.
        ke_leaked = make_leaked_key(ke_genuine, leakage_p, ctm.hash_dim, rng)
        idx2 = rng.choice(np.where(labels == id2)[0])
        re_impostor = ctm.authenticate(binary_codes[idx2], ke_leaked)

        ok, _ = sstm.authenticate(re_impostor, stored)
        accept += int(ok)

    return accept / n_trials


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_attack_experiment(binary_codes, hash_codes, labels, ctm, output_dir):
    print("\n" + "="*60)
    print("Attack Model Evaluation: FAR vs Key Leakage")
    print(f"  BCH  operating point: k={BCH_K} bits (m={BCH_M}, t={BCH_T})")
    print(f"  RGSS operating point: k={RGSS_K} bits (m={RGSS_M}, t={RGSS_T})")
    print(f"  N_TRIALS per leakage level: {N_TRIALS}")
    print("="*60)

    rng = np.random.default_rng(42)

    far_bch  = []
    far_rgss = []

    for p in LEAKAGE_LEVELS:
        label = f"{int(p*100):3d}%"
        if p == 0.0:
            attack_name = "Unknown-key attack"
        elif p == 1.0:
            attack_name = "Stolen-key attack"
        else:
            attack_name = f"Partial key leakage ({label})"

        print(f"\n[{attack_name}]")

        bch_far = compute_far_bch(
            binary_codes, labels, ctm,
            leakage_p=p, m=BCH_M, t=BCH_T,
            n_trials=N_TRIALS, rng=rng
        )
        rgss_far = compute_far_rgss(
            binary_codes, hash_codes, labels, ctm,
            leakage_p=p, k_bits=RGSS_K, m=RGSS_M, t=RGSS_T,
            n_trials=N_TRIALS, rng=rng
        )

        far_bch.append(bch_far * 100)
        far_rgss.append(rgss_far * 100 if rgss_far is not None else 0.0)

        print(f"  BCH  FAR = {bch_far*100:.2f}%")
        print(f"  RGSS FAR = {rgss_far*100:.2f}%")

    # ── Plot ──────────────────────────────────
    leakage_pct = [p * 100 for p in LEAKAGE_LEVELS]

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(leakage_pct, far_bch,  'r-o', linewidth=2, markersize=6,
            label=f'BCH  (k={BCH_K} bits)')
    ax.plot(leakage_pct, far_rgss, 'b-s', linewidth=2, markersize=6,
            label=f'RGSS (k={RGSS_K} bits, proposed)')

    # Annotate the three distinct attack scenarios
    ax.axvline(x=0,   color='gray', linestyle=':', alpha=0.6)
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.6)
    ax.text(1,   max(far_bch+far_rgss)*0.92, 'Unknown-key\nattack',
            fontsize=8, color='gray')
    ax.text(82,  max(far_bch+far_rgss)*0.92, 'Stolen-key\nattack',
            fontsize=8, color='gray')

    ax.set_xlabel('Key Leakage Level (%)')
    ax.set_ylabel('False Accept Rate FAR (%)')
    ax.set_title(f'FAR vs Key Leakage Level\n'
                 f'BCH vs RGSS (G={G}, FVC2004, StableCTM)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, max(far_bch + far_rgss + [2]) * 1.15)
    ax.set_xlim(-3, 103)

    save_path = os.path.join(output_dir, "attack_far_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return {
        "leakage_levels (%)": [p * 100 for p in LEAKAGE_LEVELS],
        "BCH_FAR (%)":  [round(v, 3) for v in far_bch],
        "RGSS_FAR (%)": [round(v, 3) for v in far_rgss],
        "config": {
            "BCH":  {"m": BCH_M,  "t": BCH_T,  "k_bits": BCH_K},
            "RGSS": {"m": RGSS_M, "t": RGSS_T, "k_bits": RGSS_K},
            "G": G, "n_trials": N_TRIALS,
        }
    }


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

    # Compute flip_rate from training set
    print("\nExtracting training codes...")
    train_binary, _, train_labels = extract_codes(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    print(f"Training flip rate mean: {flip_rate.mean()*100:.2f}%")

    print("\nExtracting test codes...")
    test_binary, test_hash, test_labels = extract_codes(model, test_loader, device)
    print(f"Test set: {test_binary.shape}")

    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO)

    # Run attack experiment
    results = run_attack_experiment(
        test_binary, test_hash, test_labels, ctm, OUTPUT_DIR
    )

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY: FAR under Different Attack Models")
    print(f"{'Leakage':>10}  {'Attack Model':<30}  {'BCH FAR':>8}  {'RGSS FAR':>9}")
    print("-"*65)
    attack_labels = [
        "Unknown-key attack",
        "Partial leakage 10%",
        "Partial leakage 25%",
        "Partial leakage 50%",
        "Partial leakage 75%",
        "Partial leakage 90%",
        "Stolen-key attack",
    ]
    for p, alabel, bf, rf in zip(
            results["leakage_levels (%)"],
            attack_labels,
            results["BCH_FAR (%)"],
            results["RGSS_FAR (%)"]
    ):
        print(f"  {p:>6.0f}%  {alabel:<30}  {bf:>7.2f}%  {rf:>8.2f}%")

    # Save JSON
    results["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_path = os.path.join(OUTPUT_DIR, "attack_models_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
