"""
evaluate_attack_advanced.py — Advanced Attack Models for TBIOM

Extends evaluate_attack_models.py with stronger attacks:

Attack 1: Old-Template-Assisted Attack (most relevant for cancelability)
  - Attacker obtains old template T1 (under key k1) and now tries to attack
    the new template T2 (under revoked key k2).
  - Tests: does holding the old template help attack the new template?
  - Expected result: FAR stays ~0% even with old template (proves strong cancelability).

Attack 2: Reliability-Aware Attack
  - Attacker knows RGSS prefers high |tanh| bits at the population level.
  - Instead of randomly guessing unknown key positions, guesses most
    commonly-selected (population-level reliable) positions.
  - Compares FAR against naive random guessing from evaluate_attack_models.py.

Output: results_attack_advanced/
  old_template_attack.png          — FAR: standard vs old-template-assisted
  reliability_aware_attack.png     — FAR: random guessing vs reliability-aware
  attack_advanced_results.json

Usage:
  python evaluate_attack_advanced.py
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
OUTPUT_DIR   = "results_attack_advanced"
G            = 512
STABLE_RATIO = 0.8
N_TRIALS     = 1000

# SSTM operating points (GAR=50% inflection)
BCH_M, BCH_T, BCH_K    = 9, 41, 208
RGSS_M, RGSS_T, RGSS_K = 9, 29, 264

# Leakage levels for old-template-assisted attack
LEAKAGE_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]


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
# Population-level reliable positions
# ──────────────────────────────────────────────

def compute_population_reliable_positions(hash_codes, labels, ctm, top_k):
    """
    Aggregate reliable positions across all users (population level).
    For each user, compute RGSS ranking of positions within their ke.
    Return the top_k positions most frequently selected as "reliable"
    at the FULL 1024-dim level (before CTM).

    Used by reliability-aware attack: attacker guesses these positions
    are more likely to be in a victim's key.
    """
    unique_ids = np.unique(labels)
    position_counts = np.zeros(ctm.hash_dim)   # (1024,) counts

    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]
        _, ke = ctm.enroll(hash_codes[enroll_idx])
        embed = hash_codes[enroll_idx][ke]
        perm  = np.argsort(-np.abs(embed))
        # All G positions, weighted by reliability rank
        for rank, pos_in_ke in enumerate(perm):
            actual_pos = ke[pos_in_ke]
            position_counts[actual_pos] += (len(perm) - rank)  # higher weight = more reliable

    # Top-k most frequently/reliably selected positions
    top_positions = np.argsort(-position_counts)[:top_k]
    return top_positions


def make_leaked_key(ke_genuine, leakage_p, hash_dim, rng):
    """Standard partial key leakage (same as evaluate_attack_models.py)."""
    n_known  = int(round(leakage_p * len(ke_genuine)))
    n_random = len(ke_genuine) - n_known
    if n_random == 0:
        return ke_genuine.copy()
    known_indices  = ke_genuine[:n_known].copy()
    remaining      = np.setdiff1d(np.arange(hash_dim), known_indices)
    random_indices = rng.choice(remaining, size=n_random, replace=False)
    return np.sort(np.concatenate([known_indices, random_indices])).astype(np.int64)


def make_reliability_aware_key(ke_genuine, leakage_p, pop_reliable_positions,
                                hash_dim, rng):
    """
    Reliability-aware partial key leakage.
    Attacker knows leakage_p fraction of ke exactly,
    and uses population-level reliable positions to guess the rest.
    """
    n_known  = int(round(leakage_p * len(ke_genuine)))
    n_guess  = len(ke_genuine) - n_known
    if n_guess == 0:
        return ke_genuine.copy()

    known_indices = ke_genuine[:n_known].copy()

    # From population-reliable positions, pick those NOT already known
    candidates = np.setdiff1d(pop_reliable_positions, known_indices)
    if len(candidates) >= n_guess:
        guessed = candidates[:n_guess]
    else:
        # If not enough, fill with random
        extra_pool = np.setdiff1d(
            np.setdiff1d(np.arange(hash_dim), known_indices),
            candidates
        )
        extra = rng.choice(extra_pool, size=n_guess - len(candidates), replace=False)
        guessed = np.concatenate([candidates, extra])

    return np.sort(np.concatenate([known_indices, guessed])).astype(np.int64)


# ──────────────────────────────────────────────
# FAR computation helpers
# ──────────────────────────────────────────────

def compute_far_bch_with_key(binary_codes, labels, ctm, ke_fn, n_trials, rng):
    """General FAR for BCH under a custom key construction function ke_fn."""
    sstm = SSTM_BCH(G=G, m=BCH_M, t=BCH_T)
    unique_ids = np.unique(labels)
    accept = 0
    for _ in range(n_trials):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        re_genuine, ke_genuine = ctm.enroll(binary_codes[idx1])
        stored, _ = sstm.enroll(re_genuine)
        ke_attack = ke_fn(ke_genuine, rng)
        re_impostor = ctm.authenticate(binary_codes[idx2], ke_attack)
        ok, _ = sstm.authenticate(re_impostor, stored)
        accept += int(ok)
    return accept / n_trials


def compute_far_rgss_with_key(binary_codes, hash_codes, labels, ctm, ke_fn,
                               n_trials, rng):
    """General FAR for RGSS under a custom key construction function ke_fn."""
    try:
        sstm = SSTM_PolarEmbed(G=G, k_bits=RGSS_K, m=RGSS_M, t=RGSS_T)
    except AssertionError:
        return None
    unique_ids = np.unique(labels)
    accept = 0
    for _ in range(n_trials):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        re_genuine, ke_genuine = ctm.enroll(binary_codes[idx1])
        embed_e = hash_codes[idx1][ke_genuine]
        stored, _ = sstm.enroll(re_genuine, embed_e)
        ke_attack = ke_fn(ke_genuine, rng)
        re_impostor = ctm.authenticate(binary_codes[idx2], ke_attack)
        ok, _ = sstm.authenticate(re_impostor, stored)
        accept += int(ok)
    return accept / n_trials


# ──────────────────────────────────────────────
# Attack 1: Old-Template-Assisted
# ──────────────────────────────────────────────

def run_old_template_attack(binary_codes, hash_codes, labels, ctm, n_trials, rng):
    """
    Old-Template-Assisted Attack.

    Scenario: user enrolled under key k1, then revoked and enrolled under k2.
    Attacker has access to old template (registered under k1) and knows
    p% of k2. They use the old template to derive information about k2.

    Implementation:
      - We simulate: attacker has old biometric (enrolled under k1),
        and tries with a NEW impostor sample under partial k2 knowledge.
      - The "old template" assistance: attacker uses the same enrollment
        sample (that was used to generate T1) as the probe for T2.
      - This directly tests: can the OLD enrollment sample bypass the NEW template?

    Two sub-attacks:
      A. Old-sample-as-probe: use the original enrollment sample as the probe
         against the new template (k2). If T1 and T2 are cancelable, this fails.
      B. Old-key-helps-guess-new-key: attacker uses p% of k1 as hints for k2.
         (Tests if keys have any correlation).
    """
    print("\n  [Attack A: Old enrollment sample vs new template]")
    unique_ids = np.unique(labels)
    sstm_bch  = SSTM_BCH(G=G, m=BCH_M, t=BCH_T)
    try:
        sstm_rgss = SSTM_PolarEmbed(G=G, k_bits=RGSS_K, m=RGSS_M, t=RGSS_T)
    except AssertionError:
        sstm_rgss = None

    far_bch_A  = 0
    far_rgss_A = 0

    for _ in range(n_trials):
        uid = rng.choice(unique_ids)
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]   # same sample used for T1 and T2 enrollment

        # Generate T1 (old template, key k1)
        re1, ke1 = ctm.enroll(binary_codes[enroll_idx])

        # Generate T2 (new template, independent key k2)
        re2, ke2 = ctm.enroll(binary_codes[enroll_idx])

        # Enroll T2 (new template)
        stored_bch, _  = sstm_bch.enroll(re2)
        if sstm_rgss:
            embed_e = hash_codes[enroll_idx][ke2]
            stored_rgss, _ = sstm_rgss.enroll(re2, embed_e)

        # Attack A: use OLD enrollment sample mapped through NEW key k2
        # (old sample now "looks like" it was enrolled under k2, but it wasn't)
        re_old_under_k2 = ctm.authenticate(binary_codes[enroll_idx], ke2)
        ok_bch, _  = sstm_bch.authenticate(re_old_under_k2, stored_bch)
        far_bch_A  += int(ok_bch)
        if sstm_rgss:
            ok_rgss, _ = sstm_rgss.authenticate(re_old_under_k2, stored_rgss)
            far_rgss_A += int(ok_rgss)

    far_bch_A  = far_bch_A  / n_trials * 100
    far_rgss_A = (far_rgss_A / n_trials * 100) if sstm_rgss else None

    print(f"    BCH  FAR (old sample vs new template) = {far_bch_A:.2f}%")
    if far_rgss_A is not None:
        print(f"    RGSS FAR (old sample vs new template) = {far_rgss_A:.2f}%")
    print("    (Expected ≈ 0%: old sample maps to different location under new key)")

    # ── Attack B: old key used as partial hints for new key ──
    print("\n  [Attack B: Old key used as partial hints for guessing new key]")
    # Attacker uses ke1 (old key) as "guesses" for ke2 (new key).
    # If keys are truly independent, overlap between ke1 and ke2 ≈ G²/J (random)
    # This measures whether knowing old key helps guess new key.

    # Compute expected vs actual overlap
    overlaps = []
    for _ in range(n_trials):
        uid = rng.choice(unique_ids)
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]
        _, ke1 = ctm.enroll(binary_codes[enroll_idx])
        _, ke2 = ctm.enroll(binary_codes[enroll_idx])
        overlap = len(np.intersect1d(ke1, ke2))
        overlaps.append(overlap)

    mean_overlap = np.mean(overlaps)
    expected_overlap = G * G / ctm.hash_dim  # expected for independent random keys

    print(f"    Mean key overlap (ke1 ∩ ke2) = {mean_overlap:.1f} bits")
    print(f"    Expected by chance (G²/J)    = {expected_overlap:.1f} bits")
    diff = mean_overlap - expected_overlap
    print(f"    Excess overlap               = {diff:+.1f} bits")
    if abs(diff) < 5:
        print("    → Keys are effectively independent (no exploitable structure)")
    else:
        print("    → Some key correlation detected")

    return {
        "attack_A_far_BCH_%":    round(far_bch_A,  3),
        "attack_A_far_RGSS_%":   round(far_rgss_A, 3) if far_rgss_A else None,
        "mean_key_overlap":      round(mean_overlap, 2),
        "expected_key_overlap":  round(expected_overlap, 2),
        "excess_overlap":        round(diff, 2),
    }


# ──────────────────────────────────────────────
# Attack 2: Reliability-Aware Attack
# ──────────────────────────────────────────────

def run_reliability_aware_attack(binary_codes, hash_codes, labels, ctm,
                                  n_trials, rng):
    """
    Reliability-Aware Attack.
    Attacker knows RGSS prefers high |tanh| positions at population level.
    Compare FAR:
      - Random guessing (baseline, same as evaluate_attack_models.py)
      - Reliability-aware guessing (attacker prioritizes pop-reliable positions)
    """
    print("\n  Computing population-level reliable positions...")
    pop_reliable = compute_population_reliable_positions(
        hash_codes, labels, ctm, top_k=ctm.hash_dim
    )
    print(f"  Top-{G} pop-reliable positions computed.")

    results = {}
    for leakage_p in LEAKAGE_LEVELS:
        label = f"{int(leakage_p*100):3d}%"
        print(f"\n  Leakage = {label}")

        # Random guessing (baseline)
        def random_ke_fn(ke, r):
            return make_leaked_key(ke, leakage_p, ctm.hash_dim, r)

        # Reliability-aware guessing
        def aware_ke_fn(ke, r):
            return make_reliability_aware_key(ke, leakage_p, pop_reliable,
                                              ctm.hash_dim, r)

        far_bch_rand  = compute_far_bch_with_key(
            binary_codes, labels, ctm, random_ke_fn, n_trials, rng) * 100
        far_bch_aware = compute_far_bch_with_key(
            binary_codes, labels, ctm, aware_ke_fn,  n_trials, rng) * 100
        far_rgss_rand  = compute_far_rgss_with_key(
            binary_codes, hash_codes, labels, ctm, random_ke_fn, n_trials, rng)
        far_rgss_aware = compute_far_rgss_with_key(
            binary_codes, hash_codes, labels, ctm, aware_ke_fn,  n_trials, rng)
        far_rgss_rand  = (far_rgss_rand  * 100) if far_rgss_rand  is not None else None
        far_rgss_aware = (far_rgss_aware * 100) if far_rgss_aware is not None else None

        print(f"    BCH  FAR  random={far_bch_rand:.2f}%  aware={far_bch_aware:.2f}%")
        if far_rgss_rand is not None:
            print(f"    RGSS FAR  random={far_rgss_rand:.2f}%  aware={far_rgss_aware:.2f}%")

        results[label.strip()] = {
            "BCH_random_%":     round(far_bch_rand,   3),
            "BCH_aware_%":      round(far_bch_aware,  3),
            "RGSS_random_%":    round(far_rgss_rand,  3) if far_rgss_rand  else None,
            "RGSS_aware_%":     round(far_rgss_aware, 3) if far_rgss_aware else None,
        }

    return results


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_reliability_aware(aware_results, output_dir):
    levels = [float(k.replace('%', '')) for k in aware_results.keys()]
    bch_rand  = [aware_results[k]["BCH_random_%"]  for k in aware_results]
    bch_aware = [aware_results[k]["BCH_aware_%"]   for k in aware_results]
    rgss_rand = [aware_results[k]["RGSS_random_%"] for k in aware_results
                 if aware_results[k]["RGSS_random_%"] is not None]
    rgss_aware= [aware_results[k]["RGSS_aware_%"]  for k in aware_results
                 if aware_results[k]["RGSS_aware_%"] is not None]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, bch_rand,  'r-o',  linewidth=2, markersize=5,
            label='BCH — random guessing')
    ax.plot(levels, bch_aware, 'r--s', linewidth=2, markersize=5,
            label='BCH — reliability-aware guessing')
    if rgss_rand and rgss_aware:
        ax.plot(levels, rgss_rand,  'b-o',  linewidth=2, markersize=5,
                label='RGSS — random guessing')
        ax.plot(levels, rgss_aware, 'b--s', linewidth=2, markersize=5,
                label='RGSS — reliability-aware guessing')
    ax.set_xlabel('Key Leakage Level (%)')
    ax.set_ylabel('FAR (%)')
    ax.set_title('Reliability-Aware Attack vs Random Guessing\n'
                 '(If curves overlap → reliability info gives no advantage)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.3, max(max(bch_rand + bch_aware), 2) * 1.2)
    save_path = os.path.join(output_dir, "reliability_aware_attack.png")
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

    print("Extracting test codes...")
    test_binary, test_hash, test_labels = extract_codes_with_embed(
        model, test_loader, device)
    print(f"Test set: {test_binary.shape}, users: {len(np.unique(test_labels))}")

    ctm = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    rng = np.random.default_rng(42)

    # ── Attack 1: Old-Template-Assisted ────────
    print(f"\n{'='*60}")
    print("Attack 1: Old-Template-Assisted")
    print("="*60)
    ota_results = run_old_template_attack(
        test_binary, test_hash, test_labels, ctm, N_TRIALS, rng
    )

    # ── Attack 2: Reliability-Aware ────────────
    print(f"\n{'='*60}")
    print("Attack 2: Reliability-Aware Attack")
    print("="*60)
    aware_results = run_reliability_aware_attack(
        test_binary, test_hash, test_labels, ctm, N_TRIALS, rng
    )

    plot_reliability_aware(aware_results, OUTPUT_DIR)

    # ── Save JSON ──────────────────────────────
    results = {
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G, "n_trials": N_TRIALS,
        "BCH_operating_point":  {"m": BCH_M,  "t": BCH_T,  "k_bits": BCH_K},
        "RGSS_operating_point": {"m": RGSS_M, "t": RGSS_T, "k_bits": RGSS_K},
        "attack1_old_template_assisted": ota_results,
        "attack2_reliability_aware": aware_results,
    }
    json_path = os.path.join(OUTPUT_DIR, "attack_advanced_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
