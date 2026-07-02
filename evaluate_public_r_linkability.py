"""
evaluate_public_r_linkability.py — Public-R vs Protected-R Linkability Comparison

Goal:
  Quantify the side-channel introduced by publishing reliable-position indices R.
  This script directly compares linkability when R is public vs when only a
  protected representation of R is available.

Experiments:
  1. Public-R position-set linkability (Jaccard-based).
  2. Protected-R baseline with salted hash tokens of R.
  3. Public helper template distance vs protected helper template distance.

Outputs: results_public_r_linkability/
  public_vs_protected_r_linkability.png
  helper_distance_linkability.png
  public_r_linkability.json

Usage:
  python evaluate_public_r_linkability.py
"""

import datetime
import hashlib
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from additional_experiment_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DB_NAMES,
    bits01,
    compute_eer_from_scores,
    ensure_dir,
    extract_codes_with_embed,
    find_bch_params_for_k,
    load_model_and_dataloaders,
    parse_rgss_template,
)
from ctm import StableCTM
from sstm_polar_embed import SSTM_PolarEmbed


MODEL_PATH = os.environ.get("RGSS_MODEL_PATH", "checkpoints/final_model.pth")
DATA_ROOT = os.environ.get("FVC2004_ROOT", DEFAULT_DATA_ROOT)
DB_NAMES = DEFAULT_DB_NAMES
OUTPUT_DIR = "results_public_r_linkability"
G = 512
STABLE_RATIO = 0.8
RGSS_K = 264
N_KEYS = 5
N_NON_MATED = 3000
RANDOM_SEED = 42

RGSS_M, RGSS_T = find_bch_params_for_k(RGSS_K)


def jaccard(a, b):
    sa, sb = set(map(int, a)), set(map(int, b))
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def protected_r_token(r_positions, salt: bytes) -> np.ndarray:
    """Hash each position with a user-specific salt to simulate protected R."""
    tokens = []
    for pos in np.asarray(r_positions, dtype=np.int64):
        h = hashlib.sha256(salt + int(pos).to_bytes(4, 'little')).digest()
        tokens.append(int.from_bytes(h[:8], 'little'))
    return np.array(sorted(tokens), dtype=np.uint64)


def helper_distance(parsed_a, parsed_b):
    h_a = parsed_a['h_bits'][:G]
    h_b = parsed_b['h_bits'][:G]
    return float(np.mean(h_a != h_b))


def generate_user_records(binary_code, hash_code, ctm, sstm, n_keys, rng):
    records = []
    for _ in range(n_keys):
        re, ke = ctm.enroll(binary_code)
        embed_e = hash_code[ke]
        stored, _ = sstm.enroll(re, embed_e)
        parsed = parse_rgss_template(stored)
        r_global = ke[parsed['perm'][:RGSS_K]]
        salt = rng.integers(0, 2**32, dtype=np.uint32).tobytes()
        records.append({
            'ke': ke,
            'stored': stored,
            'parsed': parsed,
            'r_public': r_global,
            'r_protected': protected_r_token(r_global, salt),
        })
    return records


def run_experiment(binary_codes, hash_codes, labels, ctm, sstm, rng):
    unique_ids = np.unique(labels)
    user_records = {}
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]
        user_records[int(uid)] = generate_user_records(
            binary_codes[enroll_idx], hash_codes[enroll_idx], ctm, sstm, N_KEYS, rng
        )

    public_mated, public_non = [], []
    protected_mated, protected_non = [], []
    helper_mated, helper_non = [], []

    for uid, records in user_records.items():
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                public_mated.append(jaccard(records[i]['r_public'], records[j]['r_public']))
                protected_mated.append(jaccard(records[i]['r_protected'], records[j]['r_protected']))
                helper_mated.append(helper_distance(records[i]['parsed'], records[j]['parsed']))

    all_uids = list(user_records.keys())
    for _ in range(N_NON_MATED):
        uid1, uid2 = rng.choice(all_uids, size=2, replace=False)
        rec1 = user_records[int(uid1)][rng.integers(N_KEYS)]
        rec2 = user_records[int(uid2)][rng.integers(N_KEYS)]
        public_non.append(jaccard(rec1['r_public'], rec2['r_public']))
        protected_non.append(jaccard(rec1['r_protected'], rec2['r_protected']))
        helper_non.append(helper_distance(rec1['parsed'], rec2['parsed']))

    public_mated = np.array(public_mated, dtype=np.float64)
    public_non = np.array(public_non, dtype=np.float64)
    protected_mated = np.array(protected_mated, dtype=np.float64)
    protected_non = np.array(protected_non, dtype=np.float64)
    helper_mated = np.array(helper_mated, dtype=np.float64)
    helper_non = np.array(helper_non, dtype=np.float64)

    public_eer = compute_eer_from_scores(public_mated, public_non, True) * 100
    protected_eer = compute_eer_from_scores(protected_mated, protected_non, True) * 100
    helper_eer = compute_eer_from_scores(helper_mated, helper_non, False) * 100

    return {
        'public_mated': public_mated,
        'public_non': public_non,
        'protected_mated': protected_mated,
        'protected_non': protected_non,
        'helper_mated': helper_mated,
        'helper_non': helper_non,
        'public_eer': public_eer,
        'protected_eer': protected_eer,
        'helper_eer': helper_eer,
    }


def plot_public_vs_protected(results):
    x = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mated, non_mated, title in [
        (axes[0], results['public_mated'], results['public_non'],
         f'Public R (EER={results["public_eer"]:.1f}%)'),
        (axes[1], results['protected_mated'], results['protected_non'],
         f'Protected R token (EER={results["protected_eer"]:.1f}%)'),
    ]:
        kde_m = gaussian_kde(mated, bw_method=0.07)
        kde_n = gaussian_kde(non_mated, bw_method=0.07)
        ax.plot(x, kde_m(x), 'b-', linewidth=2, label=f'Mated μ={mated.mean():.3f}')
        ax.plot(x, kde_n(x), 'r--', linewidth=2, label=f'Non-mated μ={non_mated.mean():.3f}')
        ax.fill_between(x, kde_m(x), color='blue', alpha=0.12)
        ax.fill_between(x, kde_n(x), color='red', alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel('Similarity score')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Probability density')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'public_vs_protected_r_linkability.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_helper_distance(results):
    x = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(7, 5))
    kde_m = gaussian_kde(results['helper_mated'], bw_method=0.07)
    kde_n = gaussian_kde(results['helper_non'], bw_method=0.07)
    ax.plot(x, kde_m(x), 'b-', linewidth=2, label=f'Mated μ={results["helper_mated"].mean():.3f}')
    ax.plot(x, kde_n(x), 'r--', linewidth=2, label=f'Non-mated μ={results["helper_non"].mean():.3f}')
    ax.fill_between(x, kde_m(x), color='blue', alpha=0.12)
    ax.fill_between(x, kde_n(x), color='red', alpha=0.12)
    ax.set_xlabel('Normalised Hamming distance of helper h')
    ax.set_ylabel('Probability density')
    ax.set_title(f'Helper-template distance linkability (EER={results["helper_eer"]:.1f}%)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'helper_distance_linkability.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)
    rng = np.random.default_rng(RANDOM_SEED)

    device, model, train_loader, test_loader, _ = load_model_and_dataloaders(
        data_root=DATA_ROOT, db_names=DB_NAMES, model_path=MODEL_PATH
    )
    print(f'Device: {device}')

    print('\nExtracting training codes...')
    train_binary, train_hash, train_labels = extract_codes_with_embed(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)

    print('Extracting test codes...')
    test_binary, test_hash, test_labels = extract_codes_with_embed(model, test_loader, device)

    ctm = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    sstm = SSTM_PolarEmbed(G=G, k_bits=RGSS_K, m=RGSS_M, t=RGSS_T)

    results = run_experiment(test_binary, test_hash, test_labels, ctm, sstm, rng)
    plot_public_vs_protected(results)
    plot_helper_distance(results)

    json_results = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'MODEL_PATH': MODEL_PATH,
            'DATA_ROOT': DATA_ROOT,
            'DB_NAMES': DB_NAMES,
            'OUTPUT_DIR': OUTPUT_DIR,
            'G': G,
            'RGSS_K': RGSS_K,
            'N_KEYS': N_KEYS,
            'N_NON_MATED': N_NON_MATED,
        },
        'public_R': {
            'mated_mean': round(float(results['public_mated'].mean()), 4),
            'non_mated_mean': round(float(results['public_non'].mean()), 4),
            'eer_%': round(float(results['public_eer']), 2),
        },
        'protected_R_token': {
            'mated_mean': round(float(results['protected_mated'].mean()), 4),
            'non_mated_mean': round(float(results['protected_non'].mean()), 4),
            'eer_%': round(float(results['protected_eer']), 2),
        },
        'helper_h_distance': {
            'mated_mean': round(float(results['helper_mated'].mean()), 4),
            'non_mated_mean': round(float(results['helper_non'].mean()), 4),
            'eer_%': round(float(results['helper_eer']), 2),
        },
    }
    json_path = os.path.join(OUTPUT_DIR, 'public_r_linkability.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nResults saved: {json_path}')


if __name__ == '__main__':
    main()
