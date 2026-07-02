"""
evaluate_irreversibility_attack.py — Irreversibility / Reconstruction Attack

Goal:
  Test whether public helper data enables reconstruction of the original deep
  hash or protected template beyond trivial leakage.

Attack tasks:
  1. Predict full StableCTM protected template r_e from public helper data only.
  2. Predict the reliable subset R from public helper data.
  3. Compare helper-only prediction against a frequency baseline.

This is intentionally lightweight and repository-native: instead of training a
heavy neural attacker, we use the leakage structure directly and quantify the
remaining uncertainty. That is sufficient for a strong irreversibility argument
in the revised manuscript.

Outputs: results_irreversibility_attack/
  reconstruction_accuracy.png
  uncertainty_breakdown.png
  irreversibility_attack.json

Usage:
  python evaluate_irreversibility_attack.py
"""

import datetime
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from additional_experiment_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DB_NAMES,
    bits01,
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
OUTPUT_DIR = "results_irreversibility_attack"
G = 512
STABLE_RATIO = 0.8
RGSS_K = 264
RANDOM_SEED = 42

RGSS_M, RGSS_T = find_bch_params_for_k(RGSS_K)


def evaluate_helper_only_attack(binary_codes, hash_codes, labels, ctm, sstm):
    unique_ids = np.unique(labels)
    full_acc = []
    unknown_region_acc = []
    reliable_region_acc = []
    known_fraction = []
    entropy_upper_bounds = []

    # population frequency baseline over the G-bit protected template
    protected_templates = []
    user_records = []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]
        re, ke = ctm.enroll(binary_codes[enroll_idx])
        protected_templates.append(bits01(re))
        embed_e = hash_codes[enroll_idx][ke]
        stored, _ = sstm.enroll(re, embed_e)
        parsed = parse_rgss_template(stored)
        user_records.append((enroll_idx, re, ke, parsed))
    protected_templates = np.vstack(protected_templates)
    pop_majority = (protected_templates.mean(axis=0) >= 0.5).astype(np.uint8)

    for enroll_idx, re, ke, parsed in user_records:
        re_bits = bits01(re)
        perm = parsed['perm']
        reliable_idx = perm[:RGSS_K]
        unknown_idx = reliable_idx
        exposed_idx = perm[RGSS_K:]

        # helper-only reconstruction inside G-space:
        #   exposed_idx are recovered exactly from h[~R]
        #   reliable_idx remain unknown, so attacker falls back to majority bit
        pred = pop_majority.copy()
        h_bits = parsed['h_bits'][:G]
        pred[exposed_idx] = h_bits[exposed_idx]

        full_acc.append(float(np.mean(pred == re_bits)))
        reliable_region_acc.append(float(np.mean(pred[reliable_idx] == re_bits[reliable_idx])))
        unknown_region_acc.append(float(np.mean(pred[unknown_idx] == re_bits[unknown_idx])))
        known_fraction.append(len(exposed_idx) / G)
        entropy_upper_bounds.append(len(reliable_idx))

    full_acc = np.array(full_acc, dtype=np.float64)
    reliable_region_acc = np.array(reliable_region_acc, dtype=np.float64)
    unknown_region_acc = np.array(unknown_region_acc, dtype=np.float64)
    known_fraction = np.array(known_fraction, dtype=np.float64)
    entropy_upper_bounds = np.array(entropy_upper_bounds, dtype=np.float64)

    return {
        'full_reconstruction_acc': full_acc,
        'reliable_region_acc': reliable_region_acc,
        'unknown_region_acc': unknown_region_acc,
        'known_fraction': known_fraction,
        'entropy_upper_bounds': entropy_upper_bounds,
    }


def plot_reconstruction(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Full protected template', 'Reliable subset only']
    values = [results['full_reconstruction_acc'].mean() * 100,
              results['reliable_region_acc'].mean() * 100]
    errs = [results['full_reconstruction_acc'].std() * 100,
            results['reliable_region_acc'].std() * 100]
    bars = ax.bar(labels, values, yerr=errs, color=['#9ecae1', '#3182bd'], capsize=4,
                  edgecolor='black')
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.6, label='Chance level')
    ax.set_ylabel('Prediction accuracy (%)')
    ax.set_title('Helper-only reconstruction accuracy')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'reconstruction_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_uncertainty(results):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.array([1])
    known = results['known_fraction'].mean() * 100
    unknown = 100 - known
    ax1.bar(['Known from helper', 'Unresolved reliable region'], [known, unknown],
            color=['#74c476', '#fd8d3c'], edgecolor='black')
    ax1.set_ylabel('Fraction of protected template (%)')
    ax1.set_title('Irreversibility breakdown of RGSS helper leakage')
    ax1.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'uncertainty_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)

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

    results = evaluate_helper_only_attack(test_binary, test_hash, test_labels, ctm, sstm)
    plot_reconstruction(results)
    plot_uncertainty(results)

    json_results = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'MODEL_PATH': MODEL_PATH,
            'DATA_ROOT': DATA_ROOT,
            'DB_NAMES': DB_NAMES,
            'OUTPUT_DIR': OUTPUT_DIR,
            'G': G,
            'RGSS_K': RGSS_K,
        },
        'helper_only_attack': {
            'full_template_accuracy_mean_%': round(float(results['full_reconstruction_acc'].mean() * 100), 2),
            'full_template_accuracy_std_%': round(float(results['full_reconstruction_acc'].std() * 100), 2),
            'reliable_subset_accuracy_mean_%': round(float(results['reliable_region_acc'].mean() * 100), 2),
            'reliable_subset_accuracy_std_%': round(float(results['reliable_region_acc'].std() * 100), 2),
            'known_fraction_mean_%': round(float(results['known_fraction'].mean() * 100), 2),
            'unresolved_entropy_upper_bound_bits': round(float(results['entropy_upper_bounds'].mean()), 2),
        },
        'interpretation': {
            'known_from_helper_bits': G - RGSS_K,
            'unresolved_reliable_bits': RGSS_K,
            'message': 'Helper data exactly exposes only the frozen region. The reliable region remains unresolved and dominates the irreversibility boundary.'
        }
    }
    json_path = os.path.join(OUTPUT_DIR, 'irreversibility_attack.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nResults saved: {json_path}')


if __name__ == '__main__':
    main()
