"""
evaluate_multi_helper_leakage.py — Multi-Helper / Multi-Revocation Leakage Analysis

Goal:
  Evaluate whether repeated re-enrollment (multiple helper-data releases for the
  same user) accumulates exploitable leakage beyond the single-record analysis.

This script follows the style of the existing supplementary evaluations and
creates a dedicated output directory for easy inspection on the server.

Experiments:
  A. Cumulative helper-bit union:
     How many unique 1024-dim positions become exposed through h[~R] after
     multiple revocations?

  B. Multi-record reliable-position linkability:
     If an attacker observes M helper records for one user, how separable is the
     union/intersection pattern from that of different users?

  C. Multi-record helper-aware impostor attack:
     Compare one-record vs multi-record public helper leakage, using the union of
     exposed helper bits as auxiliary information for candidate selection.

Outputs: results_multi_helper_leakage/
  cumulative_exposure.png
  multi_record_linkability.png
  multi_record_attack.png
  multi_helper_leakage.json

Usage:
  python evaluate_multi_helper_leakage.py
"""

import datetime
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
    ensure_dir,
    extract_codes_with_embed,
    load_model_and_dataloaders,
    parse_rgss_template,
    compute_eer_from_scores,
    find_bch_params_for_k,
)
from ctm import StableCTM
from sstm_polar_embed import SSTM_PolarEmbed


MODEL_PATH = os.environ.get("RGSS_MODEL_PATH", "checkpoints/final_model.pth")
DATA_ROOT = os.environ.get("FVC2004_ROOT", DEFAULT_DATA_ROOT)
DB_NAMES = DEFAULT_DB_NAMES
OUTPUT_DIR = "results_multi_helper_leakage"
G = 512
STABLE_RATIO = 0.8
RGSS_K = 264
N_RECORDS = 5
N_NON_MATED = 1000
POOL_SIZE = 10
RANDOM_SEED = 42

RGSS_M, RGSS_T = find_bch_params_for_k(RGSS_K)


def get_exposed_positions(binary_code, hash_code, ctm, sstm, n_records):
    """Generate n_records helper templates and return exposed-position sets."""
    records = []
    for _ in range(n_records):
        re, ke = ctm.enroll(binary_code)
        embed_e = hash_code[ke]
        stored, _ = sstm.enroll(re, embed_e)
        parsed = parse_rgss_template(stored)
        perm = parsed["perm"]
        exposed_in_ke = perm[RGSS_K:]
        exposed_global = ke[exposed_in_ke]
        reliable_global = ke[perm[:RGSS_K]]
        exposed_bits = (binary_code[exposed_global] > 0).astype(np.uint8)
        reliable_bits = (binary_code[reliable_global] > 0).astype(np.uint8)
        records.append({
            "ke": ke,
            "exposed_positions": exposed_global,
            "exposed_bits": exposed_bits,
            "reliable_positions": reliable_global,
            "reliable_bits": reliable_bits,
        })
    return records


def jaccard(a, b):
    sa, sb = set(map(int, a)), set(map(int, b))
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def compute_union_and_intersection(records, field):
    sets = [set(map(int, r[field])) for r in records]
    union = set().union(*sets) if sets else set()
    inter = set(sets[0]) if sets else set()
    for s in sets[1:]:
        inter &= s
    return union, inter


def run_experiment(binary_codes, hash_codes, labels, ctm, sstm, rng):
    unique_ids = np.unique(labels)
    users = []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_idx = idx[0]
        records = get_exposed_positions(binary_codes[enroll_idx], hash_codes[enroll_idx],
                                        ctm, sstm, N_RECORDS)
        union_exp, inter_exp = compute_union_and_intersection(records, "exposed_positions")
        union_rel, inter_rel = compute_union_and_intersection(records, "reliable_positions")
        users.append({
            "uid": int(uid),
            "index": int(enroll_idx),
            "records": records,
            "union_exposed": np.array(sorted(union_exp), dtype=np.int64),
            "intersection_exposed": np.array(sorted(inter_exp), dtype=np.int64),
            "union_reliable": np.array(sorted(union_rel), dtype=np.int64),
            "intersection_reliable": np.array(sorted(inter_rel), dtype=np.int64),
        })

    # A. cumulative exposed coverage
    cumulative_unique = []
    cumulative_fraction = []
    for u in users:
        seen = set()
        curve = []
        frac_curve = []
        for rec in u["records"]:
            seen |= set(map(int, rec["exposed_positions"]))
            curve.append(len(seen))
            frac_curve.append(len(seen) / 1024.0)
        cumulative_unique.append(curve)
        cumulative_fraction.append(frac_curve)
    cumulative_unique = np.array(cumulative_unique, dtype=np.float64)
    cumulative_fraction = np.array(cumulative_fraction, dtype=np.float64)

    # B. multi-record linkability
    mated_union_jaccard = []
    mated_inter_jaccard = []
    non_union_jaccard = []
    non_inter_jaccard = []

    for u in users:
        half = max(1, N_RECORDS // 2)
        left = u["records"][:half]
        right = u["records"][half:]
        if not right:
            right = u["records"][-1:]
        left_union, left_inter = compute_union_and_intersection(left, "reliable_positions")
        right_union, right_inter = compute_union_and_intersection(right, "reliable_positions")
        mated_union_jaccard.append(jaccard(left_union, right_union))
        mated_inter_jaccard.append(jaccard(left_inter, right_inter))

    for _ in range(N_NON_MATED):
        a, b = rng.choice(len(users), size=2, replace=False)
        ua, ub = users[a], users[b]
        non_union_jaccard.append(jaccard(ua["union_reliable"], ub["union_reliable"]))
        non_inter_jaccard.append(jaccard(ua["intersection_reliable"], ub["intersection_reliable"]))

    mated_union_jaccard = np.array(mated_union_jaccard, dtype=np.float64)
    mated_inter_jaccard = np.array(mated_inter_jaccard, dtype=np.float64)
    non_union_jaccard = np.array(non_union_jaccard, dtype=np.float64)
    non_inter_jaccard = np.array(non_inter_jaccard, dtype=np.float64)

    link_eer_union = compute_eer_from_scores(mated_union_jaccard, non_union_jaccard, True) * 100
    link_eer_inter = compute_eer_from_scores(mated_inter_jaccard, non_inter_jaccard, True) * 100

    # C. multi-record helper-aware impostor attack
    single_far = 0
    multi_far = 0
    for _ in range(N_NON_MATED):
        victim = users[rng.integers(len(users))]
        victim_bits = (binary_codes[victim["index"]] > 0).astype(np.uint8)
        other_user_ids = [i for i in range(len(users)) if i != victim["uid"] and users[i]["uid"] != victim["uid"]]
        if len(other_user_ids) == 0:
            continue
        sampled = rng.choice([i for i in range(len(users)) if users[i]["uid"] != victim["uid"]],
                             size=min(POOL_SIZE, len(users)-1), replace=False)
        candidate_indices = [users[i]["index"] for i in sampled]

        # single-record helper knowledge
        rec0 = victim["records"][0]
        pos_single = rec0["exposed_positions"]
        bits_single = rec0["exposed_bits"]
        best_single = None
        best_single_score = -1

        # multi-record helper knowledge (union)
        union_positions = victim["union_exposed"]
        union_bits = victim_bits[union_positions]
        best_multi = None
        best_multi_score = -1

        for idx in candidate_indices:
            cand_bits = (binary_codes[idx] > 0).astype(np.uint8)
            score_single = int(np.sum(cand_bits[pos_single] == bits_single))
            score_multi = int(np.sum(cand_bits[union_positions] == union_bits))
            if score_single > best_single_score:
                best_single_score = score_single
                best_single = idx
            if score_multi > best_multi_score:
                best_multi_score = score_multi
                best_multi = idx

        # attack against latest record only, fair 1-vs-1 verification
        target_record = victim["records"][-1]
        re_target = binary_codes[victim["index"]][target_record["ke"]]
        embed_target = hash_codes[victim["index"]][target_record["ke"]]
        stored_target, _ = sstm.enroll(re_target, embed_target)

        re_probe_single = ctm.authenticate(binary_codes[best_single], target_record["ke"])
        re_probe_multi = ctm.authenticate(binary_codes[best_multi], target_record["ke"])
        ok_single, _ = sstm.authenticate(re_probe_single, stored_target)
        ok_multi, _ = sstm.authenticate(re_probe_multi, stored_target)
        single_far += int(ok_single)
        multi_far += int(ok_multi)

    single_far = single_far / N_NON_MATED * 100
    multi_far = multi_far / N_NON_MATED * 100

    return {
        "cumulative_unique": cumulative_unique,
        "cumulative_fraction": cumulative_fraction,
        "mated_union_jaccard": mated_union_jaccard,
        "mated_inter_jaccard": mated_inter_jaccard,
        "non_union_jaccard": non_union_jaccard,
        "non_inter_jaccard": non_inter_jaccard,
        "single_far": single_far,
        "multi_far": multi_far,
        "link_eer_union": link_eer_union,
        "link_eer_intersection": link_eer_inter,
    }


def plot_cumulative(results):
    x = np.arange(1, N_RECORDS + 1)
    mean_unique = results["cumulative_unique"].mean(axis=0)
    std_unique = results["cumulative_unique"].std(axis=0)
    mean_frac = results["cumulative_fraction"].mean(axis=0) * 100
    std_frac = results["cumulative_fraction"].std(axis=0) * 100

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, mean_unique, 'o-', color='#1f77b4', label='Unique exposed positions')
    ax1.fill_between(x, mean_unique-std_unique, mean_unique+std_unique,
                     color='#1f77b4', alpha=0.15)
    ax1.set_xlabel('Number of helper records for the same user')
    ax1.set_ylabel('Unique exposed positions in 1024-dim space')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, mean_frac, 's--', color='#d62728', label='Coverage (%)')
    ax2.fill_between(x, mean_frac-std_frac, mean_frac+std_frac,
                     color='#d62728', alpha=0.12)
    ax2.set_ylabel('Coverage of full 1024-dim code (%)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax1.set_title('Cumulative helper-data exposure across multiple revocations')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cumulative_exposure.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_linkability(results):
    x = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, mated, non_mated, title in [
        (axes[0], results['mated_union_jaccard'], results['non_union_jaccard'],
         f'Union-of-records linkability (EER={results["link_eer_union"]:.1f}%)'),
        (axes[1], results['mated_inter_jaccard'], results['non_inter_jaccard'],
         f'Intersection-of-records linkability (EER={results["link_eer_intersection"]:.1f}%)'),
    ]:
        kde_m = gaussian_kde(mated, bw_method=0.08)
        kde_n = gaussian_kde(non_mated, bw_method=0.08)
        ax.plot(x, kde_m(x), 'b-', linewidth=2, label=f'Mated μ={mated.mean():.3f}')
        ax.plot(x, kde_n(x), 'r--', linewidth=2, label=f'Non-mated μ={non_mated.mean():.3f}')
        ax.fill_between(x, kde_m(x), color='blue', alpha=0.12)
        ax.fill_between(x, kde_n(x), color='red', alpha=0.12)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Jaccard similarity')
        ax.set_title(title)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Probability density')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_record_linkability.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_attack(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Single helper record', 'Union of 5 helper records']
    values = [results['single_far'], results['multi_far']]
    bars = ax.bar(labels, values, color=['#9ecae1', '#3182bd'], edgecolor='black')
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f'{v:.2f}%', ha='center', va='bottom')
    ax.set_ylabel('FAR (%)')
    ax.set_title('Helper-aware impostor attack: single vs multi-record leakage')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_record_attack.png'), dpi=150, bbox_inches='tight')
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
    plot_cumulative(results)
    plot_linkability(results)
    plot_attack(results)

    json_results = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'MODEL_PATH': MODEL_PATH,
            'DATA_ROOT': DATA_ROOT,
            'DB_NAMES': DB_NAMES,
            'OUTPUT_DIR': OUTPUT_DIR,
            'G': G,
            'RGSS_K': RGSS_K,
            'N_RECORDS': N_RECORDS,
            'N_NON_MATED': N_NON_MATED,
            'POOL_SIZE': POOL_SIZE,
        },
        'cumulative_unique_mean': results['cumulative_unique'].mean(axis=0).round(3).tolist(),
        'cumulative_unique_std': results['cumulative_unique'].std(axis=0).round(3).tolist(),
        'cumulative_fraction_mean_%': (results['cumulative_fraction'].mean(axis=0) * 100).round(3).tolist(),
        'union_linkability': {
            'mated_mean': round(float(results['mated_union_jaccard'].mean()), 4),
            'non_mated_mean': round(float(results['non_union_jaccard'].mean()), 4),
            'eer_%': round(float(results['link_eer_union']), 2),
        },
        'intersection_linkability': {
            'mated_mean': round(float(results['mated_inter_jaccard'].mean()), 4),
            'non_mated_mean': round(float(results['non_inter_jaccard'].mean()), 4),
            'eer_%': round(float(results['link_eer_intersection']), 2),
        },
        'helper_aware_attack_far_%': {
            'single_record': round(float(results['single_far']), 3),
            'multi_record_union': round(float(results['multi_far']), 3),
        },
    }
    json_path = os.path.join(OUTPUT_DIR, 'multi_helper_leakage.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nResults saved: {json_path}')


if __name__ == '__main__':
    main()
