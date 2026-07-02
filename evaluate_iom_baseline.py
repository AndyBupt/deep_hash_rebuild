"""
evaluate_iom_baseline.py — IoM-style baseline vs StableCTM vs RGSS

Goal:
  Add a stronger cancelable-biometrics baseline beyond BioHashing by using an
  IoM-style ranking transform with a BCH back-end.

Notes:
  This implementation is intentionally lightweight and repository-native: it
  keeps the same evaluation style and focuses on whether an IoM-style front-end
  can provide a competitive GAR–key-length operating point under the same
  experimental protocol.

Outputs: results_iom_baseline/
  roc_iom_vs_ctm.png
  gs_iom_vs_ctm_rgss.png
  iom_baseline_results.json

Usage:
  python evaluate_iom_baseline.py
"""

import datetime
import json
import os

import bchlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from additional_experiment_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DB_NAMES,
    ensure_dir,
    extract_codes_with_embed,
    find_bch_params_for_k,
    load_model_and_dataloaders,
    summarise_k50,
)
from biohashing import BioHashing
from ctm import StableCTM
from sstm_bch import SSTM_BCH
from sstm_polar_embed import SSTM_PolarEmbed


MODEL_PATH = os.environ.get("RGSS_MODEL_PATH", "checkpoints/final_model.pth")
DATA_ROOT = os.environ.get("FVC2004_ROOT", DEFAULT_DATA_ROOT)
DB_NAMES = DEFAULT_DB_NAMES
OUTPUT_DIR = "results_iom_baseline"
G = 512
STABLE_RATIO = 0.8
N_IMPOSTOR = 2000
IOM_WORDS = 128
IOM_Q = 4
RGSS_K = 264
RANDOM_SEED = 42

RGSS_M, RGSS_T = find_bch_params_for_k(RGSS_K)


class IoMHashing:
    """A simple repository-native IoM-style ranking transform."""

    def __init__(self, hash_dim: int = 1024, words: int = 128, q: int = 4):
        self.hash_dim = hash_dim
        self.words = words
        self.q = q
        self.G = words * q

    def _make_groups(self, key: int):
        rng = np.random.default_rng(int(key))
        perm = rng.permutation(self.hash_dim)
        groups = perm[:self.words * self.q].reshape(self.words, self.q)
        return groups

    def enroll(self, hash_vec: np.ndarray, key: int = None, seed=None):
        if key is None:
            key = int(seed if seed is not None else np.random.randint(0, 2**31))
        groups = self._make_groups(key)
        token = self._transform(hash_vec, groups)
        return token, key

    def authenticate(self, hash_vec: np.ndarray, key: int):
        groups = self._make_groups(key)
        return self._transform(hash_vec, groups)

    def _transform(self, hash_vec, groups):
        token = np.empty(self.G, dtype=np.int8)
        for i, g in enumerate(groups):
            scores = np.asarray(hash_vec[g], dtype=np.float32)
            winner = int(np.argmax(scores))
            token[i*self.q:(i+1)*self.q] = -1
            token[i*self.q + winner] = 1
        return token

    def hamming_distance(self, a, b):
        return int(np.sum(a != b))


def extract_binary_and_hash(model, loader, device):
    return extract_codes_with_embed(model, loader, device)


def compute_roc_for_frontend(labels, binary_codes, hash_codes, frontend, use_hash=False):
    rng = np.random.default_rng(RANDOM_SEED)
    unique_ids = np.unique(labels)
    genuine, impostor = [], []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        source = hash_codes if use_hash else binary_codes
        re, key = frontend.enroll(source[idx[0]])
        for i in idx[1:]:
            rp = frontend.authenticate(source[i], key)
            genuine.append(frontend.hamming_distance(re, rp) / frontend.G)

    n_imp = min(len(genuine) * 5, N_IMPOSTOR)
    for _ in range(n_imp):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        source = hash_codes if use_hash else binary_codes
        re, key = frontend.enroll(source[idx1])
        rp = frontend.authenticate(source[idx2], key)
        impostor.append(frontend.hamming_distance(re, rp) / frontend.G)

    genuine = np.array(genuine, dtype=np.float64)
    impostor = np.array(impostor, dtype=np.float64)
    scores = np.concatenate([-genuine, -impostor])
    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return genuine, impostor, fpr, tpr, eer, float(auc(fpr, tpr))


def get_bch_params(G):
    params = []
    for t in range(1, 57):
        try:
            b = bchlib.BCH(t=t, m=9)
            k_bits = ((b.n - b.ecc_bits) // 8) * 8
            if 40 <= k_bits < G:
                params.append((9, b.t, k_bits))
        except Exception:
            break
    seen = {}
    for m, t, k in params:
        if k not in seen or t > seen[k][1]:
            seen[k] = (m, t, k)
    return sorted(seen.values(), key=lambda x: x[2])


def compute_gar_bch_frontend(binary_codes, labels, frontend):
    unique_ids = np.unique(labels)
    bch_params = get_bch_params(frontend.G)
    k_list, gar_list = [], []
    for m, t, k_bits in bch_params:
        sstm = SSTM_BCH(G=frontend.G, m=m, t=t)
        passed = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, key = frontend.enroll(binary_codes[idx[0]])
            stored, _ = sstm.enroll(re)
            for i in idx[1:]:
                rp = frontend.authenticate(binary_codes[i], key)
                ok, _ = sstm.authenticate(rp, stored)
                passed += int(ok)
                total += 1
        k_list.append(k_bits)
        gar_list.append((passed / total * 100) if total else 0.0)
    return k_list, gar_list


def compute_gar_iom(hash_codes, labels, iom):
    unique_ids = np.unique(labels)
    bch_params = get_bch_params(iom.G)
    k_list, gar_list = [], []
    for m, t, k_bits in bch_params:
        sstm = SSTM_BCH(G=iom.G, m=m, t=t)
        passed = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, key = iom.enroll(hash_codes[idx[0]])
            stored, _ = sstm.enroll(re)
            for i in idx[1:]:
                rp = iom.authenticate(hash_codes[i], key)
                ok, _ = sstm.authenticate(rp, stored)
                passed += int(ok)
                total += 1
        k_list.append(k_bits)
        gar_list.append((passed / total * 100) if total else 0.0)
    return k_list, gar_list


def compute_gar_rgss(binary_codes, hash_codes, labels, ctm):
    unique_ids = np.unique(labels)
    params = get_bch_params(G)
    k_list, gar_list = [], []
    for m, t, k_bits in params:
        try:
            sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
        except AssertionError:
            continue
        passed = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, key = ctm.enroll(binary_codes[idx[0]])
            embed_e = hash_codes[idx[0]][key]
            stored, _ = sstm.enroll(re, embed_e)
            for i in idx[1:]:
                rp = ctm.authenticate(binary_codes[i], key)
                ok, _ = sstm.authenticate(rp, stored)
                passed += int(ok)
                total += 1
        k_list.append(k_bits)
        gar_list.append((passed / total * 100) if total else 0.0)
    return k_list, gar_list


def plot_roc(roc_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, result, style in [
        ('StableCTM', roc_results['stable'], 'b-'),
        ('BioHashing', roc_results['biohash'], 'r--'),
        ('IoM-style', roc_results['iom'], 'g-.'),
    ]:
        ax.semilogx(result['fpr'] * 100, result['tpr'] * 100, style, linewidth=2,
                    label=f"{name} (AUC={result['auc']:.3f}, EER={result['eer']*100:.2f}%)")
    ax.set_xlabel('FAR (%)')
    ax.set_ylabel('GAR (%)')
    ax.set_title('ROC comparison: StableCTM vs BioHashing vs IoM-style baseline')
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(0, 105)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_iom_vs_ctm.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_gs(gs_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, result, style in [
        ('StableCTM+BCH', gs_results['stable'], 'b-o'),
        ('BioHashing+BCH', gs_results['biohash'], 'r-s'),
        ('IoM-style+BCH', gs_results['iom'], 'g-^'),
        ('RGSS', gs_results['rgss'], 'k--D'),
    ]:
        k50 = summarise_k50(result['k_bits'], result['gars'])
        label = f'{name} (k50={k50})' if k50 is not None else name
        ax.plot(result['k_bits'], result['gars'], style, linewidth=2, markersize=4, label=label)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Key length (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title('GAR–key-length comparison with IoM-style baseline')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gs_iom_vs_ctm_rgss.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)
    device, model, train_loader, test_loader, _ = load_model_and_dataloaders(
        data_root=DATA_ROOT, db_names=DB_NAMES, model_path=MODEL_PATH
    )
    print(f'Device: {device}')

    print('\nExtracting training codes...')
    train_binary, train_hash, train_labels = extract_binary_and_hash(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)

    print('Extracting test codes...')
    test_binary, test_hash, test_labels = extract_binary_and_hash(model, test_loader, device)

    stable = StableCTM(hash_dim=1024, G=G, flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    biohash = BioHashing(hash_dim=1024, G=G)
    iom = IoMHashing(hash_dim=1024, words=IOM_WORDS, q=IOM_Q)

    print('\nRunning ROC comparisons...')
    roc_results = {}
    for name, frontend, use_hash in [
        ('stable', stable, False),
        ('biohash', biohash, False),
        ('iom', iom, True),
    ]:
        gen, imp, fpr, tpr, eer, auc_val = compute_roc_for_frontend(
            test_labels, test_binary, test_hash, frontend, use_hash=use_hash
        )
        roc_results[name] = {
            'genuine_mean': float(gen.mean()),
            'impostor_mean': float(imp.mean()),
            'eer': eer,
            'auc': auc_val,
            'fpr': fpr,
            'tpr': tpr,
        }

    print('\nRunning GAR–key-length comparisons...')
    stable_k, stable_g = compute_gar_bch_frontend(test_binary, test_labels, stable)
    bio_k, bio_g = compute_gar_bch_frontend(test_binary, test_labels, biohash)
    iom_k, iom_g = compute_gar_iom(test_hash, test_labels, iom)
    rgss_k, rgss_g = compute_gar_rgss(test_binary, test_hash, test_labels, stable)

    gs_results = {
        'stable': {'k_bits': stable_k, 'gars': stable_g},
        'biohash': {'k_bits': bio_k, 'gars': bio_g},
        'iom': {'k_bits': iom_k, 'gars': iom_g},
        'rgss': {'k_bits': rgss_k, 'gars': rgss_g},
    }

    plot_roc(roc_results)
    plot_gs(gs_results)

    json_results = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'MODEL_PATH': MODEL_PATH,
            'DATA_ROOT': DATA_ROOT,
            'DB_NAMES': DB_NAMES,
            'OUTPUT_DIR': OUTPUT_DIR,
            'G': G,
            'IOM_WORDS': IOM_WORDS,
            'IOM_Q': IOM_Q,
        },
        'roc': {
            k: {
                'genuine_mean': round(v['genuine_mean'], 4),
                'impostor_mean': round(v['impostor_mean'], 4),
                'eer_%': round(v['eer'] * 100, 2),
                'auc': round(v['auc'], 4),
            } for k, v in roc_results.items()
        },
        'k50_summary': {
            'StableCTM_BCH': summarise_k50(stable_k, stable_g),
            'BioHashing_BCH': summarise_k50(bio_k, bio_g),
            'IoM_style_BCH': summarise_k50(iom_k, iom_g),
            'RGSS': summarise_k50(rgss_k, rgss_g),
        },
        'gs_curves': {
            name: {'k_bits': res['k_bits'], 'GAR (%)': [round(float(g), 2) for g in res['gars']]}
            for name, res in gs_results.items()
        },
    }
    json_path = os.path.join(OUTPUT_DIR, 'iom_baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nResults saved: {json_path}')


if __name__ == '__main__':
    main()
