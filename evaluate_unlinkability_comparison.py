"""
evaluate_unlinkability_comparison.py
Compare formal unlinkability metrics across four cancelable template frontends.

Fills the teacher-requested table:
  Method | Genuine same-key dist ↓ | Genuine diff-key dist ≈50% | Dsys ↓ | Linkability EER ↑
  -------+-------------------------+----------------------------+--------+-----------------
  CTM
  StableCTM
  StableCTM + RGSS   (= StableCTM at template level; same-key dist reflects reliable-ch only)
  BioHashing

Output: results_unlinkability_comparison/
  comparison_table.json   -- numerical results
  comparison_table.png    -- formatted table figure
  mated_vs_nonmated_*.png -- per-method distribution plots
"""

import os
import json
import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import StableCTM

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_unlinkability_comparison"
G            = 512
STABLE_RATIO = 0.8
N_KEYS       = 5        # keys per user for mated (cross-key) distances
N_NON_MATED  = 3000     # non-mated pairs to sample
RNG_SEED     = 42


# ──────────────────────────────────────────────
# BioHashing wrapper  (same API as StableCTM)
# ──────────────────────────────────────────────
class BioHashCTM:
    """
    BioHashing cancelable template (Teoh et al. 2004).
    key  = the G×D random projection matrix (flattened as a 1-D array for storage,
           but we keep it as a 2-D matrix internally).
    The template is sign(W @ x) ∈ {−1,+1}^G.
    """
    def __init__(self, G: int, hash_dim: int = 1024):
        self.G        = G
        self.hash_dim = hash_dim

    def enroll(self, binary_vec: np.ndarray,
               key: np.ndarray = None) -> tuple:
        """
        Parameters
        ----------
        binary_vec : (D,) int/float, the raw deep binary code
        key        : (G, D) float projection matrix, or None → generate new key

        Returns
        -------
        template : (G,) ±1
        key      : (G, D) projection matrix (to be stored for same-key reuse)
        """
        x = binary_vec.astype(float)
        D = len(x)
        if key is None:
            W = np.random.randn(self.G, D)
            # row-normalise so different feature dimensions are treated equally
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            W = W / np.maximum(norms, 1e-12)
        else:
            W = key  # (G, D)
        proj     = W @ x
        template = np.sign(proj)          # ±1
        template[template == 0] = 1       # break ties
        return template, W


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def hamming_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = (np.asarray(a) > 0).astype(np.uint8)
    b = (np.asarray(b) > 0).astype(np.uint8)
    return float(np.sum(a != b) / len(a))


def extract_codes(model, loader, device):
    model.eval()
    all_bin, all_lbl, all_tanh = [], [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            z, tanh_out, binary = model(imgs.to(device))
            all_bin.append(binary.cpu().numpy())
            all_lbl.append(lbs.numpy())
            all_tanh.append(tanh_out.cpu().numpy())
    return (np.vstack(all_bin),
            np.concatenate(all_lbl),
            np.vstack(all_tanh))


# ──────────────────────────────────────────────
# Per-method distance computation
# ──────────────────────────────────────────────
def compute_genuine_same_key(binary_codes, tanh_codes, labels,
                              ctm, rng, method_name):
    """
    Genuine same-key: same user, same key, different images.
    Uses the key returned by the first enroll() call and re-applies it.
    For StableCTM+RGSS, we additionally report the distance in the
    reliable channel (top-k positions by tanh confidence).
    """
    unique_ids = np.unique(labels)
    dists = []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        # enroll first image → capture key
        t1, ke = ctm.enroll(binary_codes[idx[0]])
        # apply the SAME key to the second image
        t2, _  = ctm.enroll(binary_codes[idx[1]], key=ke)
        dists.append(hamming_dist(t1, t2))
    return np.array(dists)


def compute_reliable_channel_same_key(binary_codes, tanh_codes, labels,
                                       ctm, k_reliable):
    """
    For StableCTM + RGSS: genuine same-key distance measured only in the
    k most reliable bits (as RGSS would select them).
    """
    unique_ids = np.unique(labels)
    dists = []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        # StableCTM enrolls with same key
        t1, ke = ctm.enroll(binary_codes[idx[0]])
        t2, _  = ctm.enroll(binary_codes[idx[1]], key=ke)
        # Reliable positions = top-k by tanh confidence of enrollment image
        # tanh_codes is (N, D=1024); select the G positions chosen by CTM, then top-k
        rel = np.abs(tanh_codes[idx[0]][ke])   # (G,) reliabilities in the G-dim space
        top_k = np.argsort(-rel)[:k_reliable]  # indices within the G-dim template
        d = hamming_dist(t1[top_k], t2[top_k])
        dists.append(d)
    return np.array(dists)


def compute_mated_nonmated(binary_codes, labels, ctm,
                            n_keys, n_non_mated, rng):
    """
    Mated   : same user, DIFFERENT keys  (cross-key)
    Non-mated: different users, DIFFERENT keys
    """
    unique_ids = np.unique(labels)

    # ── Mated ────────────────────────────────────
    mated = []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 1:
            continue
        enroll_img = binary_codes[idx[0]]
        templates  = []
        for _ in range(n_keys):
            t, _ = ctm.enroll(enroll_img)   # each call generates a new random key
            templates.append(t)
        for i in range(n_keys):
            for j in range(i + 1, n_keys):
                mated.append(hamming_dist(templates[i], templates[j]))

    # ── Non-mated ────────────────────────────────
    non_mated = []
    for _ in range(n_non_mated):
        id1, id2 = rng.choice(unique_ids, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == id1)[0])
        idx2 = rng.choice(np.where(labels == id2)[0])
        t1, _ = ctm.enroll(binary_codes[idx1])
        t2, _ = ctm.enroll(binary_codes[idx2])
        non_mated.append(hamming_dist(t1, t2))

    return np.array(mated), np.array(non_mated)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_linkability_eer(mated, non_mated):
    scores = np.concatenate([-mated, -non_mated])
    labels = np.concatenate([np.ones(len(mated)), np.zeros(len(non_mated))])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_dsys(mated, non_mated, n_bins=200):
    all_d = np.concatenate([mated, non_mated])
    s_min, s_max = all_d.min(), all_d.max()
    s_vals = np.linspace(s_min, s_max, n_bins)
    f_m  = np.array([np.mean(mated    <= s) for s in s_vals])
    f_nm = np.array([np.mean(non_mated <= s) for s in s_vals])
    ds   = s_vals[1] - s_vals[0]
    dsys = float(0.5 * np.sum(np.abs(f_m - f_nm)) * ds / max(s_max - s_min, 1e-12))
    return dsys


# ──────────────────────────────────────────────
# Plot helper
# ──────────────────────────────────────────────
def plot_distributions(mated, non_mated, method_name, output_dir):
    x = np.linspace(0, 0.8, 500)
    from scipy.stats import gaussian_kde
    fig, ax = plt.subplots(figsize=(8, 4))
    kde_m  = gaussian_kde(mated,     bw_method=0.08)
    kde_nm = gaussian_kde(non_mated, bw_method=0.08)
    ax.plot(x, kde_m(x),  'b-',  lw=2,
            label=f'Mated (same user, diff key)  μ={mated.mean()*100:.1f}%')
    ax.plot(x, kde_nm(x), 'r--', lw=2,
            label=f'Non-mated (diff user, diff key)  μ={non_mated.mean()*100:.1f}%')
    ax.fill_between(x, kde_m(x),  alpha=0.12, color='blue')
    ax.fill_between(x, kde_nm(x), alpha=0.12, color='red')
    ax.axvline(0.5, color='gray', ls=':', alpha=0.5, label='50% random')
    ax.set_xlabel('Normalised Hamming Distance')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Unlinkability: {method_name}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.8)
    path = os.path.join(output_dir, f'mated_vs_nonmated_{method_name}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ──────────────────────────────────────────────
# Summary table figure
# ──────────────────────────────────────────────
def save_table_figure(rows, output_dir):
    """
    rows: list of dicts with keys:
        method, same_key_mean, diff_key_mean, dsys, eer_pct
    """
    headers = ['Method',
               'Genuine same-key\ndist ↓ (%)',
               'Genuine diff-key\ndist ≈50% (%)',
               'Dsys ↓',
               'Linkability\nEER ↑ (%)']
    col_data = []
    for r in rows:
        col_data.append([
            r['method'],
            f"{r['same_key_mean']*100:.2f}",
            f"{r['diff_key_mean']*100:.2f} ± {r['diff_key_std']*100:.2f}",
            f"{r['dsys']:.4f}",
            f"{r['eer_pct']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 2.5 + 0.5 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=col_data, colLabels=headers,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    # Style header
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor('#DCE6F1')
        tbl[(0, j)].set_text_props(weight='bold')
    plt.title('Unlinkability Comparison Across Frontend Methods',
              fontsize=12, pad=10)
    path = os.path.join(output_dir, 'comparison_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved summary table: {path}')


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model & data ────────────────────────
    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8
    )
    model = FingerprintHashNet(num_classes=num_classes,
                               hash_dim=1024, pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f'Model loaded: {MODEL_PATH}')
    else:
        print('WARNING: using random weights')
    model = model.to(device)
    model.set_beta(32)

    print('\nExtracting training codes...')
    train_bin, train_lbl, _ = extract_codes(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_bin, train_lbl)
    print(f'  Training flip rate mean: {flip_rate.mean()*100:.2f}%')

    print('Extracting test codes...')
    test_bin, test_lbl, test_tanh = extract_codes(model, test_loader, device)
    print(f'  Test set: {test_bin.shape}, users: {len(np.unique(test_lbl))}')

    rng = np.random.default_rng(RNG_SEED)

    # ── Build CTM instances ───────────────────────
    # CTM  (stable_ratio=1.0 → all positions in pool → effectively random CTM)
    ctm_ctm   = StableCTM(hash_dim=1024, G=G,
                           flip_rate=flip_rate, stable_ratio=1.0)
    # StableCTM (stable_ratio=0.8)
    ctm_stable = StableCTM(hash_dim=1024, G=G,
                            flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    # BioHashing
    ctm_biohash = BioHashCTM(G=G, hash_dim=1024)

    # ── Configurations ────────────────────────────
    configs = [
        ('CTM',              ctm_ctm,    False),
        ('StableCTM',        ctm_stable, False),
        ('StableCTM + RGSS', ctm_stable, True),   # same template, reliable-ch distance
        ('BioHashing',       ctm_biohash, False),
    ]

    results = []

    for method_name, ctm, is_rgss in configs:
        print(f'\n{"="*60}')
        print(f'Method: {method_name}')
        print('='*60)

        # ── Genuine same-key dist ─────────────────
        if is_rgss:
            # Measure distance in reliable channel (RGSS selects top-k bits)
            k_reliable = 264   # RGSS operating point k50
            same_key_dists = compute_reliable_channel_same_key(
                test_bin, test_tanh, test_lbl, ctm, k_reliable
            )
            print(f'  [RGSS] Genuine same-key in reliable channel '
                  f'(k={k_reliable}): '
                  f'mean={same_key_dists.mean()*100:.2f}%, '
                  f'std={same_key_dists.std()*100:.2f}%')
        else:
            same_key_dists = compute_genuine_same_key(
                test_bin, test_tanh, test_lbl, ctm, rng, method_name
            )
            print(f'  Genuine same-key: '
                  f'mean={same_key_dists.mean()*100:.2f}%, '
                  f'std={same_key_dists.std()*100:.2f}%')

        # ── Mated / Non-mated ──────────────────────
        mated, non_mated = compute_mated_nonmated(
            test_bin, test_lbl, ctm, N_KEYS, N_NON_MATED, rng
        )
        print(f'  Genuine diff-key (mated):   '
              f'n={len(mated)}, '
              f'mean={mated.mean()*100:.2f}%, '
              f'std={mated.std()*100:.2f}%')
        print(f'  Non-mated:                  '
              f'n={len(non_mated)}, '
              f'mean={non_mated.mean()*100:.2f}%, '
              f'std={non_mated.std()*100:.2f}%')

        # ── Formal unlinkability ──────────────────
        eer  = compute_linkability_eer(mated, non_mated)
        dsys = compute_dsys(mated, non_mated)
        print(f'  Linkability EER = {eer*100:.2f}%   '
              f'(50% = perfect, 0% = fully linkable)')
        print(f'  Dsys            = {dsys:.4f}        '
              f'(0 = perfectly unlinkable)')

        # ── Plot ──────────────────────────────────
        plot_distributions(mated, non_mated,
                           method_name.replace(' ', '_'), OUTPUT_DIR)

        results.append({
            'method':        method_name,
            'same_key_mean': float(same_key_dists.mean()),
            'same_key_std':  float(same_key_dists.std()),
            'diff_key_mean': float(mated.mean()),
            'diff_key_std':  float(mated.std()),
            'nm_mean':       float(non_mated.mean()),
            'nm_std':        float(non_mated.std()),
            'dsys':          dsys,
            'eer_pct':       eer * 100,
        })

    # ── Save JSON ────────────────────────────────
    out = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'G': G, 'n_keys': N_KEYS, 'n_non_mated': N_NON_MATED,
        'results': results,
    }
    json_path = os.path.join(OUTPUT_DIR, 'comparison_table.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved: {json_path}')

    # ── Summary ───────────────────────────────────
    print('\n' + '='*70)
    print(f'{"Method":<22} {"same-key↓":>10} {"diff-key≈50%":>14} '
          f'{"Dsys↓":>8} {"EER↑":>8}')
    print('-'*70)
    for r in results:
        print(f"{r['method']:<22} "
              f"{r['same_key_mean']*100:>9.2f}% "
              f"{r['diff_key_mean']*100:>12.2f}% "
              f"{r['dsys']:>8.4f} "
              f"{r['eer_pct']:>7.1f}%")

    # ── Table figure ─────────────────────────────
    save_table_figure(results, OUTPUT_DIR)


if __name__ == '__main__':
    main()
