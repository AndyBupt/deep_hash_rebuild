"""
evaluate_stable.py — Full Evaluation with StableCTM

与 evaluate.py 完全相同的评估流程，唯一区别：
  所有 CTM 均替换为 StableCTM（stable_ratio=0.8）

输出：
  - ROC 曲线、距离分布图（StableCTM 版本）
  - G-S 曲线（RS 码，StableCTM）
  - results_stable/results_summary.json

运行方式：
  python evaluate_stable.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import StableCTM
from sstm import SSTM


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH = "checkpoints/final_model.pth"
DATA_ROOT  = "/root/autodl-tmp/FVC2004"
DB_NAMES   = ["DB1_A/image", "DB1_B/image",
              "DB2_A/image", "DB2_B/image",
              "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR = "results_stable"
G_VALUES   = [128, 256, 512]
STABLE_RATIO = 0.8


# ─────────────────────────────────────────────
# 工具函数（与 evaluate.py 相同）
# ─────────────────────────────────────────────

def extract_all_binary_codes(model, loader, device):
    model.eval()
    all_codes, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, _, binary = model(imgs)
            all_codes.append(binary.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_codes), np.concatenate(all_labels)


def compute_genuine_impostor_distances(codes, labels, ctm, scenario="unknown_key"):
    genuine_dists, impostor_dists = [], []
    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)

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
        for _ in range(n_impostors):
            id1, id2 = rng.choice(unique_ids, size=2, replace=False)
            idx1 = rng.choice(np.where(labels == id1)[0])
            idx2 = rng.choice(np.where(labels == id2)[0])
            re, _ = ctm.enroll(codes[idx1], seed=int(idx1))
            _, ke_random = ctm.enroll(codes[idx2],
                                      seed=int(rng.integers(0, 99999)))
            rp = ctm.authenticate(codes[idx2], ke_random)
            d = ctm.hamming_distance(re, rp)
            impostor_dists.append(d / ctm.G)
    elif scenario == "stolen_key":
        for _ in range(n_impostors):
            id1, id2 = rng.choice(unique_ids, size=2, replace=False)
            idx1 = rng.choice(np.where(labels == id1)[0])
            idx2 = rng.choice(np.where(labels == id2)[0])
            re, ke = ctm.enroll(codes[idx1])
            rp = ctm.authenticate(codes[idx2], ke)
            d = ctm.hamming_distance(re, rp)
            impostor_dists.append(d / ctm.G)

    return np.array(genuine_dists), np.array(impostor_dists)


def compute_eer(genuine_dists, impostor_dists):
    scores = np.concatenate([-genuine_dists, -impostor_dists])
    y_true = np.concatenate([np.ones(len(genuine_dists)),
                              np.zeros(len(impostor_dists))])
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer, fpr, tpr, thresholds


def compute_gar_at_far(fpr, tpr, target_far=0.005):
    idx = np.searchsorted(fpr, target_far)
    if idx >= len(tpr):
        return tpr[-1]
    return tpr[idx]


def plot_distributions(genuine_dists, impostor_uk_dists, impostor_sk_dists,
                       G, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 1, 300)
    g_mean, g_std   = genuine_dists.mean(), genuine_dists.std()
    uk_mean, uk_std = impostor_uk_dists.mean(), impostor_uk_dists.std()
    sk_mean, sk_std = impostor_sk_dists.mean(), impostor_sk_dists.std()

    ax.plot(x, norm.pdf(x, g_mean, g_std),   'b-',  linewidth=2, label='Genuine')
    ax.plot(x, norm.pdf(x, uk_mean, uk_std),  'r-',  linewidth=2,
            label='Impostor (Unknown Key)')
    ax.plot(x, norm.pdf(x, sk_mean, sk_std),  'g--', linewidth=2,
            label='Impostor (Stolen Key)')
    ax.fill_between(x, norm.pdf(x, g_mean, g_std),  alpha=0.15, color='blue')
    ax.fill_between(x, norm.pdf(x, uk_mean, uk_std), alpha=0.15, color='red')
    ax.fill_between(x, norm.pdf(x, sk_mean, sk_std), alpha=0.15, color='green')

    ax.set_xlabel('Normalized Hamming Distance')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Genuine/Impostor Distribution (G={G}, StableCTM)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved: {save_path}")
    plt.close()


def plot_roc(fpr_uk, tpr_uk, eer_uk, fpr_sk, tpr_sk, eer_sk, G, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_uk = auc(fpr_uk, tpr_uk)
    auc_sk = auc(fpr_sk, tpr_sk)

    fpr_uk_pct, tpr_uk_pct = fpr_uk * 100, tpr_uk * 100
    fpr_sk_pct, tpr_sk_pct = fpr_sk * 100, tpr_sk * 100

    ax.semilogx(fpr_uk_pct, tpr_uk_pct, 'b-', linewidth=2,
                label=f'Unknown Key (AUC={auc_uk:.3f}, EER={eer_uk*100:.2f}%)')
    ax.semilogx(fpr_sk_pct, tpr_sk_pct, 'r--', linewidth=2,
                label=f'Stolen Key  (AUC={auc_sk:.3f}, EER={eer_sk*100:.2f}%)')
    ax.scatter([eer_uk * 100], [100 - eer_uk * 100], color='blue', s=60, zorder=5)
    ax.scatter([eer_sk * 100], [100 - eer_sk * 100], color='red',  s=60, zorder=5)

    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(0, 105)
    ax.set_xlabel('FAR (%)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'ROC Curve (G={G}, StableCTM)')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved: {save_path}")
    plt.close()


def _compute_gar_with_sstm(codes, labels, ctm, sstm):
    """
    计算真实用户的 GAR。
    G-S 曲线中真实用户永远用自己的 ke，与 scenario 无关。
    """
    unique_ids = np.unique(labels)
    pass_count = total = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(codes[idx[0]])
        stored_hash, _ = sstm.enroll(re)
        for i in idx[1:]:
            rp = ctm.authenticate(codes[i], ke)
            is_genuine, _ = sstm.authenticate(rp, stored_hash)
            pass_count += int(is_genuine)
            total += 1
    return pass_count / total if total > 0 else 0


def plot_gs_curve(codes, labels, ctm, K_values, G, save_path=None):
    k_bits_list = [K * 8 for K in K_values]
    gars = []

    print(f"\nComputing G-S curve (G={G}, StableCTM)...")
    for K in K_values:
        if G // 8 <= K:
            gars.append(0)
            continue
        sstm = SSTM(G=G, K=K)
        gar = _compute_gar_with_sstm(codes, labels, ctm, sstm)
        gars.append(gar * 100)
        print(f"  K={K:3d} (k={K*8:4d} bits): GAR={gar*100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_bits_list, gars, 'g-^', linewidth=2, markersize=4,
            label='StableCTM + RS Code')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S Curve: StableCTM (G={G} bits)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"G-S curve saved: {save_path}")
    plt.close()

    return k_bits_list, gars


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据（需要 train_loader 计算翻转率）
    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8
    )

    # 加载模型
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024,
                               pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print("WARNING: Using random model (for testing only)")
    model = model.to(device)
    model.set_beta(32)

    # 提取训练集特征，计算翻转率（StableCTM 需要）
    print("\nExtracting training set codes (for flip rate)...")
    train_codes, train_labels = extract_all_binary_codes(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_codes, train_labels)
    print(f"Flip rate: mean={flip_rate.mean()*100:.1f}%, "
          f"std={flip_rate.std()*100:.1f}%")

    # 提取测试集特征
    print("\nExtracting test set codes...")
    codes, labels = extract_all_binary_codes(model, test_loader, device)
    print(f"Test codes shape: {codes.shape}")

    # ── 评估各 G 值 ──────────────────────────────
    results = {}
    for G in G_VALUES:
        print(f"\n{'='*50}")
        print(f"Evaluating G={G} bits  [StableCTM, stable_ratio={STABLE_RATIO}]")
        print(f"{'='*50}")

        # 初始化 StableCTM
        ctm = StableCTM(hash_dim=1024, G=G,
                        flip_rate=flip_rate, stable_ratio=STABLE_RATIO)

        # 距离分布
        genuine_dists, imp_uk = compute_genuine_impostor_distances(
            codes, labels, ctm, scenario="unknown_key"
        )
        _, imp_sk = compute_genuine_impostor_distances(
            codes, labels, ctm, scenario="stolen_key"
        )

        p_bit = float(genuine_dists.mean())
        ser   = 1 - (1 - p_bit) ** 8

        print(f"  Genuine:       mean={genuine_dists.mean():.3f}, "
              f"std={genuine_dists.std():.3f}")
        print(f"  Bit flip rate: {p_bit*100:.1f}%  "
              f"Symbol error rate: {ser*100:.1f}%")
        print(f"  Impostor(UK):  mean={imp_uk.mean():.3f}")
        print(f"  Impostor(SK):  mean={imp_sk.mean():.3f}")

        # EER / ROC
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
                           save_path=os.path.join(OUTPUT_DIR, f"dist_G{G}.png"))
        plot_roc(fpr_uk, tpr_uk, eer_uk, fpr_sk, tpr_sk, eer_sk, G,
                 save_path=os.path.join(OUTPUT_DIR, f"roc_G{G}.png"))

    # ── 汇总表 ────────────────────────────────────
    print("\n" + "="*65)
    print("Summary Results (StableCTM):")
    print(f"{'G':>6} | {'EER_UK(%)':>10} | {'EER_SK(%)':>10} | "
          f"{'GAR_UK@0.5%':>12} | {'GAR_SK@0.5%':>12}")
    print("-" * 65)
    for G in G_VALUES:
        r = results[G]
        print(f"{G:>6} | {r['eer_uk']*100:>10.2f} | {r['eer_sk']*100:>10.2f} | "
              f"{r['gar_uk']*100:>12.2f} | {r['gar_sk']*100:>12.2f}")

    # ── G-S 曲线（最大 G 值）────────────────────
    best_G = max(G_VALUES)
    N = best_G // 8
    K_values = list(range(7, N, 2))

    ctm_best = StableCTM(hash_dim=1024, G=best_G,
                         flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    k_bits_list, gars_gs = plot_gs_curve(
        codes, labels, ctm_best, K_values, best_G,
        save_path=os.path.join(OUTPUT_DIR, f"gs_curve_G{best_G}.png")
    )

    # ── 保存 JSON ─────────────────────────────────
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": MODEL_PATH,
        "dataset": DATA_ROOT,
        "ctm_method": f"StableCTM(stable_ratio={STABLE_RATIO})",
        "sstm_method": "fuzzy_commitment_RS_GF2^8",
        "G_values": G_VALUES,
        "per_G_results": {},
        "gs_curve": {
            "G": best_G,
            "K_values": K_values,
            "k_bits": k_bits_list,
            "GAR_StableCTM (%)": [round(g, 2) for g in gars_gs],
        }
    }
    for G in G_VALUES:
        r = results[G]
        p_bit = float(r['genuine_dists'].mean())
        ser   = 1 - (1 - p_bit) ** 8
        summary["per_G_results"][str(G)] = {
            "genuine_hamming_mean":  round(p_bit, 4),
            "genuine_hamming_std":   round(float(r['genuine_dists'].std()), 4),
            "genuine_bit_flip_rate": f"{p_bit*100:.1f}%",
            "symbol_error_rate":     f"{ser*100:.1f}%",
            "impostor_uk_mean":      round(float(r['imp_uk'].mean()), 4),
            "impostor_sk_mean":      round(float(r['imp_sk'].mean()), 4),
            "EER_unknown_key (%)":   round(r['eer_uk'] * 100, 2),
            "EER_stolen_key (%)":    round(r['eer_sk'] * 100, 2),
            "GAR@FAR=0.5%_uk (%)":  round(r['gar_uk'] * 100, 2),
            "GAR@FAR=0.5%_sk (%)":  round(r['gar_sk'] * 100, 2),
            "AUC_uk": round(float(auc(r['fpr_uk'], r['tpr_uk'])), 4),
            "AUC_sk": round(float(auc(r['fpr_sk'], r['tpr_sk'])), 4),
        }

    json_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
