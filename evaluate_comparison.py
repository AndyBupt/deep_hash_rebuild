"""
evaluate_comparison.py — BioHashing vs StableCTM 对比实验

三层实验结构：
  实验一（已有）：CTM + RS  vs  CTM + BCH  →  G-S 曲线消融实验
  实验二（本脚本）：BioHashing vs StableCTM  →  ROC/EER 对比（前端方法质量）
  实验三（本脚本）：BioHashing + BCH  vs  StableCTM + BCH  →  G-S 曲线对比

控制变量：
  - 实验二：相同 SSTM（无），相同测试集，只换前端方法
  - 实验三：相同 SSTM（BCH），相同测试集，只换前端方法

运行方式：
  python evaluate_comparison.py
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
from biohashing import BioHashing
from sstm_bch import SSTM_BCH


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH = "checkpoints/final_model.pth"
DATA_ROOT  = "/root/autodl-tmp/FVC2004"
DB_NAMES   = ["DB1_A/image", "DB1_B/image",
              "DB2_A/image", "DB2_B/image",
              "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR = "results_comparison"
G          = 512


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def extract_codes(model, loader, device):
    model.eval()
    codes, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, _, binary = model(imgs.to(device))
            codes.append(binary.cpu().numpy())
            labels.append(lbs.numpy())
    return np.vstack(codes), np.concatenate(labels)


def compute_distances(codes, labels, ctm):
    """
    计算 Genuine 和 Impostor（stolen key）汉明距离分布。
    用于实验二的 ROC 对比。
    """
    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)
    genuine_dists, impostor_dists = [], []

    # Genuine：同一用户，相同 key
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(codes[idx[0]])
        for i in idx[1:]:
            rp = ctm.authenticate(codes[i], ke)
            d = ctm.hamming_distance(re, rp)
            genuine_dists.append(d / ctm.G)

    # Impostor（stolen key）：不同用户，但用真实用户的 key
    n_imp = min(len(genuine_dists) * 5, 2000)
    for _ in range(n_imp):
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
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, fpr, tpr


def compute_gar_bch(codes, labels, ctm):
    """
    计算 BCH SSTM 下的 GAR（用于 G-S 曲线，实验三）。
    遍历 m=9 所有有效 t 值。
    """
    import bchlib
    bch_params = []
    for t in range(1, 57):
        try:
            bch_test = bchlib.BCH(t=t, m=9)
            k_bytes = (bch_test.n - bch_test.ecc_bits) // 8
            k_bits = k_bytes * 8
            if k_bits >= 40:
                bch_params.append((9, bch_test.t, k_bits))
        except Exception:
            break

    seen_k = {}
    for m, t, k in bch_params:
        if k not in seen_k or t > seen_k[k][1]:
            seen_k[k] = (m, t, k)
    bch_params = sorted(seen_k.values(), key=lambda x: x[2])

    unique_ids = np.unique(labels)
    k_bits_list, gars = [], []

    for m, t, k_bits in bch_params:
        sstm = SSTM_BCH(G=G, m=m, t=t)
        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm.enroll(codes[idx[0]])
            stored, _ = sstm.enroll(re)
            for i in idx[1:]:
                rp = ctm.authenticate(codes[i], ke)
                ok, _ = sstm.authenticate(rp, stored)
                pass_count += int(ok)
                total += 1
        gar = pass_count / total if total > 0 else 0.0
        gars.append(gar * 100)
        k_bits_list.append(k_bits)

    return k_bits_list, gars


# ─────────────────────────────────────────────
# 实验二：ROC 对比
# ─────────────────────────────────────────────

def run_exp2_roc(codes, labels, ctm_stable, ctm_bh, output_dir):
    """
    实验二：BioHashing vs StableCTM 的 ROC/EER 对比

    相同条件，只换前端方法，比较可分性（Genuine/Impostor 分布）。
    """
    print("\n" + "="*55)
    print("Exp 2: ROC Comparison (BioHashing vs StableCTM)")
    print("="*55)

    results = {}
    for name, ctm in [("StableCTM", ctm_stable), ("BioHashing", ctm_bh)]:
        print(f"\n[{name}] Computing distances...")
        gen, imp = compute_distances(codes, labels, ctm)
        eer, fpr, tpr = compute_eer(gen, imp)
        auc_val = auc(fpr, tpr)
        print(f"  Genuine:  mean={gen.mean():.3f}, std={gen.std():.3f}, "
              f"flip_rate={gen.mean()*100:.1f}%")
        print(f"  Impostor: mean={imp.mean():.3f}")
        print(f"  EER={eer*100:.2f}%  AUC={auc_val:.4f}")
        results[name] = {
            "genuine_dists": gen,
            "impostor_dists": imp,
            "eer": eer,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_val,
        }

    # 绘制 ROC 对比图
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"StableCTM": "blue", "BioHashing": "red"}
    styles = {"StableCTM": "-", "BioHashing": "--"}
    for name, r in results.items():
        fpr_pct = r["fpr"] * 100
        tpr_pct = r["tpr"] * 100
        ax.semilogx(fpr_pct, tpr_pct,
                    color=colors[name], linestyle=styles[name], linewidth=2,
                    label=f"{name} (AUC={r['auc']:.3f}, EER={r['eer']*100:.2f}%)")
        ax.scatter([r["eer"] * 100], [100 - r["eer"] * 100],
                   color=colors[name], s=60, zorder=5)

    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("GAR (%)")
    ax.set_title(f"ROC Curve: BioHashing vs StableCTM (G={G})\n"
                 f"[Stolen Key Scenario]")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    save_path = os.path.join(output_dir, f"roc_comparison_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nROC plot saved: {save_path}")
    plt.close()

    # 绘制距离分布对比图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.linspace(0, 1, 300)
    for ax, (name, r) in zip(axes, results.items()):
        gen, imp = r["genuine_dists"], r["impostor_dists"]
        ax.plot(x, norm.pdf(x, gen.mean(), gen.std()),
                "b-", linewidth=2, label="Genuine")
        ax.plot(x, norm.pdf(x, imp.mean(), imp.std()),
                "r--", linewidth=2, label="Impostor (Stolen Key)")
        ax.fill_between(x, norm.pdf(x, gen.mean(), gen.std()),
                        alpha=0.15, color="blue")
        ax.fill_between(x, norm.pdf(x, imp.mean(), imp.std()),
                        alpha=0.15, color="red")
        ax.set_xlabel("Normalized Hamming Distance")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"{name} (G={G})\n"
                     f"Genuine flip={gen.mean()*100:.1f}%  "
                     f"EER={r['eer']*100:.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Distance Distribution: BioHashing vs StableCTM", fontsize=13)
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, f"dist_comparison_G{G}.png")
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    print(f"Distribution plot saved: {save_path2}")
    plt.close()

    return results


# ─────────────────────────────────────────────
# 实验三：G-S 曲线对比
# ─────────────────────────────────────────────

def run_exp3_gs(codes, labels, ctm_stable, ctm_bh, output_dir):
    """
    实验三：BioHashing + BCH  vs  StableCTM + BCH 的 G-S 曲线对比

    相同 SSTM（BCH），只换前端方法，比较谁的 G-S 拐点更靠右。
    """
    print("\n" + "="*55)
    print("Exp 3: G-S Curve Comparison (BCH SSTM, G=512)")
    print("="*55)

    gs_results = {}
    for name, ctm in [("StableCTM", ctm_stable), ("BioHashing", ctm_bh)]:
        print(f"\n[{name}] Computing BCH G-S curve...")
        k_bits, gars = compute_gar_bch(codes, labels, ctm)
        gs_results[name] = {"k_bits": k_bits, "gars": gars}
        # 打印拐点
        inflect = [(k, g) for k, g in zip(k_bits, gars) if g >= 50]
        if inflect:
            print(f"  GAR=50% inflection: k={inflect[-1][0]} bits "
                  f"(GAR={inflect[-1][1]:.1f}%)")
        else:
            print(f"  GAR never reaches 50%")

    # 绘图
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"StableCTM": "blue", "BioHashing": "red"}
    markers = {"StableCTM": "s", "BioHashing": "^"}
    for name, r in gs_results.items():
        ax.plot(r["k_bits"], r["gars"],
                color=colors[name], marker=markers[name],
                linewidth=2, markersize=4, label=name)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="GAR=50%")
    ax.set_xlabel("Security Level k (bits)")
    ax.set_ylabel("GAR (%)")
    ax.set_title(f"G-S Curve: BioHashing vs StableCTM + BCH SSTM (G={G})")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)
    save_path = os.path.join(output_dir, f"gs_comparison_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nG-S plot saved: {save_path}")
    plt.close()

    return gs_results


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
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

    # 计算 StableCTM 翻转率
    print("\nExtracting training codes (for StableCTM flip rate)...")
    train_codes, train_labels = extract_codes(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_codes, train_labels)
    print(f"Flip rate: mean={flip_rate.mean()*100:.1f}%")

    # 提取测试集特征
    print("\nExtracting test codes...")
    codes, labels = extract_codes(model, test_loader, device)
    print(f"Codes shape: {codes.shape}")

    # 初始化两个前端方法
    ctm_stable = StableCTM(hash_dim=1024, G=G,
                            flip_rate=flip_rate, stable_ratio=0.8)
    ctm_bh = BioHashing(hash_dim=1024, G=G)

    # 实验二：ROC 对比
    roc_results = run_exp2_roc(codes, labels, ctm_stable, ctm_bh, OUTPUT_DIR)

    # 实验三：G-S 曲线对比
    gs_results = run_exp3_gs(codes, labels, ctm_stable, ctm_bh, OUTPUT_DIR)

    # 保存 JSON
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "exp2_roc": {
            name: {
                "genuine_flip_rate (%)": round(float(r["genuine_dists"].mean()) * 100, 2),
                "genuine_hamming_std":   round(float(r["genuine_dists"].std()), 4),
                "impostor_mean":         round(float(r["impostor_dists"].mean()), 4),
                "EER (%)":               round(r["eer"] * 100, 2),
                "AUC":                   round(r["auc"], 4),
            }
            for name, r in roc_results.items()
        },
        "exp3_gs": {
            name: {
                "k_bits": r["k_bits"],
                "GAR (%)": [round(g, 2) for g in r["gars"]],
            }
            for name, r in gs_results.items()
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {json_path}")

    # 打印汇总表
    print("\n" + "="*55)
    print("Summary (Exp 2 - ROC):")
    print(f"{'Method':<15} {'Flip Rate':>12} {'EER (%)':>10} {'AUC':>8}")
    print("-" * 48)
    for name, r in roc_results.items():
        print(f"{name:<15} "
              f"{r['genuine_dists'].mean()*100:>11.1f}%"
              f"{r['eer']*100:>10.2f}"
              f"{r['auc']:>8.4f}")

    print("\nSummary (Exp 3 - G-S inflection at GAR=50%):")
    for name, r in gs_results.items():
        inflect = [(k, g) for k, g in zip(r["k_bits"], r["gars"]) if g >= 50]
        if inflect:
            print(f"  {name}: k={inflect[-1][0]} bits (GAR={inflect[-1][1]:.1f}%)")
        else:
            print(f"  {name}: GAR never reaches 50%")


if __name__ == "__main__":
    main()
