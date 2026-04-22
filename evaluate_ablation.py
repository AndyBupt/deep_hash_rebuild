"""
evaluate_ablation.py — CTM vs StableCTM Ablation Study (BCH SSTM)

消融实验：在相同 BCH SSTM 下，对比三种前端方法的 G-S 曲线：
  1. CTM (random bit selection)    — 随机选位基线
  2. StableCTM (stable pool)       — 稳定位选择（我们的改进）
  3. BioHashing                    — 经典对比方法

目的：量化稳定位选择（建议二）对 BCH G-S 曲线的实际贡献。

输出目录：results_ablation/
运行方式：python evaluate_ablation.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import CTM, StableCTM
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
OUTPUT_DIR = "results_ablation"
G          = 512


def extract_codes(model, loader, device):
    model.eval()
    codes, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, _, binary = model(imgs.to(device))
            codes.append(binary.cpu().numpy())
            labels.append(lbs.numpy())
    return np.vstack(codes), np.concatenate(labels)


def compute_gar_bch(codes, labels, ctm):
    """计算 BCH SSTM 下的 G-S 曲线（遍历 m=9 所有有效 t 值）"""
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


def compute_genuine_flip_rate(codes, labels, ctm):
    """计算 Genuine 汉明距离均值（翻转率）"""
    unique_ids = np.unique(labels)
    dists = []
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(codes[idx[0]])
        for i in idx[1:]:
            rp = ctm.authenticate(codes[i], ke)
            d = ctm.hamming_distance(re, rp) / ctm.G
            dists.append(d)
    return float(np.mean(dists))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(
        DATA_ROOT, DB_NAMES, train_ratio=0.7, batch_size=8
    )

    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024,
                               pretrained=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print("WARNING: Using random model (for testing only)")
    model = model.to(device)
    model.set_beta(32)

    # 计算翻转率（StableCTM 需要）
    print("\nExtracting training codes (for flip rate)...")
    train_codes, train_labels = extract_codes(model, train_loader, device)
    flip_rate = StableCTM.compute_flip_rate(train_codes, train_labels)
    print(f"Flip rate: mean={flip_rate.mean()*100:.1f}%")

    print("\nExtracting test codes...")
    codes, labels = extract_codes(model, test_loader, device)
    print(f"Codes shape: {codes.shape}")

    # 三种前端方法
    methods = {
        "CTM (Random)":   CTM(hash_dim=1024, G=G),
        "StableCTM":      StableCTM(hash_dim=1024, G=G,
                                    flip_rate=flip_rate, stable_ratio=0.8),
        "BioHashing":     BioHashing(hash_dim=1024, G=G),
    }
    colors  = {"CTM (Random)": "green",  "StableCTM": "blue",  "BioHashing": "red"}
    markers = {"CTM (Random)": "^",      "StableCTM": "s",     "BioHashing": "o"}

    results = {}

    for name, ctm in methods.items():
        print(f"\n{'='*55}")
        print(f"[{name}] Computing BCH G-S curve (G={G})...")

        flip = compute_genuine_flip_rate(codes, labels, ctm)
        print(f"  Genuine flip rate: {flip*100:.1f}%")

        k_bits, gars = compute_gar_bch(codes, labels, ctm)

        inflect = [(k, g) for k, g in zip(k_bits, gars) if g >= 50]
        if inflect:
            print(f"  GAR=50% inflection: k={inflect[-1][0]} bits (GAR={inflect[-1][1]:.1f}%)")
        else:
            print(f"  GAR never reaches 50%")

        results[name] = {
            "genuine_flip_rate (%)": round(flip * 100, 2),
            "k_bits": k_bits,
            "GAR (%)": [round(g, 2) for g in gars],
        }

    # ── G-S 曲线对比图 ────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in results.items():
        ax.plot(r["k_bits"], r["GAR (%)"],
                color=colors[name], marker=markers[name],
                linewidth=2, markersize=4,
                label=f"{name} (flip={r['genuine_flip_rate (%)']:.1f}%)")

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'Ablation Study: CTM vs StableCTM vs BioHashing + BCH SSTM (G={G})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    gs_path = os.path.join(OUTPUT_DIR, f"ablation_gs_G{G}.png")
    plt.savefig(gs_path, dpi=150, bbox_inches='tight')
    print(f"\nG-S ablation plot saved: {gs_path}")
    plt.close()

    # ── 保存 JSON ─────────────────────────────────
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "description": "Ablation: CTM vs StableCTM vs BioHashing, all with BCH SSTM",
        "results": results,
    }
    json_path = os.path.join(OUTPUT_DIR, f"ablation_results_G{G}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")

    # ── 汇总表 ────────────────────────────────────
    print("\n" + "="*60)
    print("Summary:")
    print(f"{'Method':<20} {'Flip Rate':>12} {'GAR=50% inflection':>20}")
    print("-" * 55)
    for name, r in results.items():
        inflect = [(k, g) for k, g in zip(r["k_bits"], r["GAR (%)"]) if g >= 50]
        infl_str = f"k={inflect[-1][0]} bits" if inflect else "N/A"
        print(f"{name:<20} {r['genuine_flip_rate (%)']:>11.1f}% {infl_str:>20}")


if __name__ == "__main__":
    main()
