"""
evaluate_polar_vs_bch.py — 极化码 vs BCH G-S 曲线对比实验

固定变量：G=512, StableCTM(stable_ratio=0.8), FVC2004
变化变量：SSTM 方案（BCH vs 极化码，不同安全性 k）

对比三条曲线：
  1. BCH(m=9)：现有方案（基准）
  2. Polar（标准极化码，巴氏参数选位）
  3. PolarEmbed（极化码+tanh置信度，`sstm_polar_embed.py`）

输出目录：results_polar_vs_bch/
运行方式：python evaluate_polar_vs_bch.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import StableCTM
from sstm_bch import SSTM_BCH
from sstm_polar import SSTM_Polar
from sstm_polar_embed import SSTM_PolarEmbed


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH = "checkpoints/final_model.pth"
DATA_ROOT  = "/root/autodl-tmp/FVC2004"
DB_NAMES   = ["DB1_A/image", "DB1_B/image",
              "DB2_A/image", "DB2_B/image",
              "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR = "results_polar_vs_bch"
G          = 512
STABLE_RATIO = 0.8
FLIP_PROB    = 0.112   # 服务器统计的 per-bit 翻转率均值


def extract_codes_with_embed(model, loader, device):
    """同时返回二值码和连续特征（tanh 输出）"""
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


def compute_gar_bch(binary_codes, labels, ctm, G):
    """BCH G-S 曲线：遍历 m=9 所有有效 t 值"""
    import bchlib
    bch_params = []
    for t in range(1, 57):
        try:
            b = bchlib.BCH(t=t, m=9)
            k_bytes = (b.n - b.ecc_bits) // 8
            k_bits = k_bytes * 8
            if k_bits >= 40:
                bch_params.append((9, b.t, k_bits))
        except Exception:
            break
    seen_k = {}
    for m, t, k in bch_params:
        if k not in seen_k or t > seen_k[k][1]:
            seen_k[k] = (m, t, k)
    bch_params = sorted(seen_k.values(), key=lambda x: x[2])

    unique_ids = np.unique(labels)
    k_bits_list, gars = [], []
    print(f"\n[BCH] G={G}...")
    for m, t, k_bits in bch_params:
        sstm = SSTM_BCH(G=G, m=m, t=t)
        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm.enroll(binary_codes[idx[0]])
            stored, _ = sstm.enroll(re)
            for i in idx[1:]:
                rp = ctm.authenticate(binary_codes[i], ke)
                ok, _ = sstm.authenticate(rp, stored)
                pass_count += int(ok)
                total += 1
        gar = pass_count / total if total > 0 else 0.0
        gars.append(gar * 100)
        k_bits_list.append(k_bits)
        print(f"  t={t:3d}  k={k_bits:4d} bits  GAR={gar*100:.1f}%")
    return k_bits_list, gars


def compute_gar_polar(binary_codes, labels, ctm, G, flip_prob,
                      hash_codes=None):
    """
    极化码 G-S 曲线：遍历不同 k 值（安全性）。

    hash_codes 传入时使用 SSTM_PolarEmbed（置信度融入），
    否则使用标准 SSTM_Polar。
    """
    # 遍历的 k 值（与 BCH 对齐，均为 8 的倍数）
    k_list = list(range(40, G, 8))

    unique_ids = np.unique(labels)
    k_bits_list, gars = [], []

    use_embed = hash_codes is not None
    label = "Polar+Embed" if use_embed else "Polar"
    print(f"\n[{label}] G={G}, flip_prob={flip_prob}...")

    for k_bits in k_list:
        if k_bits % 8 != 0 or k_bits >= G:
            continue

        try:
            if use_embed:
                sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits,
                                       m=9, t=56)
            else:
                sstm = SSTM_Polar(G=G, k=k_bits, flip_prob=flip_prob)
        except Exception as e:
            continue

        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm.enroll(binary_codes[idx[0]])

            if use_embed:
                embed_e = hash_codes[idx[0]][ke]
                stored, _ = sstm.enroll(re, embed_e)
            else:
                stored, _ = sstm.enroll(re)

            for i in idx[1:]:
                rp = ctm.authenticate(binary_codes[i], ke)
                if use_embed:
                    embed_p = hash_codes[i][ke]
                    ok, _ = sstm.authenticate(rp, stored, embed_p)
                else:
                    ok, _ = sstm.authenticate(rp, stored)
                pass_count += int(ok)
                total += 1

        gar = pass_count / total if total > 0 else 0.0
        gars.append(gar * 100)
        k_bits_list.append(k_bits)
        print(f"  k={k_bits:4d} bits  GAR={gar*100:.1f}%")

    return k_bits_list, gars


def plot_comparison(results, G, output_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        "BCH":         ("r-o",  "BCH Code (Baseline)"),
        "Polar":       ("b-s",  "Polar Code (Bhattacharyya)"),
        "PolarEmbed":  ("g-^",  "PolarEmbed (Confidence-guided)"),
    }
    for name, (style, label) in styles.items():
        if name not in results:
            continue
        r = results[name]
        ax.plot(r["k_bits"], r["GAR (%)"],
                style, linewidth=2, markersize=4, label=label)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S Curve: BCH vs Polar Code SSTM (G={G}, StableCTM)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    save_path = os.path.join(output_dir, f"polar_vs_bch_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()


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
    model = model.to(device)
    model.set_beta(32)

    # 提取训练集（用于 StableCTM 翻转率）
    print("\nExtracting training codes...")
    train_binary, _, train_labels = extract_codes_with_embed(
        model, train_loader, device
    )
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    flip_prob_mean = float(flip_rate.mean())
    print(f"Per-bit flip rate mean: {flip_prob_mean*100:.2f}%  "
          f"(used as BSC flip_prob for Polar Code)")

    # 提取测试集
    print("\nExtracting test codes (binary + hash)...")
    binary_codes, hash_codes, labels = extract_codes_with_embed(
        model, test_loader, device
    )
    print(f"Binary: {binary_codes.shape}, Hash: {hash_codes.shape}")

    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO)

    results = {}

    # ── BCH（基准）─────────────────────────────────
    print(f"\n{'='*55}\nMethod: BCH (Baseline)")
    k_bch, gars_bch = compute_gar_bch(binary_codes, labels, ctm, G)
    results["BCH"] = {
        "k_bits": k_bch,
        "GAR (%)": [round(g, 2) for g in gars_bch],
    }

    # ── 标准极化码─────────────────────────────────
    print(f"\n{'='*55}\nMethod: Polar Code (Standard)")
    k_polar, gars_polar = compute_gar_polar(
        binary_codes, labels, ctm, G,
        flip_prob=flip_prob_mean,
        hash_codes=None
    )
    results["Polar"] = {
        "k_bits": k_polar,
        "GAR (%)": [round(g, 2) for g in gars_polar],
    }

    # ── 极化码 + Embedding──────────────────────────
    print(f"\n{'='*55}\nMethod: PolarEmbed (Confidence-guided)")
    k_pe, gars_pe = compute_gar_polar(
        binary_codes, labels, ctm, G,
        flip_prob=flip_prob_mean,
        hash_codes=hash_codes
    )
    results["PolarEmbed"] = {
        "k_bits": k_pe,
        "GAR (%)": [round(g, 2) for g in gars_pe],
    }

    # ── 绘图 ──────────────────────────────────────
    plot_comparison(results, G, OUTPUT_DIR)

    # ── 保存 JSON ─────────────────────────────────
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "ctm": f"StableCTM(stable_ratio={STABLE_RATIO})",
        "flip_prob_mean": round(flip_prob_mean, 4),
        "results": results,
    }
    json_path = os.path.join(OUTPUT_DIR, f"polar_vs_bch_G{G}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")

    # ── 汇总表 ────────────────────────────────────
    print("\n" + "="*60)
    print("Summary (GAR=50% inflection):")
    for name, r in results.items():
        inflect = [(k, g) for k, g in zip(r["k_bits"], r["GAR (%)"]) if g >= 50]
        if inflect:
            print(f"  {name:<20}: k={inflect[-1][0]} bits "
                  f"(GAR={inflect[-1][1]:.1f}%)")
        else:
            print(f"  {name:<20}: GAR never reaches 50%")


if __name__ == "__main__":
    main()
