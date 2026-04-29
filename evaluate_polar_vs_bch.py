"""
evaluate_polar_vs_bch.py — 极化码 vs BCH G-S 曲线对比实验

固定变量：G=512, StableCTM(stable_ratio=0.8), FVC2004
变化变量：SSTM 方案（BCH vs 极化码，不同安全性 k）

对比三条曲线：
  1. BCH(m=9)：现有方案（基准）
  2. Polar（标准极化码，巴氏参数选位 + SC 译码）
  3. PolarEmbed（置信度引导的模糊承诺，sstm_polar_embed.py）

PolarEmbed 与 BCH 使用相同的参数遍历方式（遍历 m=9 所有有效 t 值），
每个 t 对应一个安全性 k_bits，保证横轴可比。

输出目录：results_polar_vs_bch/
运行方式：python evaluate_polar_vs_bch.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import bchlib

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import StableCTM
from sstm_bch import SSTM_BCH
from sstm_polar import SSTM_Polar
from sstm_polar_embed import SSTM_PolarEmbed


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_polar_vs_bch"
G            = 512
STABLE_RATIO = 0.8


def _get_bch_params():
    """返回 m=9 所有有效 (m, t, k_bits) 组合，按 k_bits 升序排列"""
    params = []
    for t in range(1, 57):
        try:
            b = bchlib.BCH(t=t, m=9)
            k_bytes = (b.n - b.ecc_bits) // 8
            k_bits = k_bytes * 8
            if k_bits >= 40:
                params.append((9, b.t, k_bits))
        except Exception:
            break
    seen_k = {}
    for m, t, k in params:
        if k not in seen_k or t > seen_k[k][1]:
            seen_k[k] = (m, t, k)
    return sorted(seen_k.values(), key=lambda x: x[2])


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
    bch_params = _get_bch_params()
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


def compute_gar_polar_standard(binary_codes, labels, ctm, G, flip_prob):
    """
    标准极化码 G-S 曲线：遍历不同 k 值（步长 8）。
    使用 SSTM_Polar（巴氏参数选位 + SC 译码）。
    """
    unique_ids = np.unique(labels)
    k_bits_list, gars = [], []

    print(f"\n[Polar] G={G}, flip_prob={flip_prob:.4f}...")
    for k_bits in range(40, G, 8):
        try:
            sstm = SSTM_Polar(G=G, k=k_bits, flip_prob=flip_prob)
        except Exception:
            continue

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
        print(f"  k={k_bits:4d} bits  GAR={gar*100:.1f}%")
    return k_bits_list, gars


def compute_gar_polar_embed(binary_codes, hash_codes, labels, ctm, G):
    """
    PolarEmbed G-S 曲线：与 BCH 使用相同的参数遍历方式。
    遍历 m=9 所有有效 (m, t, k_bits) 组合，
    每组对应一个安全性 k_bits，横轴与 BCH 完全对齐。
    使用 SSTM_PolarEmbed（置信度引导的模糊承诺）。
    """
    bch_params = _get_bch_params()
    unique_ids = np.unique(labels)
    k_bits_list, gars = [], []

    print(f"\n[PolarEmbed] G={G}...")
    for m, t, k_bits in bch_params:
        try:
            sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
        except Exception as e:
            print(f"  skip k={k_bits}: {e}")
            continue

        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm.enroll(binary_codes[idx[0]])
            embed_e = hash_codes[idx[0]][ke]   # 注册时的 tanh 置信度
            stored, _ = sstm.enroll(re, embed_e)
            for i in idx[1:]:
                rp = ctm.authenticate(binary_codes[i], ke)
                embed_p = hash_codes[i][ke]    # 探针的 tanh 置信度
                ok, _ = sstm.authenticate(rp, stored, embed_p)
                pass_count += int(ok)
                total += 1

        gar = pass_count / total if total > 0 else 0.0
        gars.append(gar * 100)
        k_bits_list.append(k_bits)
        t_eff = sstm.get_effective_correction_capacity()
        print(f"  t={t:3d}  k={k_bits:4d} bits  "
              f"t_eff={t_eff:.0f}  GAR={gar*100:.1f}%")
    return k_bits_list, gars


def plot_comparison(results, G, output_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        "BCH":        ("r-o", "BCH Code (Baseline)"),
        "Polar":      ("b-s", "Polar Code (SC Decoder)"),
        "PolarEmbed": ("g-^", "PolarEmbed (Confidence-guided)"),
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

    # 训练集：用于 StableCTM 翻转率
    print("\nExtracting training codes...")
    train_binary, _, train_labels = extract_codes_with_embed(
        model, train_loader, device
    )
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    flip_prob_mean = float(flip_rate.mean())
    print(f"Per-bit flip rate mean: {flip_prob_mean*100:.2f}%")

    # 测试集：二值码 + 连续特征
    print("\nExtracting test codes (binary + hash)...")
    binary_codes, hash_codes, labels = extract_codes_with_embed(
        model, test_loader, device
    )
    print(f"Binary: {binary_codes.shape}, Hash: {hash_codes.shape}")

    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO)
    results = {}

    # ── BCH（基准）────────────────────────────────
    print(f"\n{'='*55}\nMethod 1: BCH (Baseline)")
    k_bch, gars_bch = compute_gar_bch(binary_codes, labels, ctm, G)
    results["BCH"] = {
        "k_bits": k_bch,
        "GAR (%)": [round(g, 2) for g in gars_bch],
    }

    # ── 标准极化码──────────────────────────────────
    print(f"\n{'='*55}\nMethod 2: Polar Code (Standard, SC Decoder)")
    k_polar, gars_polar = compute_gar_polar_standard(
        binary_codes, labels, ctm, G, flip_prob_mean
    )
    results["Polar"] = {
        "k_bits": k_polar,
        "GAR (%)": [round(g, 2) for g in gars_polar],
    }

    # ── PolarEmbed───────────────────────────────────
    print(f"\n{'='*55}\nMethod 3: PolarEmbed (Confidence-guided)")
    k_pe, gars_pe = compute_gar_polar_embed(
        binary_codes, hash_codes, labels, ctm, G
    )
    results["PolarEmbed"] = {
        "k_bits": k_pe,
        "GAR (%)": [round(g, 2) for g in gars_pe],
    }

    # ── 绘图 ─────────────────────────────────────
    plot_comparison(results, G, OUTPUT_DIR)

    # ── 保存 JSON ────────────────────────────────
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

    # ── 汇总表 ───────────────────────────────────
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
