"""
evaluate_scl.py — SCL+CRC 极化码 vs BCH vs PolarEmbed G-S 曲线对比

固定变量：G=512, StableCTM(stable_ratio=0.8), FVC2004
变化变量：SSTM 方案（BCH vs PolarEmbed vs SCL+CRC，不同安全性 k）

对比曲线：
  1. BCH(m=9)：基准
  2. PolarEmbed：置信度引导模糊承诺（已证最优）
  3. SCL L=8 CRC-8：真正的极化码 SCL 译码

输出目录：results_scl/
运行方式：python evaluate_scl.py
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
from sstm_polar_embed import SSTM_PolarEmbed
from sstm_polar_scl import SSTM_PolarSCL


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH   = "checkpoints/final_model.pth"
DATA_ROOT    = "/root/autodl-tmp/FVC2004"
DB_NAMES     = ["DB1_A/image", "DB1_B/image",
                "DB2_A/image", "DB2_B/image",
                "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR   = "results_scl"
G            = 512
STABLE_RATIO = 0.8
SCL_L        = 8    # SCL 列表大小
SCL_CRC      = 8    # CRC 位数


def _get_bch_params():
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


def compute_gar_polar_embed(binary_codes, hash_codes, labels, ctm, G):
    bch_params = _get_bch_params()
    unique_ids = np.unique(labels)
    k_bits_list, gars = [], []
    print(f"\n[PolarEmbed] G={G}...")
    for m, t, k_bits in bch_params:
        try:
            sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
        except Exception as e:
            continue
        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm.enroll(binary_codes[idx[0]])
            embed_e = hash_codes[idx[0]][ke]
            stored, _ = sstm.enroll(re, embed_e)
            for i in idx[1:]:
                rp = ctm.authenticate(binary_codes[i], ke)
                embed_p = hash_codes[i][ke]
                ok, _ = sstm.authenticate(rp, stored, embed_p)
                pass_count += int(ok)
                total += 1
        gar = pass_count / total if total > 0 else 0.0
        gars.append(gar * 100)
        k_bits_list.append(k_bits)
        t_eff = sstm.get_effective_correction_capacity()
        print(f"  t={t:3d}  k={k_bits:4d} bits  t_eff={t_eff:.0f}  GAR={gar*100:.1f}%")
    return k_bits_list, gars


def compute_gar_scl(binary_codes, hash_codes, labels, ctm, G,
                    flip_prob, L=8, crc_bits=8):
    """
    SCL+CRC G-S 曲线。

    参数对齐说明：
      - 横轴 k_bits 表示"密钥长度"（即安全性），与 BCH 保持一致
      - actual_k = k_bits（密钥长度，不减 CRC）
      - 极化码信息位 = actual_k + crc_bits（密钥 + CRC）
      - 这样横轴才是真正可比的安全性 k

    特殊情况：
      - L=1, crc_bits=0 时等价于标准 SC 译码，可用于验证代码一致性
    """
    bch_params = _get_bch_params()
    unique_ids = np.unique(labels)
    k_bits_list, gars = [], []

    label_str = f"SCL L={L}" + (f" CRC-{crc_bits}" if crc_bits > 0 else " no-CRC")
    print(f"\n[{label_str}] G={G}...")

    for m, t, k_bits in bch_params:
        # actual_k = k_bits（密钥长度与 BCH 对齐）
        # 极化码总信息位 = actual_k + crc_bits
        actual_k = k_bits
        k_total = actual_k + crc_bits
        if k_total >= G:
            continue
        try:
            sstm = SSTM_PolarSCL(G=G, k=actual_k, flip_prob=flip_prob,
                                  L=L, crc_bits=crc_bits)
        except Exception as e:
            print(f"  skip k={k_bits}: {e}")
            continue

        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue
            re, ke = ctm.enroll(binary_codes[idx[0]])
            embed_e = hash_codes[idx[0]][ke]
            stored, _ = sstm.enroll(re, embed_e)
            for i in idx[1:]:
                rp = ctm.authenticate(binary_codes[i], ke)
                embed_p = hash_codes[i][ke]
                ok, _ = sstm.authenticate(rp, stored, embed_p)
                pass_count += int(ok)
                total += 1

        gar = pass_count / total if total > 0 else 0.0
        gars.append(gar * 100)
        k_bits_list.append(k_bits)
        print(f"  t={t:3d}  k={k_bits:4d} bits  "
              f"(k_total={k_total})  GAR={gar*100:.1f}%")

    return k_bits_list, gars


def plot_comparison(results, G, output_dir, scl_L, scl_crc):
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        "BCH":          ("r-o",  "BCH Code (Baseline)"),
        "PolarEmbed":   ("g-^",  "PolarEmbed (Confidence-guided)"),
        "SCL_L1_noCRC": ("k--",  "SCL L=1 no-CRC (≡ SC, sanity check)"),
        "SCL_CRC":      ("b-s",  f"Polar SCL(L={scl_L})+CRC-{scl_crc}"),
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
    ax.set_title(f'G-S Curve: BCH vs PolarEmbed vs SCL+CRC (G={G}, StableCTM)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    save_path = os.path.join(output_dir, f"scl_vs_bch_G{G}.png")
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

    print("\nExtracting training codes...")
    train_binary, _, train_labels = extract_codes_with_embed(
        model, train_loader, device
    )
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    flip_prob_mean = float(flip_rate.mean())
    print(f"Per-bit flip rate mean: {flip_prob_mean*100:.2f}%")

    print("\nExtracting test codes...")
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

    # ── PolarEmbed（置信度融合，已证最优）──────────
    print(f"\n{'='*55}\nMethod 2: PolarEmbed (Confidence-guided)")
    k_pe, gars_pe = compute_gar_polar_embed(
        binary_codes, hash_codes, labels, ctm, G
    )
    results["PolarEmbed"] = {
        "k_bits": k_pe,
        "GAR (%)": [round(g, 2) for g in gars_pe],
    }

    # ── SCL(L=1, no CRC) 验证基准（理论上 = SC 译码）──
    print(f"\n{'='*55}\nMethod 3a: SCL L=1 no-CRC (sanity check, should ≈ SC)")
    k_sc, gars_sc = compute_gar_scl(
        binary_codes, hash_codes, labels, ctm, G,
        flip_prob=flip_prob_mean, L=1, crc_bits=0
    )
    results["SCL_L1_noCRC"] = {
        "k_bits": k_sc,
        "GAR (%)": [round(g, 2) for g in gars_sc],
        "note": "L=1 no-CRC is equivalent to SC decoding"
    }

    # ── SCL + CRC（真正极化码，参数对齐版本）──────
    print(f"\n{'='*55}\nMethod 3b: Polar SCL(L={SCL_L})+CRC-{SCL_CRC} (aligned k)")
    k_scl, gars_scl = compute_gar_scl(
        binary_codes, hash_codes, labels, ctm, G,
        flip_prob=flip_prob_mean, L=SCL_L, crc_bits=SCL_CRC
    )
    results["SCL_CRC"] = {
        "k_bits": k_scl,
        "GAR (%)": [round(g, 2) for g in gars_scl],
        "note": f"k_bits = actual key length (security), k_total = k + {SCL_CRC}"
    }

    # ── 绘图 ─────────────────────────────────────
    plot_comparison(results, G, OUTPUT_DIR, SCL_L, SCL_CRC)

    # ── 保存 JSON ────────────────────────────────
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "ctm": f"StableCTM(stable_ratio={STABLE_RATIO})",
        "flip_prob_mean": round(flip_prob_mean, 4),
        "scl_L": SCL_L,
        "scl_crc_bits": SCL_CRC,
        "results": results,
    }
    json_path = os.path.join(OUTPUT_DIR, f"scl_vs_bch_G{G}.json")
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
