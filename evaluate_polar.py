"""
evaluate_polar.py — Polar+Embed SSTM vs BCH SSTM 对比实验

实验设计：
  固定变量：G=512, StableCTM(stable_ratio=0.8), FVC2004 数据集
  变化变量：SSTM 方案（BCH vs PolarEmbed，遍历不同 k_bits）

对比曲线：
  1. BCH(m=9) + StableCTM：现有方案（基准）
  2. PolarEmbed + StableCTM：新方案（使用 tanh 置信度选可靠信道）
  3. PolarEmbed(no embed) + StableCTM：消融（embed=None，退化为随机选位）

输出目录：results_polar/（全新目录，不影响现有结果）

注意：本脚本完全独立，不修改任何现有脚本。

运行方式：
  python evaluate_polar.py
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
from sstm_polar_embed import SSTM_PolarEmbed


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH = "checkpoints/final_model.pth"
DATA_ROOT  = "/root/autodl-tmp/FVC2004"
DB_NAMES   = ["DB1_A/image", "DB1_B/image",
              "DB2_A/image", "DB2_B/image",
              "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR = "results_polar"
G          = 512
STABLE_RATIO = 0.8


# ─────────────────────────────────────────────
# 特征提取（同时返回连续特征）
# ─────────────────────────────────────────────

def extract_codes_with_embed(model, loader, device):
    """
    同时提取二值码和连续特征（tanh 输出）。

    Returns:
        binary_codes: (N, 1024)  sign(tanh) 二值码，{-1,+1}
        hash_codes:   (N, 1024)  tanh 连续值，范围 (-1,+1)，作为置信度
        labels:       (N,)
    """
    model.eval()
    all_binary, all_hash, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            imgs = imgs.to(device)
            _, hash_c, binary_c = model(imgs)
            all_binary.append(binary_c.cpu().numpy())
            all_hash.append(hash_c.cpu().numpy())
            all_labels.append(lbs.numpy())
    return (np.vstack(all_binary),
            np.vstack(all_hash),
            np.concatenate(all_labels))


# ─────────────────────────────────────────────
# GAR 计算
# ─────────────────────────────────────────────

def compute_gar_bch(binary_codes, labels, ctm, G):
    """
    BCH 方案的 G-S 曲线：遍历 m=9 所有有效 t 值。
    与 evaluate_ablation.py 中的逻辑相同，但独立实现。
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
        t_eff = sstm.get_effective_correction_capacity()
        print(f"  m={m} t={t:3d}  k={k_bits:4d} bits  "
              f"t_eff={t_eff:.0f}  GAR={gar*100:.1f}%")

    return k_bits_list, gars


def compute_gar_polar(binary_codes, hash_codes, labels, ctm, G,
                      use_embed=True):
    """
    PolarEmbed 方案的 G-S 曲线：遍历不同 k_bits。

    Args:
        use_embed: True 时使用 tanh 置信度（完整方案），
                   False 时 embed=None（退化为随机选位，消融实验）
    """
    import bchlib

    # 与 BCH 使用相同的 k_bits 范围，方便对比
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

    label_str = "PolarEmbed" if use_embed else "PolarEmbed(no embed)"
    print(f"\n[{label_str}] G={G}...")

    for m, t, k_bits in bch_params:
        # k_bits 必须是 8 的倍数（SSTM_PolarEmbed 要求）
        if k_bits % 8 != 0:
            continue

        try:
            sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits, m=m, t=t)
        except AssertionError as e:
            print(f"  k={k_bits}: skip ({e})")
            continue

        pass_count = total = 0
        for uid in unique_ids:
            idx = np.where(labels == uid)[0]
            if len(idx) < 2:
                continue

            # 注册：用注册样本的 tanh 值作为 embed_e
            re, ke = ctm.enroll(binary_codes[idx[0]])
            if use_embed:
                embed_e = hash_codes[idx[0]][ke]  # 选出位置的 tanh 值
            else:
                embed_e = None

            stored, _ = sstm.enroll(re, embed_e)

            for i in idx[1:]:
                rp = ctm.authenticate(binary_codes[i], ke)
                if use_embed:
                    embed_p = hash_codes[i][ke]   # 探针的 tanh 值
                else:
                    embed_p = None
                ok, _ = sstm.authenticate(rp, stored, embed_p)
                pass_count += int(ok)
                total += 1

        gar = pass_count / total if total > 0 else 0.0
        gars.append(gar * 100)
        k_bits_list.append(k_bits)
        t_eff = sstm.get_effective_correction_capacity()
        print(f"  m={m} t={t:3d}  k={k_bits:4d} bits  "
              f"t_eff={t_eff:.0f}  GAR={gar*100:.1f}%")

    return k_bits_list, gars


# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────

def plot_comparison(results, G, output_dir):
    """绘制多方案 G-S 曲线对比图"""
    fig, ax = plt.subplots(figsize=(12, 7))

    styles = {
        "BCH":                    ("r-o",  "BCH Code (Baseline)"),
        "PolarEmbed":             ("b-s",  "PolarEmbed (with tanh confidence)"),
        "PolarEmbed(no embed)":   ("g-^",  "PolarEmbed (no embed, ablation)"),
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
    ax.set_title(f'G-S Curve: BCH vs PolarEmbed SSTM (G={G}, StableCTM)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    save_path = os.path.join(output_dir, f"polar_vs_bch_G{G}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()


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
    train_binary, _, train_labels = extract_codes_with_embed(
        model, train_loader, device
    )
    flip_rate = StableCTM.compute_flip_rate(train_binary, train_labels)
    print(f"Flip rate: mean={flip_rate.mean()*100:.1f}%")

    # 提取测试集特征（二值码 + 连续特征）
    print("\nExtracting test codes (binary + hash)...")
    binary_codes, hash_codes, labels = extract_codes_with_embed(
        model, test_loader, device
    )
    print(f"Binary codes shape: {binary_codes.shape}")
    print(f"Hash codes shape:   {hash_codes.shape}")
    print(f"  hash_codes range: [{hash_codes.min():.3f}, {hash_codes.max():.3f}]")
    print(f"  |hash_codes| mean: {np.abs(hash_codes).mean():.3f}")

    # 初始化 StableCTM
    ctm = StableCTM(hash_dim=1024, G=G,
                    flip_rate=flip_rate, stable_ratio=STABLE_RATIO)

    results = {}

    # ── 方案1：BCH（基准）────────────────────────────
    print(f"\n{'='*55}")
    print("Method 1: BCH (Baseline)")
    k_bch, gars_bch = compute_gar_bch(binary_codes, labels, ctm, G)
    results["BCH"] = {
        "k_bits": k_bch,
        "GAR (%)": [round(g, 2) for g in gars_bch],
    }

    # ── 方案2：PolarEmbed（完整方案）────────────────────
    print(f"\n{'='*55}")
    print("Method 2: PolarEmbed (with tanh confidence)")
    k_pe, gars_pe = compute_gar_polar(
        binary_codes, hash_codes, labels, ctm, G, use_embed=True
    )
    results["PolarEmbed"] = {
        "k_bits": k_pe,
        "GAR (%)": [round(g, 2) for g in gars_pe],
    }

    # ── 方案3：PolarEmbed 消融（无 embed）───────────────
    print(f"\n{'='*55}")
    print("Method 3: PolarEmbed (no embed, ablation)")
    k_pe_no, gars_pe_no = compute_gar_polar(
        binary_codes, hash_codes, labels, ctm, G, use_embed=False
    )
    results["PolarEmbed(no embed)"] = {
        "k_bits": k_pe_no,
        "GAR (%)": [round(g, 2) for g in gars_pe_no],
    }

    # ── 绘图 ──────────────────────────────────────────
    plot_comparison(results, G, OUTPUT_DIR)

    # ── 保存 JSON ─────────────────────────────────────
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "ctm": f"StableCTM(stable_ratio={STABLE_RATIO})",
        "description": "BCH vs PolarEmbed SSTM comparison",
        "results": results,
    }
    json_path = os.path.join(OUTPUT_DIR, f"polar_vs_bch_G{G}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")

    # ── 汇总表 ────────────────────────────────────────
    print("\n" + "="*60)
    print("Summary (GAR=50% inflection):")
    for name, r in results.items():
        inflect = [(k, g) for k, g in zip(r["k_bits"], r["GAR (%)"]) if g >= 50]
        if inflect:
            print(f"  {name:<30}: k={inflect[-1][0]} bits "
                  f"(GAR={inflect[-1][1]:.1f}%)")
        else:
            print(f"  {name:<30}: GAR never reaches 50%")

    # 打印置信度分析
    print("\n" + "="*60)
    print("Confidence Analysis (|tanh| statistics):")
    print(f"  |tanh| mean:   {np.abs(hash_codes).mean():.4f}")
    print(f"  |tanh| median: {np.median(np.abs(hash_codes)):.4f}")
    print(f"  |tanh| > 0.5:  {(np.abs(hash_codes) > 0.5).mean()*100:.1f}% of bits")
    print(f"  |tanh| > 0.8:  {(np.abs(hash_codes) > 0.8).mean()*100:.1f}% of bits")
    print(f"  (Higher |tanh| = more reliable channel)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
