"""
evaluate_bch.py — BCH vs RS G-S Curve Comparison

对比实验：
  - RS 码（sstm.py）：按符号纠错，符号错误率 ~84%，GAR 接近 0
  - BCH 码（sstm_bch.py）：按比特纠错，有效纠错能力更强，GAR 预期更高

支持：
  - 多个 G 值（128, 256, 512）
  - 两种场景（stolen_key, unknown_key）
  - 每个 G 值分别输出 PNG + JSON

运行方式：
  python evaluate_bch.py
"""

import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import build_dataloaders
from model import FingerprintHashNet
from ctm import CTM
from sstm import SSTM
from sstm_bch import SSTM_BCH


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_PATH  = "checkpoints/final_model.pth"
DATA_ROOT   = "/root/autodl-tmp/FVC2004"
DB_NAMES    = ["DB1_A/image", "DB1_B/image",
               "DB2_A/image", "DB2_B/image",
               "DB3_A/image", "DB3_B/image"]
OUTPUT_DIR  = "results_bch"
G_VALUES    = [128, 256, 512]          # 评估的 G 值列表
SCENARIOS   = ["stolen_key", "unknown_key"]   # 两种场景都跑


def extract_codes(model, loader, device):
    model.eval()
    codes, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, _, binary = model(imgs.to(device))
            codes.append(binary.cpu().numpy())
            labels.append(lbs.numpy())
    return np.vstack(codes), np.concatenate(labels)


def compute_gar(codes, labels, ctm, sstm_obj, scenario):
    """计算给定 CTM + SSTM 组合的 GAR。"""
    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)
    pass_count = total = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(codes[idx[0]])
        stored, _ = sstm_obj.enroll(re)
        for i in idx[1:]:
            if scenario == "unknown_key":
                _, ke_rand = ctm.enroll(codes[i],
                                        seed=int(rng.integers(0, 99999)))
                rp = ctm.authenticate(codes[i], ke_rand)
            else:
                rp = ctm.authenticate(codes[i], ke)
            ok, _ = sstm_obj.authenticate(rp, stored)
            pass_count += int(ok)
            total += 1
    return pass_count / total if total > 0 else 0.0


def run_rs_gs_curve(codes, labels, ctm, G, scenario):
    """RS 码 G-S 曲线：遍历 K 值（安全性 k = K*8 bits）"""
    N = G // 8
    K_values = list(range(7, N, 2))
    k_bits_list, gars = [], []

    print(f"\n[RS] G={G}, scenario={scenario}...")
    for K in K_values:
        if G // 8 <= K:
            gars.append(0.0)
            k_bits_list.append(K * 8)
            continue
        sstm = SSTM(G=G, K=K)
        gar = compute_gar(codes, labels, ctm, sstm, scenario)
        gars.append(gar * 100)
        k_bits_list.append(K * 8)
        print(f"  K={K:3d}  k={K*8:4d} bits  GAR={gar*100:.1f}%")

    return k_bits_list, gars


def run_bch_gs_curve(codes, labels, ctm, G, scenario):
    """BCH 码 G-S 曲线：遍历 m=9 的所有有效 t 值"""
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

    # 去重（k_bits 相同只保留 t 最大的）
    seen_k = {}
    for m, t, k in bch_params:
        if k not in seen_k or t > seen_k[k][1]:
            seen_k[k] = (m, t, k)
    bch_params = sorted(seen_k.values(), key=lambda x: x[2])

    k_bits_list, gars = [], []

    print(f"\n[BCH] G={G}, scenario={scenario}...")
    for m, t, k_bits in bch_params:
        try:
            sstm_bch = SSTM_BCH(G=G, m=m, t=t)
            gar = compute_gar(codes, labels, ctm, sstm_bch, scenario)
            gars.append(gar * 100)
            k_bits_list.append(k_bits)
            t_eff = sstm_bch.get_effective_correction_capacity()
            print(f"  m={m} t={t:3d}  k={k_bits:4d} bits  "
                  f"t_eff={t_eff:.0f}  GAR={gar*100:.1f}%")
        except Exception as e:
            print(f"  m={m} t={t}: skip ({e})")

    return k_bits_list, gars


def plot_rs_vs_bch(k_rs, gars_rs, k_bch, gars_bch, G, scenario, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(k_rs,  gars_rs,  'r-o', linewidth=2, markersize=4,
            label='RS Code (Symbol-level ECC)')
    ax.plot(k_bch, gars_bch, 'b-s', linewidth=2, markersize=4,
            label='BCH Code (Bit-level ECC)')

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    scenario_label = "Stolen Key" if scenario == "stolen_key" else "Unknown Key"
    ax.set_title(f'G-S Curve: RS vs BCH (G={G} bits, {scenario_label})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def run_one(codes, labels, G, scenario, output_dir):
    """跑单个 (G, scenario) 组合，保存图和 JSON。"""
    ctm = CTM(hash_dim=1024, G=G)

    k_rs,  gars_rs  = run_rs_gs_curve(codes, labels, ctm, G, scenario)
    k_bch, gars_bch = run_bch_gs_curve(codes, labels, ctm, G, scenario)

    # 绘图
    plot_rs_vs_bch(
        k_rs, gars_rs, k_bch, gars_bch, G, scenario,
        save_path=os.path.join(output_dir,
                               f"gs_rs_vs_bch_G{G}_{scenario}.png")
    )

    # 保存 JSON
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "scenario": scenario,
        "RS":  {"k_bits": k_rs,  "GAR (%)": [round(g, 2) for g in gars_rs]},
        "BCH": {"k_bits": k_bch, "GAR (%)": [round(g, 2) for g in gars_bch]},
    }
    json_path = os.path.join(output_dir,
                             f"gs_rs_vs_bch_G{G}_{scenario}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")

    return k_rs, gars_rs, k_bch, gars_bch


def plot_all_G_comparison(all_results, scenario, output_dir):
    """把同一 scenario 下所有 G 值的 BCH 曲线画在一张图上，方便对比。"""
    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 6),
                             sharey=True)
    if len(all_results) == 1:
        axes = [axes]

    scenario_label = "Stolen Key" if scenario == "stolen_key" else "Unknown Key"

    for ax, (G, k_rs, gars_rs, k_bch, gars_bch) in zip(axes, all_results):
        ax.plot(k_rs,  gars_rs,  'r-o', linewidth=2, markersize=3,
                label='RS Code')
        ax.plot(k_bch, gars_bch, 'b-s', linewidth=2, markersize=3,
                label='BCH Code')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Security Level k (bits)')
        ax.set_ylabel('GAR (%)' if G == all_results[0][0] else '')
        ax.set_title(f'G={G} bits')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 108)

    fig.suptitle(f'G-S Curve: RS vs BCH ({scenario_label})', fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"gs_rs_vs_bch_all_{scenario}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    _, test_loader, num_classes = build_dataloaders(
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

    # 提取特征（只提取一次，所有实验共用）
    print("\nExtracting binary codes...")
    codes, labels = extract_codes(model, test_loader, device)
    print(f"Codes shape: {codes.shape}")

    # 遍历所有 scenario 和 G 值
    for scenario in SCENARIOS:
        print(f"\n{'='*55}")
        print(f"Scenario: {scenario}")
        print(f"{'='*55}")

        all_results = []
        for G in G_VALUES:
            print(f"\n--- G={G} ---")
            k_rs, gars_rs, k_bch, gars_bch = run_one(
                codes, labels, G, scenario, OUTPUT_DIR
            )
            all_results.append((G, k_rs, gars_rs, k_bch, gars_bch))

        # 多 G 值汇总图
        plot_all_G_comparison(all_results, scenario, OUTPUT_DIR)

    print("\nAll done.")


if __name__ == "__main__":
    main()
