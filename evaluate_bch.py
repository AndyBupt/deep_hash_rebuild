"""
evaluate_bch.py — BCH vs RS G-S Curve Comparison

对比实验：
  - RS 码（sstm.py）：按符号纠错，符号错误率 ~84%，GAR 接近 0
  - BCH 码（sstm_bch.py）：按比特纠错，有效纠错能力更强，GAR 预期更高

G-S 曲线横轴：安全性 k（bits）
G-S 曲线纵轴：GAR（%）

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
G           = 512
SCENARIO    = "stolen_key"   # stolen_key 或 unknown_key


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


def run_rs_gs_curve(codes, labels, ctm, G):
    """
    RS 码 G-S 曲线：遍历 K 值（安全性 k = K*8 bits）
    K 范围：7 到 N-1（步长 2），N = G//8
    """
    N = G // 8
    K_values = list(range(7, N, 2))
    k_bits_list, gars = [], []

    print(f"\n[RS] Computing G-S curve (G={G}, scenario={SCENARIO})...")
    for K in K_values:
        if G // 8 <= K:
            gars.append(0.0)
            k_bits_list.append(K * 8)
            continue
        sstm = SSTM(G=G, K=K)
        gar = compute_gar(codes, labels, ctm, sstm, SCENARIO)
        gars.append(gar * 100)
        k_bits_list.append(K * 8)
        print(f"  K={K:3d}  k={K*8:4d} bits  GAR={gar*100:.1f}%")

    return k_bits_list, gars


def run_bch_gs_curve(codes, labels, ctm, G):
    """
    BCH 码 G-S 曲线：遍历不同 (m, t) 参数组合
    每组参数对应一个安全性 k = k_bytes * 8 bits
    """
    # 参数组合：(m, t_request)
    # 选取 t 从小到大，覆盖从高安全低纠错到低安全高纠错
    bch_params = []
    import bchlib
    for m in [9]:
        for t in range(1, 57):
            try:
                bch_test = bchlib.BCH(t=t, m=m)
                k_bytes = (bch_test.n - bch_test.ecc_bits) // 8
                k_bits = k_bytes * 8
                if k_bits >= 40:   # 安全性至少 40 bits
                    bch_params.append((m, bch_test.t, k_bits))
            except Exception:
                break

    # 去重（k_bits 相同的只保留 t 最大的）
    seen_k = {}
    for m, t, k in bch_params:
        if k not in seen_k or t > seen_k[k][1]:
            seen_k[k] = (m, t, k)
    bch_params = sorted(seen_k.values(), key=lambda x: x[2])  # 按 k 排序

    k_bits_list, gars = [], []

    print(f"\n[BCH] Computing G-S curve (G={G}, scenario={SCENARIO})...")
    for m, t, k_bits in bch_params:
        try:
            sstm_bch = SSTM_BCH(G=G, m=m, t=t)
            gar = compute_gar(codes, labels, ctm, sstm_bch, SCENARIO)
            gars.append(gar * 100)
            k_bits_list.append(k_bits)
            t_eff = sstm_bch.get_effective_correction_capacity()
            print(f"  m={m} t={t:3d}  k={k_bits:4d} bits  "
                  f"t_eff={t_eff:.0f}  GAR={gar*100:.1f}%")
        except Exception as e:
            print(f"  m={m} t={t}: skip ({e})")

    return k_bits_list, gars


def plot_rs_vs_bch(k_rs, gars_rs, k_bch, gars_bch, G, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(k_rs,  gars_rs,  'r-o', linewidth=2, markersize=4,
            label='RS Code (Symbol-level ECC)')
    ax.plot(k_bch, gars_bch, 'b-s', linewidth=2, markersize=4,
            label='BCH Code (Bit-level ECC)')

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
    ax.set_xlabel('Security Level k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S Curve: RS vs BCH Error Correction (G={G} bits, {SCENARIO})')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


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

    # 提取特征
    print("\nExtracting binary codes...")
    codes, labels = extract_codes(model, test_loader, device)
    print(f"Codes shape: {codes.shape}")

    ctm = CTM(hash_dim=1024, G=G)

    # RS G-S 曲线
    k_rs, gars_rs = run_rs_gs_curve(codes, labels, ctm, G)

    # BCH G-S 曲线
    k_bch, gars_bch = run_bch_gs_curve(codes, labels, ctm, G)

    # 绘图
    plot_rs_vs_bch(
        k_rs, gars_rs, k_bch, gars_bch, G,
        save_path=os.path.join(OUTPUT_DIR, f"gs_rs_vs_bch_G{G}.png")
    )

    # 保存数值结果
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "G": G,
        "scenario": SCENARIO,
        "RS": {"k_bits": k_rs, "GAR (%)": [round(g, 2) for g in gars_rs]},
        "BCH": {"k_bits": k_bch, "GAR (%)": [round(g, 2) for g in gars_bch]},
    }
    json_path = os.path.join(OUTPUT_DIR, f"gs_rs_vs_bch_G{G}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
