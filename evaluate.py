"""
评估脚本
对应论文 Section V 和 Section VI

评估指标:
  - EER (Equal Error Rate，等错误率)
  - GAR@FAR (Genuine Accept Rate at given False Accept Rate)
  - ROC 曲线
  - Genuine/Impostor 汉明距离分布
  - G-S 曲线（GAR vs 安全性 bits）
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'  # macOS 中文支持

from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

from dataset import build_dataloaders, get_transforms
from model import FingerprintHashNet
from ctm import CTM
from sstm import SSTM


def extract_all_binary_codes(model, loader, device):
    """
    提取所有测试样本的二值哈希码和标签

    Returns:
        codes: (N, hash_dim) numpy array，值为 {-1, +1}
        labels: (N,) numpy array
    """
    model.eval()
    all_codes = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, _, binary = model(imgs)
            all_codes.append(binary.cpu().numpy())
            all_labels.append(labels.numpy())

    codes = np.vstack(all_codes)
    labels = np.concatenate(all_labels)
    return codes, labels


def compute_genuine_impostor_distances(codes, labels, ctm, scenario="unknown_key"):
    """
    计算所有 genuine pair 和 impostor pair 的汉明距离

    对应论文 Section V-B 的两种场景:
      unknown_key: impostor 不知道 genuine 的密钥，使用随机密钥
      stolen_key:  impostor 知道 genuine 的密钥，但提交自己的生物特征

    Args:
        scenario: "unknown_key" 或 "stolen_key"
    """
    genuine_dists = []
    impostor_dists = []
    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)

    # Genuine pairs: 同一身份内，用第一个样本注册，其余样本认证
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
        # Unknown key 场景: impostor 用随机密钥 + 自己的生物特征
        # 论文: "The impostor tries to present random indices for random-bit selection"
        for _ in range(n_impostors):
            id1, id2 = rng.choice(unique_ids, size=2, replace=False)
            idx1 = rng.choice(np.where(labels == id1)[0])
            idx2 = rng.choice(np.where(labels == id2)[0])
            # id1 注册（生成密钥），但 id2 用随机密钥认证（不知道 id1 的密钥）
            re, ke_genuine = ctm.enroll(codes[idx1], seed=int(idx1))
            _, ke_random = ctm.enroll(codes[idx2], seed=int(rng.integers(0, 99999)))
            rp = ctm.authenticate(codes[idx2], ke_random)  # 随机密钥
            d = ctm.hamming_distance(re, rp)
            impostor_dists.append(d / ctm.G)

    elif scenario == "stolen_key":
        # Stolen key 场景: impostor 知道 genuine 的密钥，但提交自己的生物特征
        # 论文: "the impostor has access to the actual key of the genuine user
        #        and tries to break the system by presenting actual key with impostor biometrics"
        for _ in range(n_impostors):
            id1, id2 = rng.choice(unique_ids, size=2, replace=False)
            idx1 = rng.choice(np.where(labels == id1)[0])
            idx2 = rng.choice(np.where(labels == id2)[0])
            re, ke_genuine = ctm.enroll(codes[idx1])  # id1 的密钥
            rp = ctm.authenticate(codes[idx2], ke_genuine)  # 用 id1 的密钥，但 id2 的特征
            d = ctm.hamming_distance(re, rp)
            impostor_dists.append(d / ctm.G)

    else:
        raise ValueError(f"scenario 必须是 'unknown_key' 或 'stolen_key'，得到: {scenario}")

    return np.array(genuine_dists), np.array(impostor_dists)


def compute_eer(genuine_dists, impostor_dists):
    """计算 EER（等错误率）"""
    # 将距离转为相似度（距离越小越相似）
    # genuine: 距离小 → 相似度高 → 标签1
    # impostor: 距离大 → 相似度低 → 标签0
    scores = np.concatenate([-genuine_dists, -impostor_dists])
    y_true = np.concatenate([np.ones(len(genuine_dists)),
                              np.zeros(len(impostor_dists))])

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr

    # EER: FPR == FNR 的点
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return eer, fpr, tpr, thresholds


def compute_gar_at_far(fpr, tpr, target_far=0.005):
    """计算指定 FAR 下的 GAR"""
    idx = np.searchsorted(fpr, target_far)
    if idx >= len(tpr):
        return tpr[-1]
    return tpr[idx]


def plot_distributions(genuine_dists, impostor_uk_dists, impostor_sk_dists,
                       G, save_path=None):
    """
    绘制 Genuine/Impostor 汉明距离分布（对应论文 Fig.4）
    同时展示 unknown key 和 stolen key 两种场景的 impostor 分布
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 1, 300)

    g_mean, g_std = genuine_dists.mean(), genuine_dists.std()
    uk_mean, uk_std = impostor_uk_dists.mean(), impostor_uk_dists.std()
    sk_mean, sk_std = impostor_sk_dists.mean(), impostor_sk_dists.std()

    ax.plot(x, norm.pdf(x, g_mean, g_std),  'b-', linewidth=2, label='Genuine')
    ax.plot(x, norm.pdf(x, uk_mean, uk_std), 'r-', linewidth=2, label='Impostor (Unknown Key)')
    ax.plot(x, norm.pdf(x, sk_mean, sk_std), 'g--', linewidth=2, label='Impostor (Stolen Key)')
    ax.fill_between(x, norm.pdf(x, g_mean, g_std),  alpha=0.15, color='blue')
    ax.fill_between(x, norm.pdf(x, uk_mean, uk_std), alpha=0.15, color='red')
    ax.fill_between(x, norm.pdf(x, sk_mean, sk_std), alpha=0.15, color='green')

    ax.set_xlabel('归一化汉明距离')
    ax.set_ylabel('概率密度')
    ax.set_title(f'Genuine/Impostor 分布 (G={G})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"分布图已保存: {save_path}")
    plt.show()


def plot_roc(fpr_uk, tpr_uk, eer_uk, fpr_sk, tpr_sk, eer_sk,
             G, save_path=None):
    """
    绘制 ROC 曲线（对应论文 Fig.7/8）
    同时展示 unknown key 和 stolen key 两种场景
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    auc_uk = auc(fpr_uk, tpr_uk)
    auc_sk = auc(fpr_sk, tpr_sk)
    ax.plot(fpr_uk, tpr_uk, 'b-', linewidth=2,
            label=f'Unknown Key (AUC={auc_uk:.3f}, EER={eer_uk*100:.2f}%)')
    ax.plot(fpr_sk, tpr_sk, 'r--', linewidth=2,
            label=f'Stolen Key  (AUC={auc_sk:.3f}, EER={eer_sk*100:.2f}%)')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FAR (False Accept Rate)')
    ax.set_ylabel('GAR (Genuine Accept Rate)')
    ax.set_title(f'ROC 曲线 (G={G})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC图已保存: {save_path}")
    plt.show()


def _compute_gar_with_sstm(codes, labels, ctm, sstm, scenario):
    """辅助函数：用 SSTM 计算指定场景下的 GAR"""
    unique_ids = np.unique(labels)
    rng = np.random.default_rng(42)
    pass_count = total = 0
    for uid in unique_ids:
        idx = np.where(labels == uid)[0]
        if len(idx) < 2:
            continue
        re, ke = ctm.enroll(codes[idx[0]])
        stored_hash, _ = sstm.enroll(re)
        for i in idx[1:]:
            if scenario == "unknown_key":
                # impostor 用随机密钥
                _, ke_rand = ctm.enroll(codes[i],
                                        seed=int(rng.integers(0, 99999)))
                rp = ctm.authenticate(codes[i], ke_rand)
            else:
                # genuine probe: 用注册密钥
                rp = ctm.authenticate(codes[i], ke)
            is_genuine, _ = sstm.authenticate(rp, stored_hash)
            pass_count += int(is_genuine)
            total += 1
    return pass_count / total if total > 0 else 0


def plot_gs_curve(codes, labels, ctm, K_values, save_path=None):
    """
    绘制 G-S 曲线（GAR vs 安全性 bits）（对应论文 Fig.9/10）
    同时展示 unknown key 和 stolen key 两种场景

    Args:
        K_values: RS 信息符号数列表，安全性 k = K*8 bits
                  论文实验: K ∈ {7,10,13,15} → k ∈ {56,80,104,120} bits
    """
    gars_uk = []
    gars_sk = []
    k_bits_list = [K * 8 for K in K_values]

    for K in K_values:
        k = K * 8
        # G 必须是8的倍数，且 N=G//8 > K
        if ctm.G % 8 != 0 or ctm.G // 8 <= K:
            gars_uk.append(0)
            gars_sk.append(0)
            print(f"  K={K} (k={k} bits): 跳过（G={ctm.G}不满足条件）")
            continue
        sstm = SSTM(G=ctm.G, K=K)

        # Genuine（stolen key场景中genuine用自己密钥，等价于正常认证）
        gar_genuine = _compute_gar_with_sstm(codes, labels, ctm, sstm,
                                              scenario="stolen_key")
        gar_uk = _compute_gar_with_sstm(codes, labels, ctm, sstm,
                                         scenario="unknown_key")
        gars_sk.append(gar_genuine * 100)
        gars_uk.append(gar_uk * 100)
        print(f"  K={K} (k={k} bits): GAR(genuine)={gar_genuine*100:.1f}%  "
              f"GAR(unknown_key_impostor)={gar_uk*100:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_bits_list, gars_sk, 'b-o', linewidth=2, label='Stolen Key (Genuine)')
    ax.plot(k_bits_list, gars_uk, 'r--s', linewidth=2, label='Unknown Key (Genuine)')
    ax.set_xlabel('安全性 k (bits)')
    ax.set_ylabel('GAR (%)')
    ax.set_title(f'G-S 曲线 (G={ctm.G} bits)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"G-S曲线已保存: {save_path}")
    plt.show()
    return k_bits_list, gars_sk, gars_uk


def run_evaluation(model_path=None, G_values=None, data_root="fingerprints",
                   db_names=None, output_dir="results"):
    """
    完整评估流程

    Args:
        model_path: 模型权重路径，None时使用随机初始化（测试流程用）
        G_values: 测试的随机比特数列表，如 [128, 256, 512, 768]
        data_root: 数据根目录
        db_names: 使用的DB列表
        output_dir: 结果保存目录
    """
    if G_values is None:
        G_values = [128, 256, 512, 768]
    if db_names is None:
        db_names = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    _, test_loader, num_classes = build_dataloaders(
        data_root, db_names, train_ratio=0.7, batch_size=8
    )

    # 加载模型
    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024, pretrained=False)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型: {model_path}")
    else:
        print("使用随机初始化模型（仅用于流程验证）")
    model = model.to(device)
    model.set_beta(32)  # 推理时使用最大 beta

    # 提取二值码
    print("\n提取测试集二值哈希码...")
    codes, labels = extract_all_binary_codes(model, test_loader, device)
    print(f"提取完成: {codes.shape}")

    # 对不同 G 值进行评估（两种场景）
    results = {}
    for G in G_values:
        print(f"\n{'='*45}")
        print(f"评估 G={G} bits")
        ctm = CTM(hash_dim=1024, G=G)

        genuine_dists, imp_uk = compute_genuine_impostor_distances(
            codes, labels, ctm, scenario="unknown_key"
        )
        _, imp_sk = compute_genuine_impostor_distances(
            codes, labels, ctm, scenario="stolen_key"
        )

        print(f"  Genuine:           均值={genuine_dists.mean():.3f}, "
              f"std={genuine_dists.std():.3f}")
        print(f"  Impostor(UK):      均值={imp_uk.mean():.3f}, "
              f"std={imp_uk.std():.3f}")
        print(f"  Impostor(SK):      均值={imp_sk.mean():.3f}, "
              f"std={imp_sk.std():.3f}")

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
                           save_path=os.path.join(output_dir, f"dist_G{G}.png"))
        plot_roc(fpr_uk, tpr_uk, eer_uk, fpr_sk, tpr_sk, eer_sk, G,
                 save_path=os.path.join(output_dir, f"roc_G{G}.png"))

    # 汇总表（对应论文 Table I 格式）
    print("\n" + "="*60)
    print("汇总结果（对应论文 Table I）:")
    print(f"{'G':>6} | {'EER_UK(%)':>10} | {'EER_SK(%)':>10} | "
          f"{'GAR_UK@0.5%':>12} | {'GAR_SK@0.5%':>12}")
    print("-" * 60)
    for G in G_values:
        r = results[G]
        print(f"{G:>6} | {r['eer_uk']*100:>10.2f} | {r['eer_sk']*100:>10.2f} | "
              f"{r['gar_uk']*100:>12.2f} | {r['gar_sk']*100:>12.2f}")

    # G-S 曲线（用最大G，对应论文 Fig.9/10）
    best_G = max([G for G in G_values if G % 8 == 0], default=G_values[-1])
    ctm_best = CTM(hash_dim=1024, G=best_G)
    # K_values: 扩展范围，覆盖 GAR 从100%下降到0%的完整区间
    # N = best_G // 8，K 必须 < N
    # 目标: 找到 GAR 开始下降的点（约 K=39）和完全失败的点（约 K=58）
    N = best_G // 8
    K_values = [k for k in range(7, N) if k % 2 == 1]  # 奇数K，步长2，覆盖完整范围
    if K_values:
        print(f"\n绘制完整 G-S 曲线 (G={best_G}, K从7到{K_values[-1]})...")
        plot_gs_curve(codes, labels, ctm_best, K_values,
                      save_path=os.path.join(output_dir, f"gs_curve_G{best_G}_full.png"))

    return results


if __name__ == "__main__":
    # 如果已有训练好的模型，传入路径；否则用随机初始化验证流程
    # G 必须是8的倍数（论文实验: 128/256/512/768）
    model_path = "checkpoints/final_model.pth"
    run_evaluation(
        model_path=model_path if os.path.exists(model_path) else None,
        G_values=[128, 256, 512],
        output_dir="results"
    )
