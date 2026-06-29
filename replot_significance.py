"""
replot_significance.py — 从已有 JSON 重新生成 significance 图片（本地运行，无需模型）
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "results_significance"
G = 512
JSON_PATH  = os.path.join(OUTPUT_DIR, "significance_results.json")

# ── 读取结果 ─────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    results = json.load(f)

seeds    = results["seeds"]
k50_bch  = results["BCH"]["k50_per_seed"]
k50_rgss = results["RGSS"]["k50_per_seed"]
advantages = results["advantage_RGSS_minus_BCH"]["per_seed"]
p_value  = results["paired_ttest"]["p_value"]

seed_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

# ── 1. significance_boxplot.png ───────────────────────────────────────
def plot_results():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: slope chart (paired comparison lines) ──────────────────
    ax = axes[0]
    for kb, kr, s, c in zip(k50_bch, k50_rgss, seeds, seed_colors):
        ax.plot([0, 1], [kb, kr], 'o-', color=c, markersize=10,
                linewidth=2.0, label=f'seed={s}', zorder=5)
        ax.annotate(f'{kb}', xy=(0, kb), xytext=(-0.12, kb),
                    fontsize=8, color=c, va='center', ha='right')
        ax.annotate(f'{kr}', xy=(1, kr), xytext=(1.12, kr),
                    fontsize=8, color=c, va='center', ha='left')

    ax.plot(0, np.mean(k50_bch),  'k^', markersize=13, zorder=10,
            label=f'Mean BCH={np.mean(k50_bch):.1f}')
    ax.plot(1, np.mean(k50_rgss), 'ks', markersize=13, zorder=10,
            label=f'Mean RGSS={np.mean(k50_rgss):.1f}')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BCH (baseline)', 'RGSS (proposed)'], fontsize=12)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel('k₅₀ inflection point (bits)', fontsize=11)
    ax.set_title(f'Paired k₅₀ comparison ({len(seeds)} splits)\n'
                 f'BCH: {np.mean(k50_bch):.1f}±{np.std(k50_bch):.1f}   '
                 f'RGSS: {np.mean(k50_rgss):.1f}±{np.std(k50_rgss):.1f} bits',
                 fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(min(k50_bch) - 20, max(k50_rgss) + 20)

    # ── Right: advantage per seed + mean line ─────────────────────────
    ax2 = axes[1]
    x = np.arange(len(seeds))
    bars = ax2.bar(x, advantages,
                   color=seed_colors, alpha=0.80, edgecolor='black', width=0.55)

    for bar, val in zip(bars, advantages):
        ax2.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 0.8,
                 f'+{val} bits', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax2.axhline(y=np.mean(advantages), color='navy', linestyle='--',
                linewidth=2, label=f'Mean Δ = {np.mean(advantages):.1f} bits')
    ax2.axhspan(np.mean(advantages) - np.std(advantages),
                np.mean(advantages) + np.std(advantages),
                alpha=0.15, color='navy',
                label=f'±1 std ({np.std(advantages):.1f} bits)')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'seed\n={s}' for s in seeds], fontsize=10)
    ax2.set_ylabel('RGSS − BCH k₅₀ advantage (bits)', fontsize=11)
    ax2.set_title(f'k₅₀ advantage per split (RGSS − BCH)\n'
                  f'Mean={np.mean(advantages):.1f} bits, '
                  f'Std={np.std(advantages):.1f} bits, '
                  f'p={p_value:.1e}',
                  fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(advantages) + 18)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "significance_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── 2. gs_curves_all_seeds.png ────────────────────────────────────────
# 这张图需要曲线数据，但 JSON 里只存了 k₅₀。
# 用 per_seed_detail 里的 k₅₀ 在 50% 处标记即可；
# 若你想重新画完整曲线，需在服务器上运行 evaluate_significance.py
def plot_k50_summary():
    """用 k₅₀ 数值画一张简洁的汇总条形图（替代曲线图，本地可生成）"""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(seeds))
    w = 0.35
    bars_b = ax.bar(x - w/2, k50_bch,  width=w, color='steelblue',
                    alpha=0.85, edgecolor='black', label='BCH (baseline)')
    bars_g = ax.bar(x + w/2, k50_rgss, width=w, color='mediumseagreen',
                    alpha=0.85, edgecolor='black', label='RGSS (proposed)')

    # value labels
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)
    for bar in bars_g:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)

    # mean lines
    ax.axhline(np.mean(k50_bch),  color='steelblue',    linestyle='--', linewidth=1.5,
               label=f'BCH mean = {np.mean(k50_bch):.1f} bits')
    ax.axhline(np.mean(k50_rgss), color='mediumseagreen', linestyle='--', linewidth=1.5,
               label=f'RGSS mean = {np.mean(k50_rgss):.1f} bits')

    ax.set_xticks(x)
    ax.set_xticklabels([f'seed={s}' for s in seeds], fontsize=10)
    ax.set_ylabel('k₅₀ inflection point (bits)', fontsize=11)
    ax.set_title(f'k₅₀ per split: BCH vs RGSS  (G={G})\n'
                 f'Advantage: RGSS − BCH = {np.mean(advantages):.1f}±{np.std(advantages):.1f} bits  '
                 f'(paired t-test p={p_value:.1e})', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(k50_rgss) + 40)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "k50_summary_bar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_results()
    plot_k50_summary()
    print("Done.")
