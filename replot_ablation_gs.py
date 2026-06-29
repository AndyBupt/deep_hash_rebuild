"""
replot_ablation_gs.py
根据已有 JSON 数据重绘两张消融图，
依照老师建议去除 "security" / "secret key" 措辞，
改为 "Key Length k (bits)" / "k₅₀"。

图1: ablation_gs.png      — 5种通道选择策略 (Exp 6)
图2: exp4_ablation_frontend.png — 3种前端方法 (Exp 4)
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
# 图1: Exp 6 — 5种通道选择策略消融
# ══════════════════════════════════════════════
JSON_PATH_1  = "重跑/results_ablation/ablation_results.json"
OUTPUT_PNG_1 = "paper/figures/ablation_gs.png"

# key = JSON中的策略名, value = (颜色, 线型, 图例显示名)
STRATEGY_STYLES = {
    "Random":          ("#7f7f7f", "--x", "Random"),
    "Flip-rate":       ("#ff7f0e", "-^",  "Flip-rate"),
    "Tanh-confidence": ("#1f77b4", "-s",  "RGSS"),          # 改为 RGSS
    "Worst-channel":   ("#d62728", "-v",  "Worst-channel"),
    "Oracle":          ("#2ca02c", "-D",  "Oracle"),
}

with open(JSON_PATH_1, "r") as f:
    data1 = json.load(f)

G1         = data1.get("G", 512)
strategies = data1.get("strategies", {})

fig, ax = plt.subplots(figsize=(12, 7))
for name, (color, style, display_name) in STRATEGY_STYLES.items():
    r = strategies.get(name)
    if r is None:
        continue
    k_bits = r.get("k_bits", [])
    gars   = r.get("GAR (%)", [])
    if not k_bits:
        continue
    marker    = style[-1]
    linestyle = style[:-1]
    infl = r.get("inflection_k_bits")
    label = f"{display_name}  (k₅₀={infl} bits)" if infl else display_name
    ax.plot(k_bits, gars,
            color=color, linestyle=linestyle, marker=marker,
            linewidth=2, markersize=4, label=label)

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
ax.set_xlabel('Key Length k (bits)', fontsize=13)
ax.set_ylabel('GAR (%)', fontsize=13)
ax.set_title(
    f'Ablation: Channel-Selection Strategy Comparison\n'
    f'(G={G1} bits, BCH ECC back-end, StableCTM front-end, FVC2004)',
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 108)
plt.savefig(OUTPUT_PNG_1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_PNG_1}")


# ══════════════════════════════════════════════
# 图2: Exp 4 — 3种前端方法对比
# ══════════════════════════════════════════════
JSON_PATH_2  = "results_ablation/ablation_results_G512.json"
OUTPUT_PNG_2 = "paper/figures/exp4_ablation_frontend.png"

FRONTEND_STYLES = {
    "CTM (Random)": ("green",  "^", 20.8),   # (color, marker, flip%)
    "StableCTM":    ("blue",   "s", 19.6),
    "BioHashing":   ("red",    "o", 29.8),
}

with open(JSON_PATH_2, "r") as f:
    data2 = json.load(f)

G2      = data2.get("G", 512)
results = data2.get("results", {})

fig, ax = plt.subplots(figsize=(11, 6))
for name, (color, marker, flip_pct) in FRONTEND_STYLES.items():
    r = results.get(name)
    if r is None:
        continue
    k_bits = r.get("k_bits", [])
    gars   = r.get("GAR (%)", [])
    if not k_bits:
        continue
    # 用 JSON 里存的真实翻转率（若有），否则用上面的默认值
    actual_flip = r.get("genuine_flip_rate (%)", flip_pct)
    label = f"{name} (flip={actual_flip:.1f}%)"
    ax.plot(k_bits, gars,
            color=color, marker=marker,
            linewidth=2, markersize=4, label=label)

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='GAR=50%')
ax.set_xlabel('Key Length k (bits)', fontsize=13)
ax.set_ylabel('GAR (%)', fontsize=13)
ax.set_title(
    f'Exp 4: Ablation — Frontend Methods + BCH SSTM (G={G2})',
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 108)
plt.savefig(OUTPUT_PNG_2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_PNG_2}")
