# SSTM 模块改进：从 BCH 码到极化码与生物特征嵌入

---

## 一、背景回顾

上次汇报确认了 BCH 码替换 RS 码的有效性：
- RS 码因符号错误率 84% 完全失效（GAR ≈ 0%）
- BCH 比特级纠错方案，G-S 拐点 k=208 bits，GAR 在低安全性下接近 100%

本次工作的出发点：**能否进一步提升 G-S 曲线，在更高安全性下维持更高 GAR？**

---

## 二、方案一：PolarEmbed（极化码思想 + 生物特征融入）

### 2.1 核心思路

BCH 方案的局限：密钥 s 被均匀放置在前 k 个固定位置，没有利用任何生物特征信息。

**关键洞察**：深度哈希网络输出的 tanh 值 h = tanh(βx) 包含了重要信息——

$$|h_i| = |\tanh(\beta x_i)| \approx 1 \Rightarrow \text{网络对第 } i \text{ 个 bit 非常确定，稳定}$$
$$|h_i| \approx 0 \Rightarrow \text{网络不确定，该位置容易翻转}$$

受**极化码信道极化思想**启发：将 bit 位置分为"可靠信道"（放密钥）和"不可靠信道"（冻结位），密钥集中在最稳定的位置，认证时该区域翻转最少，纠错最容易。

### 2.2 详细实现

#### 注册阶段

**输入**：
- `re`：（512 bits）StableCTM 输出的可撤销模板
- `embed_e`：（512 floats）tanh 连续值，即 CTM 选出的 512 个位置对应的网络输出

**Step 1：按置信度排序**

$$\text{perm} = \text{argsort}(-|embed\_e|)$$

perm[0..119] 是 |tanh| 最大的 120 个位置 → **可靠信道**（放密钥）  
perm[120..511] 是 |tanh| 较小的位置 → **不可靠信道**（冻结位，置 0）

**Step 2：密钥嵌入**

```
s_mapped = zeros(512)
s_mapped[perm[:120]] = s        ← 密钥 s（120 bits）放入可靠位置
s_mapped[perm[120:]] = 0        ← 冻结位为 0
```

**Step 3：辅助数据与存储**

```
h = re XOR s_mapped             ← 辅助数据（公开）
存储：SHA256(s) | h | ecc       ← ecc 为 BCH 校验码
```

#### 认证阶段

**输入**：
- `rp`：（512 bits）探针可撤销模板（与 re 有约 57 bits 翻转）
- `embed_p`：（512 floats）探针图像的 tanh 连续值

**Step 1：恢复含噪密钥区域**

```
sp_noisy = rp XOR h
s_noisy = sp_noisy[perm[:120]]  ← 只取可靠信道位置的 120 bits
```

数学推导：
$$sp\_noisy = rp \oplus h = rp \oplus (re \oplus s\_mapped) = \underbrace{(rp \oplus re)}_{\text{噪声}} \oplus s\_mapped$$

噪声（约 57 bits）在 512 个位置上均匀分布，落在可靠信道 120 个位置内的期望：
$$57 \times \frac{120}{512} \approx 13 \text{ bits}$$

**Step 2：BCH 纠错**

```
s_recovered = BCH.correct(s_noisy, ecc)   ← BCH 纠错（最多 t=56 bits）
```

13 bits « t=56 bits，轻松纠正。

**Step 3：比对哈希**

```
SHA256(s_recovered) == stored_hash ?
```

### 2.3 与 BCH 的核心区别

| 维度 | BCH（原方案） | PolarEmbed（本方案） |
|------|-------------|-------------------|
| 密钥放置位置 | 固定前 k 字节（随机分布） | **tanh 绝对值最大的 k 个位置** |
| 生物特征利用 | 不利用 | **直接利用 tanh 置信度** |
| 纠错方式 | BCH 比特级纠错 | BCH 比特级纠错 |
| 自适应性 | 静态（与生物特征无关） | **动态（每次认证不同）** |

### 2.4 实验结果

| 安全性 k | BCH GAR | PolarEmbed GAR |
|---------|---------|----------------|
| 120 bits | 100.0% | 99.9% |
| 160 bits | 91.1% | 96.3% |
| 208 bits | 54.9% | **86.2%** |
| 256 bits | 15.3% | **54.2%** |

**G-S 拐点（GAR=50%）**：BCH = k=208 bits，PolarEmbed = **k=256 bits**，提升 48 bits。

---

## 三、方案二：真正的极化码 SSTM（SC/SCL + CRC）

### 3.1 极化码基础

极化码由 Arikan（2009）提出，核心定理：

> N 个独立的 BSC(p) 信道，经极化变换 $G_N = F^{\otimes n}$（F=[[1,0],[1,1]]）后，子信道分化为两类：
> - **可靠信道**（巴氏参数 Z → 0）：错误概率趋近于 0
> - **不可靠信道**（Z → 1）：错误概率趋近于 1

将密钥放入可靠信道（信息位），不可靠信道固定为已知值（冻结位），认证时用 SC 或 SCL 译码恢复密钥。

### 3.2 巴氏参数：衡量信道可靠性

对于 BSC(p) 信道，初始巴氏参数：

$$Z_0 = 2\sqrt{p(1-p)}$$

极化递推（从一个信道逐级分裂，共 n = log₂N 级）：

$$Z^+ = 2Z - Z^2 \quad (\text{合并信道，更可靠})$$
$$Z^- = Z^2 \quad (\text{分裂信道，更不可靠})$$

经过 9 级递推（N=512=2⁹），得到 512 个极化子信道各自的巴氏参数 Z[i]，越小越可靠。

**生物特征融入**：将 tanh 置信度融入翻转率估计：

$$p_{eff}[i] = p \times (1 - |embed[i]| \times 0.8)$$

置信度高的位置 → `p_eff` 更小 → 对应极化子信道更可靠 → 更可能被选为信息位。

选取 Z 最小的 k 个子信道作为信息位：

$$\text{info\_positions} = \text{argsort}(Z)[:k]$$

### 3.3 极化编码

构造输入向量 u（N=512 bits）：

```
u[info_positions] = s      ← 密钥 s（k bits）
u[frozen_positions] = 0    ← 冻结位为 0
```

极化编码（G_N 是自逆矩阵，编解码用同一变换）：

$$x = u \cdot G_N$$

实现为蝶形运算（O(N log N)，类似 FFT）：

```
for step in [1, 2, 4, ..., 256]:
    for j in range(0, 512, step*2):
        x[j:j+step] XOR= x[j+step:j+step*2]
```

辅助数据：

```
h = re XOR x
存储：SHA256(s) | h | p_eff_量化
```

### 3.4 SC 译码（基础版）

认证时恢复含噪码字：

```
x_noisy = rp XOR h
```

计算非均匀 LLR（融入探针置信度）：

$$LLR[i] = (1 - 2 \cdot x\_noisy[i]) \times \log\frac{1-p_{eff}[i]}{p_{eff}[i]}$$

- `|LLR[i]| 大`：对第 i 个位置判决很有把握
- `|LLR[i]| 趋近 0`：对第 i 个位置很不确定

**SC 译码递归**（对每个比特位置 i 从左到右判决）：

```
冻结位 → 直接判 0（已知先验）
信息位 → LLR[i] >= 0 判 0，< 0 判 1
```

LLR 通过 f/g 函数在极化树中传播：

$$f(a, b) = \text{sign}(a) \cdot \text{sign}(b) \cdot \min(|a|, |b|) \quad (\text{左子树})$$
$$g(a, b, u) = b + (1-2u) \cdot a \quad (\text{右子树，利用左子树判决结果})$$

### 3.5 SCL 译码（升级版）

**SC 的问题**：单路径，一步判错则全盘皆输。

**SCL 改进**：维护 L 条候选路径（L=8），遇到信息位时：

```
每条路径 → 分裂为"判0"和"判1"两条
路径度量更新：pm += log P(该判决|LLR)
保留度量最大的 L 条路径
```

**LLR 与路径分裂的关系**：

```
LLR[i] 很大（远大于0，非常确定是0）：
    判0的路径度量几乎不变，判1的路径度量大幅下降
    → SCL 很快淘汰"判1"的路径

LLR[i] ≈ 0（非常不确定）：
    判0和判1的度量几乎相等
    → SCL 同时保留两条路径，等后续信息再做决定
```

这正是生物特征置信度的作用：置信度低的 bit 对应 LLR≈0，SCL 会为这个位置保留两种可能，减少因不确定判决导致的错误。

**CRC 校验**：密钥 s 后追加 8 bits CRC 校验码。SCL 输出 L=8 条候选路径，第一条通过 CRC 的即为正确密钥。

```
for u_hat in candidates:  # 按度量降序
    s_bits = u_hat[info_positions][:120]
    if CRC8(s_bits) == u_hat[info_positions][120:128]:
        return s_bits     # CRC 通过
```

误判概率从 O(1/L) = 12.5% 降至 O(1/2^8) ≈ 0.4%。

### 3.6 实验结果与分析

| k (bits) | BCH | SCL L=1(≈SC) | SCL L=8 CRC-8 |
|---------|-----|-------------|--------------|
| 120 | 100.0% | 52.2% | 61.3% |
| 152 | 95.7% | 34.7% | 42.4% |
| 192 | 68.5% | 18.5% | 22.0% |

**SCL(L=8) 比 SC(L=1) 高约 8-9 个百分点**，验证了列表译码的有效性。

**但极化码整体差于 BCH**，原因：

1. BCH 部分填充方案有效纠错能力 $t_{eff} = t \times G/k = 56 \times 512/120 = 239$ bits，远超实际翻转 57 bits
2. 极化码在短码长（N=512）、高码率（R=0.25）下纠错能力有限
3. 用 tanh 置信度选出的信息位与极化码自然结构不完全匹配，译码精度下降

---

## 四、两方案对比与总结

| 维度 | PolarEmbed | 极化码 SCL+CRC |
|------|-----------|--------------|
| 极化码思想利用 | 借鉴"可靠/不可靠信道"的思想 | 完整实现编解码 + SCL 译码 |
| 生物特征融入 | tanh 决定密钥放在哪些位置 | tanh 调整 p_eff → 非均匀 LLR |
| 纠错码 | BCH | 极化码（SC/SCL）|
| G-S 拐点 | **k=256 bits（最优）** | k≈120 bits |
| 实现复杂度 | 低 | 高（需要完整 SC/SCL 译码器）|

### 核心结论

1. **PolarEmbed 是本工作最优方案**：将极化码的核心思想（可靠信道嵌入信息）与深度网络的置信度信息成功融合，G-S 拐点比 BCH 提升 48 bits

2. **极化码完整实现是诚实的负结果**：在 G=512、p=11.2% 的短码场景下，BCH 部分填充方案更优；极化码在长码下有理论优势，但本场景不适用

3. **SCL 相比 SC 有效提升 8-9%**，验证了列表译码的工程价值

---

## 五、展示图清单

| 图 | 路径 | 说明 |
|----|------|------|
| 图1 | `results_bch_stable/gs_rs_vs_bch_G512.png` | RS 码失效 → BCH 有效（问题起点）|
| 图2 | `results/bit_flip_rate_analysis.png` | Bit 翻转率均匀分布（说明 StableCTM 效果有限）|
| 图3 | `results/compare_selection_G512.png` | CTM vs StableCTM 翻转率对比 |
| 图4 | `results_ablation/ablation_gs_G512.png` | 消融：CTM vs StableCTM vs BioHashing |
| 图5 | `results_polar/polar_vs_bch_G512.png` | BCH vs Polar vs PolarEmbed |
| 图6 | `results_scl/scl_vs_bch_G512.png` | 四方法对比（含 SCL） |
