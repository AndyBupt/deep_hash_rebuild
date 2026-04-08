# Experiment Log: Deep Hashing for Secure Fingerprint Templates

---

## Experiment 1 — Baseline Reproduction

**Date**: 2026-04-08
**Goal**: Reproduce the single-modal (fingerprint-only) version of "Deep Hashing for Secure Multimodal Biometrics"

---

### 1. Configuration

#### Dataset
| Item | Value |
|------|-------|
| Dataset | FVC2004 (DB1_A, DB1_B, DB2_A, DB2_B, DB3_A, DB3_B) |
| Subjects | 330 |
| Images per subject | 8 |
| Train / Test split | 70% / 30% |

#### Model
| Item | Value |
|------|-------|
| Backbone | VGG-19 (ImageNet pretrained) |
| Hash dimension | 1024 bits |
| Binarization | tanh(β·x) continuation, β ∈ {1, 2, 4, 8, 16, 32} |
| Loss | α·E1 + β·E2 + γ·E3 (α=8, β=2, γ=2) |
| Training | Two-stage: freeze backbone → end-to-end fine-tuning |

#### CTM (Cancelable Template Module)
| Item | Baseline | Improved |
|------|----------|----------|
| Method | Random bit selection | Stable bit selection (StableCTM) |
| stable_ratio | — | 0.8 (top 80% most stable bits as pool) |
| G values tested | 128, 256, 512 | 512 |

#### SSTM (Secure Sketch Template Module)
| Item | Value |
|------|-------|
| Method | Fuzzy Commitment with Reed-Solomon codes |
| Field | GF(2^8), symbol = 8 bits |
| Codeword length | N = G // 8 symbols |
| Security | k = K × 8 bits |
| Error correction | t = (N − K) // 2 symbols |
| Hash | SHA-256 |

> **Note on SSTM implementation**: The paper describes directly decoding the cancelable template `re` as a noisy RS codeword. However, since `re` is arbitrary binary data (not a valid codeword), direct RS decoding always fails. We implement the equivalent **fuzzy commitment scheme**: enroll generates a random secret `s`, computes helper data `h = re ⊕ RS.encode(s)`, and stores `SHA256(s)` + `h`. Authentication recovers `s` via RS decoding of `rp ⊕ h`.

---

### 2. Results

#### 2.1 ROC / EER Results

| G (bits) | EER Unknown Key (%) | EER Stolen Key (%) | GAR@FAR=0.5% UK (%) | GAR@FAR=0.5% SK (%) | AUC UK | AUC SK |
|----------|--------------------|--------------------|----------------------|----------------------|--------|--------|
| 128 | 1.27 | 9.16 | 97.92 | 34.52 | 0.9987 | 0.9671 |
| 256 | 0.80 | 8.32 | 99.11 | 31.25 | 0.9989 | 0.9692 |
| **512** | **0.37** | **8.12** | **99.70** | **33.33** | **0.9995** | **0.9714** |

**Observation**: EER decreases as G increases. G=512 achieves the best performance (EER=0.37% under unknown key attack).

#### 2.2 Hamming Distance Statistics

| G (bits) | Genuine Mean | Genuine Std | Bit Flip Rate | Symbol Error Rate | Impostor UK Mean | Impostor SK Mean |
|----------|-------------|-------------|---------------|-------------------|-----------------|-----------------|
| 128 | 0.2044 | 0.0778 | 20.4% | **83.9%** | 0.4992 | 0.4736 |
| 256 | 0.2066 | 0.0743 | 20.7% | **84.3%** | 0.4989 | 0.4736 |
| 512 | 0.2061 | 0.0733 | 20.6% | **84.2%** | 0.5002 | 0.4733 |

**Observation**: The genuine bit flip rate is consistently ~20% across all G values, indicating this is an intrinsic property of the model/dataset, not affected by G. The impostor unknown key mean is ~0.50, consistent with the theoretical expectation for random binary vectors.

#### 2.3 G-S Curve (G=512)

| k (bits) | GAR Baseline (%) | GAR Improved (StableCTM) (%) |
|----------|-----------------|------------------------------|
| 56 | 1.93 | 2.38 |
| 72 | 1.64 | 1.79 |
| 88 | 1.04 | 1.93 |
| 104 | 1.64 | 1.49 |
| 120 | 0.89 | 1.04 |
| 152 | 0.15 | 0.60 |
| 200 | 0.15 | 0.15 |
| 264 | 0.00 | 0.00 |
| 312+ | 0.00 | 0.00 |

**Observation**: GAR drops below 2% even at the lowest security level (k=56 bits), and reaches 0% at k=264 bits. StableCTM shows marginal improvement at low k values but no meaningful difference overall.

---

### 3. Analysis

#### 3.1 Why the G-S Curve Fails

The fundamental bottleneck is the mismatch between the genuine bit flip rate and RS code's error correction model.

**Mathematical derivation**:

Given genuine bit flip rate p = 20.6% (G=512):

- Probability a symbol (8 bits) is error-free: $(1 - 0.206)^8 \approx 0.158$
- Symbol Error Rate (SER): $1 - 0.158 = \mathbf{84.2\%}$
- Expected symbol errors in codeword: $64 \times 84.2\% \approx \mathbf{54}$ symbols

RS error correction capacity (e.g., K=13, N=64):
$$t = \frac{N - K}{2} = \frac{64 - 13}{2} = 25 \text{ symbols}$$

Since $54 \gg 25$, RS decoding fails for virtually all genuine pairs, resulting in near-zero GAR across all security levels.

#### 3.2 Why StableCTM Does Not Help

StableCTM selects bits from the most stable pool (lowest flip rate), which can reduce the genuine bit flip rate slightly. However, even if the flip rate is reduced from 20.6% to, say, 15%, the SER would still be:

$(1 - 0.15)^8 \approx 0.272$, SER $\approx 72.8\%$, expected symbol errors $\approx 47$

This still far exceeds the RS correction capacity, so the G-S curve remains near zero.

#### 3.3 Root Cause: Single-Modal vs. Multi-Modal

The original paper uses multi-modal biometrics (face + iris + fingerprint). The fusion of multiple modalities produces a much more stable hash code with a significantly lower genuine bit flip rate (estimated <5%). With p=5%:

- SER $= 1 - (0.95)^8 \approx 33.7\%$
- Expected symbol errors $\approx 22$ symbols
- This is within the RS correction capacity for reasonable K values

Single-modal fingerprint with VGG-19 (not designed for fingerprint topology) on unaligned FVC images inherently produces high intra-class variance (~20% flip rate), making the SSTM G-S curve ineffective.

#### 3.4 Model Quality is Good

Despite the G-S curve failure, the model itself performs well:
- EER=0.37% (Unknown Key, G=512) indicates excellent discriminability
- AUC=0.9995 confirms near-perfect ROC performance
- The CTM module correctly produces cancelable templates with ~50% impostor distance

The problem is entirely in the SSTM stage: the RS code cannot handle the noise level produced by single-modal fingerprint hashing.

---

### 4. Identified Issues During Reproduction

| Issue | Description | Fix Applied |
|-------|-------------|-------------|
| SSTM always returns True | Direct RS decode of random data always fails → both enroll and auth fall back to `\x00` → same hash → always True | Rewrote SSTM as fuzzy commitment scheme |
| StableCTM pool too small | stable_ratio=0.3 → pool=max(512,307)=512 → no randomness, all users get same key | Changed stable_ratio to 0.8 → pool=819 |
| VGG pretrained weights not loaded | fc1/fc2 initialized randomly | Fixed to copy from vgg.classifier[0] and [3] |

---

### 5. Next Steps / Improvement Directions

#### Direction A: Bit Interleaving in SSTM (Medium effort)
Rearrange the 512 bits before RS encoding so that random bit flips spread across different symbols. Theoretically reduces symbol errors from ~54 to ~13 (= 102 bits / 8), potentially bringing it within RS correction range.

#### Direction B: Improve Frontend Model (High effort, fundamental fix)
Reduce genuine bit flip rate from ~20% to <5% by:
- Adding fingerprint alignment/registration preprocessing
- Using a fingerprint-specific feature extractor
- Strengthening stability constraints in the loss function

#### Direction C: Use BCH Codes Instead of RS Codes (Medium effort)
BCH codes operate on bits directly, not symbols. This avoids the symbol error rate amplification problem entirely. With t=102 bits correction capacity, genuine pairs with ~102 bit flips would pass authentication.

---

### 6. Files

| File | Description |
|------|-------------|
| `model.py` | VGG-19 + hashing layer |
| `train.py` | Two-stage training pipeline |
| `ctm.py` | CTM + StableCTM |
| `sstm.py` | SSTM (fuzzy commitment with RS codes) |
| `evaluate.py` | Full evaluation: ROC, dist, G-S curve, JSON output |
| `results/results_summary.json` | Numerical results (this experiment) |
| `results/dist_G512.png` | Genuine/Impostor distance distribution |
| `results/roc_G512.png` | ROC curves (Unknown Key + Stolen Key) |
| `gs_comparison_G512.png` | G-S curve: Baseline vs. StableCTM |
