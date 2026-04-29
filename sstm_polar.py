"""
SSTM_Polar: Secure Sketch Template Module using Polar Codes

真正的极化码实现，包含：
  1. BSC 信道巴氏参数递推（Bhattacharyya Parameter）
  2. 极化编码（G_N = F^{⊗n} 蝶形运算）
  3. SC（Successive Cancellation）译码器（软信息逐位判决）

极化码核心思想：
  N 个 BSC(p) 信道经极化变换 G_N 后，子信道分化为：
    - 可靠信道（巴氏参数 Z → 0）：放密钥 s（信息位）
    - 不可靠信道（Z → 1）：固定为 0（冻结位）
  SC 译码器利用冻结位的已知先验，逐位恢复信息位的密钥 s。

生物特征置信度的融入方式：
  用 tanh 输出的绝对值 |tanh(x)| 调整每个原始信道的有效翻转率：
    p_eff[i] = p * (1 - |tanh[i]| * 0.8)
  用非均匀 p_eff 重新计算各子信道的巴氏参数，
  从而让密钥优先嵌入"生物特征置信度高且极化后可靠"的子信道。
  认证时用 embed_p 计算非均匀 LLR，软判决更精确。

与其他 SSTM 方案的区别：
  - SSTM_BCH：BCH 比特级纠错，均匀对待所有 bit，无极化码
  - SSTM_PolarEmbed：按 |tanh| 选位 + BCH，无极化编码/SC 译码
  - SSTM_Polar（本文件）：真正的极化码编解码 + SC 译码 + 置信度融入

方案流程：
  注册:
    re (G bits) + embed_e (G floats, 可选)
    → p_eff[i] = p*(1-|embed[i]|*0.8)（embed=None 时 p_eff=p 均匀）
    → 非均匀巴氏参数递推 → 选 k 个最可靠子信道为信息位
    → 生成密钥 s，u[信息位]=s，冻结位=0
    → 极化编码: x = u·G_N
    → h = re XOR x，存储 SHA256(s)|h|info_positions|p_eff

  认证:
    rp (G bits) + embed_p (G floats, 可选)
    → x_noisy = rp XOR h
    → 非均匀 LLR（用 embed_p 或存储的 p_eff）
    → SC 译码 → û
    → s_hat = û[info_positions]
    → SHA256(s_hat) == stored_hash?

参考：
  Arikan (2009). "Channel Polarization." IEEE Trans. Inf. Theory, 55(7).
"""

import hashlib
import numpy as np
from typing import Tuple, Optional


class SSTM_Polar:
    """
    基于极化码的安全草图模板模块。
    接口与 SSTM_BCH / SSTM_PolarEmbed 完全兼容。
    """

    def __init__(self, G: int = 512, k: int = 120,
                 flip_prob: float = 0.112):
        """
        Args:
            G:         码字长度，必须是 2 的幂（128/256/512）
            k:         信息位数 = 安全性 bits
            flip_prob: BSC 信道翻转概率（per-bit 均值，约 0.112）
        """
        assert (G & (G - 1)) == 0, f"G={G} 必须是 2 的幂"
        assert 0 < k < G
        assert 0 < flip_prob < 0.5

        self.G = G
        self.k = k
        self.flip_prob = flip_prob
        self.n = int(np.log2(G))

        # 预计算均匀 p 的巴氏参数（无 embed 时使用）
        self._Z_uniform = self._bhattacharyya_uniform(flip_prob, G)
        self._info_pos_uniform = np.argsort(self._Z_uniform)[:k]

    # ──────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────

    def enroll(self, re: np.ndarray,
               embed_e: Optional[np.ndarray] = None) -> Tuple[str, bytes]:
        """
        注册。

        Args:
            re:      (G,) 可撤销模板 {-1,+1} 或 {0,1}
            embed_e: (G,) tanh 连续值（置信度），可选
                     传入时融入巴氏参数计算，选出更好的信息位

        Returns:
            secure_template: "sha256|h_hex|info_pos_hex|p_eff_hex"
            s_bytes:         密钥字节（分析用）
        """
        re_bits = self._to_bits(re)

        # 1. 计算信息位和 p_eff
        info_positions, p_eff = self._get_info_positions(embed_e)

        # 2. 生成密钥 s（k bits）
        rng = np.random.default_rng(
            int.from_bytes(re_bits[:4].tobytes(), 'big')
        )
        s = rng.integers(0, 2, self.k, dtype=np.uint8)

        # 3. 构造 u
        u = np.zeros(self.G, dtype=np.uint8)
        u[info_positions] = s

        # 4. 极化编码 x = u·G_N
        x = self._polar_encode(u)

        # 5. 辅助数据
        h = re_bits ^ x
        s_bytes = np.packbits(s).tobytes()

        # 6. 存储（p_eff 量化为 uint16）
        p_eff_q = (p_eff * 10000).astype(np.uint16)
        secure_hash = hashlib.sha256(s_bytes).hexdigest()
        secure_template = (
            f"{secure_hash}|"
            f"{h.tobytes().hex()}|"
            f"{info_positions.astype(np.uint16).tobytes().hex()}|"
            f"{p_eff_q.tobytes().hex()}"
        )
        return secure_template, s_bytes

    def authenticate(self, rp: np.ndarray,
                     stored_template: str,
                     embed_p: Optional[np.ndarray] = None
                     ) -> Tuple[bool, bytes]:
        """
        认证：SC 译码恢复密钥。

        Args:
            rp:              (G,) 探针可撤销模板
            stored_template: 注册时的存储字符串
            embed_p:         (G,) 探针 tanh 值，可选
                             传入时用于计算非均匀 LLR，提升软判决精度

        Returns:
            is_genuine: 认证是否通过
            s_recovered: 恢复的密钥字节
        """
        rp_bits = self._to_bits(rp)

        # 解析存储
        parts = stored_template.split('|')
        stored_hash = parts[0]
        h = np.frombuffer(bytes.fromhex(parts[1]), dtype=np.uint8)
        info_positions = np.frombuffer(
            bytes.fromhex(parts[2]), dtype=np.uint16
        ).astype(np.int64)
        p_eff_stored = (
            np.frombuffer(bytes.fromhex(parts[3]), dtype=np.uint16)
            .astype(np.float64) / 10000.0
        )

        # 1. 含噪码字
        x_noisy = rp_bits ^ h

        # 2. 计算 LLR
        # 优先用探针自己的 embed_p 计算 p_eff（更准确）
        if embed_p is not None:
            _, p_eff_auth = self._get_info_positions(embed_p)
        else:
            p_eff_auth = p_eff_stored

        llr = self._compute_llr(x_noisy, p_eff_auth)

        # 3. SC 译码
        u_hat = self._sc_decode(llr, info_positions)

        # 4. 恢复密钥
        s_hat_bytes = np.packbits(u_hat[info_positions]).tobytes()

        # 5. 比对
        is_genuine = (hashlib.sha256(s_hat_bytes).hexdigest() == stored_hash)
        return is_genuine, s_hat_bytes

    def get_security_bits(self) -> int:
        return self.k

    def get_code_rate(self) -> float:
        return self.k / self.G

    # ──────────────────────────────────────────────
    # 极化码核心算法
    # ──────────────────────────────────────────────

    def _polar_encode(self, u: np.ndarray) -> np.ndarray:
        """
        极化编码 x = u·G_N（G_N = F^{⊗n}，F=[[1,0],[1,1]]）
        G_N 是自逆矩阵，蝶形运算，O(N log N)。
        """
        x = u.copy().astype(np.uint8)
        N = len(x)
        step = 1
        while step < N:
            for j in range(0, N, step * 2):
                x[j:j+step] ^= x[j+step:j+step*2]
            step *= 2
        return x

    def _sc_decode(self, llr: np.ndarray,
                   info_positions: np.ndarray) -> np.ndarray:
        """
        SC（Successive Cancellation）译码器。

        LLR 传播规则（minSum 近似）：
          f(a,b) = sign(a)*sign(b)*min(|a|,|b|)    ← 左子树
          g(a,b,u) = b + (1-2u)*a                   ← 右子树（利用左子树判决）
        """
        N = self.G
        n = self.n

        frozen_mask = np.ones(N, dtype=bool)
        frozen_mask[info_positions] = False  # False=信息位，True=冻结位

        # alpha[depth]：depth 层各节点的 LLR（全局 N 维数组）
        # depth=n 为根（信道 LLR），depth=0 为叶（单比特判决）
        alpha = np.zeros((n + 1, N))
        alpha[n] = llr.copy()
        beta = np.zeros((n + 1, N), dtype=np.uint8)
        u_hat = np.zeros(N, dtype=np.uint8)

        self._sc_recursive(alpha, beta, u_hat, frozen_mask, 0, N, n)
        return u_hat

    def _sc_recursive(self, alpha, beta, u_hat,
                      frozen_mask, start, size, depth):
        """SC 递归核心，处理 [start, start+size) 子树。"""
        if size == 1:
            idx = start
            if frozen_mask[idx]:
                u_hat[idx] = 0
                beta[0][idx] = 0
            else:
                u_hat[idx] = 0 if alpha[0][idx] >= 0 else 1
                beta[0][idx] = u_hat[idx]
            return

        half = size // 2
        cd = depth - 1  # child depth

        # 左子树：f 函数
        a = alpha[depth][start:start+half]
        b = alpha[depth][start+half:start+size]
        alpha[cd][start:start+half] = (
            np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        )
        self._sc_recursive(alpha, beta, u_hat, frozen_mask, start, half, cd)

        # 右子树：g 函数（利用左子树判决）
        u_l = beta[cd][start:start+half].astype(np.float64)
        alpha[cd][start+half:start+size] = b + (1 - 2 * u_l) * a
        self._sc_recursive(alpha, beta, u_hat, frozen_mask, start+half, half, cd)

        # 回传
        ul = beta[cd][start:start+half]
        ur = beta[cd][start+half:start+size]
        beta[depth][start:start+half]      = ul ^ ur
        beta[depth][start+half:start+size] = ur

    def _compute_llr(self, y: np.ndarray,
                     p_eff: np.ndarray) -> np.ndarray:
        """
        非均匀 BSC 信道 LLR：L(y_i) = (1-2y_i)*log((1-p_i)/p_i)
        p_eff[i] 越小（信道越可靠）→ LLR 幅度越大 → 判决越确定。
        """
        p_eff = np.clip(p_eff, 1e-6, 1 - 1e-6)
        llr_per_bit = np.log((1 - p_eff) / p_eff)
        return (1 - 2 * y.astype(np.float64)) * llr_per_bit

    # ──────────────────────────────────────────────
    # 信道可靠性评估（融入生物特征置信度）
    # ──────────────────────────────────────────────

    def _get_info_positions(self,
                            embed: Optional[np.ndarray] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        确定信息位位置，返回 (info_positions, p_eff)。

        embed=None → 均匀 p，标准极化码
        embed 传入 → 用 |embed[i]| 调整 p_eff[i]，
                     再用非均匀 p_eff 计算巴氏参数，
                     使密钥嵌入"置信度高且极化可靠"的子信道。
        """
        if embed is None:
            p_eff = np.full(self.G, self.flip_prob)
            return self._info_pos_uniform.copy(), p_eff

        embed = np.asarray(embed, dtype=np.float32)
        assert len(embed) == self.G

        # 用置信度调整有效翻转率
        confidence = np.abs(embed).astype(np.float64)
        confidence = np.clip(confidence, 0.0, 1.0)
        p_eff = self.flip_prob * (1 - confidence * 0.8)
        p_eff = np.clip(p_eff, 1e-4, 0.4999)

        # 非均匀巴氏参数
        Z_init = 2 * np.sqrt(p_eff * (1 - p_eff))
        Z_polar = self._bhattacharyya_nonuniform(Z_init)
        info_positions = np.argsort(Z_polar)[:self.k]

        return info_positions, p_eff

    @staticmethod
    def _bhattacharyya_uniform(p: float, N: int) -> np.ndarray:
        """
        均匀 BSC(p) 极化后各子信道的巴氏参数。
        初始 Z=2√(p(1-p))，递推：
          Z+ = 2Z-Z²（合并，更可靠），Z- = Z²（分裂，更不可靠）
        """
        Z = np.array([2 * np.sqrt(p * (1 - p))])
        n = int(np.log2(N))
        for _ in range(n):
            Z_new = np.zeros(len(Z) * 2)
            for i, z in enumerate(Z):
                Z_new[2*i]   = 2*z - z**2
                Z_new[2*i+1] = z**2
            Z = Z_new
        return Z

    @staticmethod
    def _bhattacharyya_nonuniform(Z_init: np.ndarray) -> np.ndarray:
        """
        非均匀 BSC 信道（每位置 p_eff[i] 不同）的极化巴氏参数。
        相邻两信道合并的递推：
          Z_merged = 2Z1Z2 - Z1²Z2²
          Z_split  = Z1² + Z2² - Z1²Z2²
        """
        N = len(Z_init)
        assert (N & (N-1)) == 0
        Z = Z_init.copy()
        step = 1
        while step < N:
            Z_new = Z.copy()
            for j in range(0, N, step * 2):
                for i in range(step):
                    z1, z2 = Z[j+i], Z[j+i+step]
                    Z_new[j+i]        = 2*z1*z2 - z1**2*z2**2
                    Z_new[j+i+step]   = z1**2 + z2**2 - z1**2*z2**2
            Z = Z_new
            step *= 2
        return Z

    # ──────────────────────────────────────────────
    # 工具函数
    # ──────────────────────────────────────────────

    def _to_bits(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        bits = ((vec + 1) / 2).astype(np.uint8) if vec.min() < 0 else vec.astype(np.uint8)
        assert len(bits) == self.G
        return bits


# ──────────────────────────────────────────────────
# 演示和基准测试
# ──────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("SSTM_Polar Demo (Polar Code with SC Decoder)")
    print("=" * 65)

    G, k, p = 512, 120, 0.112
    sstm = SSTM_Polar(G=G, k=k, flip_prob=p)

    Z = sstm._Z_uniform
    print(f"\nG={G}, k={k}, flip_prob={p}, 码率={k/G:.3f}")
    print(f"巴氏参数: Z_min={Z.min():.2e}, Z_max={Z.max():.2f}")
    print(f"  可靠子信道数(Z<0.1): {(Z<0.1).sum()}, 选取 k={k}")

    rng = np.random.default_rng(42)
    re = rng.choice([-1, 1], G).astype(np.float32)
    embed_e = rng.uniform(-1, 1, G).astype(np.float32)
    embed_e[:G//2] = rng.uniform(0.7, 1.0, G//2) * np.sign(embed_e[:G//2])

    print("\n--- 标准极化码（无 Embedding）---")
    stored, _ = sstm.enroll(re, None)
    for n_flip in [0, 20, 40, 57, 70, 90]:
        rp = re.copy()
        if n_flip:
            rp[rng.choice(G, n_flip, replace=False)] *= -1
        r, _ = sstm.authenticate(rp, stored)
        print(f"  flip {n_flip:3d}: {'PASS ✓' if r else 'FAIL ✗'}")

    print("\n--- 极化码 + Embedding（置信度调整巴氏参数 + LLR）---")
    stored_e, _ = sstm.enroll(re, embed_e)
    for n_flip in [0, 20, 40, 57, 70, 90]:
        rp = re.copy()
        if n_flip:
            rp[rng.choice(G, n_flip, replace=False)] *= -1
        embed_p = np.clip(embed_e + rng.normal(0, 0.02, G), -1, 1).astype(np.float32)
        r, _ = sstm.authenticate(rp, stored_e, embed_p)
        print(f"  flip {n_flip:3d}: {'PASS ✓' if r else 'FAIL ✗'}")

    rp_imp = rng.choice([-1, 1], G).astype(np.float32)
    r_imp, _ = sstm.authenticate(rp_imp, stored)
    print(f"\n[冒充者] {'PASS (误报!)' if r_imp else 'FAIL ✓'}")


def benchmark(n_trials=300):
    print("\n" + "=" * 65)
    print(f"Benchmark ({n_trials} trials, G=512, k=120, p=0.112)")
    print("=" * 65)

    G, k, p = 512, 120, 0.112
    sstm = SSTM_Polar(G=G, k=k, flip_prob=p)
    rng = np.random.default_rng(123)
    r_pure, r_embed = [], []

    for _ in range(n_trials):
        re = rng.choice([-1, 1], G).astype(np.float32)
        embed_e = rng.uniform(-1, 1, G).astype(np.float32)
        embed_e[rng.choice(G, G//2, replace=False)] = (
            rng.uniform(0.7, 1.0, G//2) *
            np.sign(embed_e[rng.choice(G, G//2, replace=False)])
        )

        stored_p, _ = sstm.enroll(re, None)
        stored_e, _ = sstm.enroll(re, embed_e)

        n_flip = rng.binomial(G, p)
        flip_idx = rng.choice(G, n_flip, replace=False)
        rp = re.copy(); rp[flip_idx] *= -1
        embed_p = np.clip(embed_e + rng.normal(0, 0.02, G), -1, 1).astype(np.float32)

        ok_p, _ = sstm.authenticate(rp, stored_p, None)
        ok_e, _ = sstm.authenticate(rp, stored_e, embed_p)
        r_pure.append(ok_p)
        r_embed.append(ok_e)

    print(f"标准极化码（均匀 p）:     GAR = {np.mean(r_pure)*100:.1f}%")
    print(f"极化码 + Embedding:      GAR = {np.mean(r_embed)*100:.1f}%")


if __name__ == "__main__":
    demo()
    benchmark()
