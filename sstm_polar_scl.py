"""
SSTM_PolarSCL: Secure Sketch Template Module
极化码 + SCL(Successive Cancellation List)译码 + CRC 校验

升级自 sstm_polar.py（SC 译码），改为 SCL + CRC：

SCL 译码原理：
  SC 像走单行道，判错一步则全盘皆输。
  SCL 维护 L 条候选路径（平行宇宙）：
    每遇到信息位，将每条路径分裂为"判0"和"判1"两条，
    按路径度量（log-likelihood）保留最好的 L 条。
  最终 L 条候选路径用 CRC 校验选出正确路径。

CRC 校验：
  密钥 s 后追加 r 位 CRC（如 CRC-8 或 CRC-16）。
  SCL 输出 L 条候选，第一条通过 CRC 的即为正确密钥。
  误判概率从 O(1/L) 降至 O(1/2^r)。

生物特征融入：
  1. 用 |tanh(x)| 调整有效翻转率 p_eff，精化 LLR 计算
  2. 冻结位填入可撤销模板的原始 bit（而非全零），
     实现"不可靠信道嵌入辅助生物特征信息"

参考：
  Tal & Vardy (2015). "List Decoding of Polar Codes."
  IEEE Trans. Inf. Theory, 61(5), 2213-2226.
"""

import hashlib
import numpy as np
from typing import Tuple, Optional


def _sc_decode_core(llr_ch: np.ndarray,
                    frozen_mask: np.ndarray,
                    frozen_vals: np.ndarray) -> np.ndarray:
    """
    单路径 SC 译码（正确实现，用于验证和作为 SCL 的基础）。
    alpha[depth] 存整个码字宽度的 LLR，用 start/size 定位子树。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    alpha = np.zeros((n + 1, N))
    beta  = np.zeros((n + 1, N), dtype=np.uint8)
    alpha[n] = llr_ch.copy()
    u_hat = np.zeros(N, dtype=np.uint8)

    def recurse(start, size, depth):
        if size == 1:
            idx = start
            if frozen_mask[idx]:
                u_hat[idx] = int(frozen_vals[idx])
                beta[0][idx] = u_hat[idx]
            else:
                u_hat[idx] = 0 if alpha[0][idx] >= 0 else 1
                beta[0][idx] = u_hat[idx]
            return

        half = size // 2
        cd = depth - 1
        a = alpha[depth][start:start+half]
        b = alpha[depth][start+half:start+size]

        # 左子树：f 函数
        alpha[cd][start:start+half] = (
            np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        )
        recurse(start, half, cd)

        # 右子树：g 函数（用左子树的 beta）
        ul = beta[cd][start:start+half].astype(np.float64)
        alpha[cd][start+half:start+size] = b + (1 - 2 * ul) * a
        recurse(start + half, half, cd)

        # 回传 beta
        ul = beta[cd][start:start+half]
        ur = beta[cd][start+half:start+size]
        beta[depth][start:start+half]      = ul ^ ur
        beta[depth][start+half:start+size] = ur

    recurse(0, N, n)
    return u_hat


class _SCLPath:
    """一条 SCL 路径的状态。"""
    __slots__ = ['pm', 'u_hat', 'alpha', 'beta']

    def __init__(self, N: int, n: int, llr_ch: np.ndarray):
        self.pm    = 0.0
        self.u_hat = np.zeros(N, dtype=np.uint8)
        self.alpha = np.zeros((n + 1, N))
        self.beta  = np.zeros((n + 1, N), dtype=np.uint8)
        self.alpha[n] = llr_ch.copy()

    def copy(self):
        p = object.__new__(_SCLPath)
        p.pm    = self.pm
        p.u_hat = self.u_hat.copy()
        p.alpha = self.alpha.copy()
        p.beta  = self.beta.copy()
        return p

    def get_leaf_llr(self, leaf: int, N: int, n: int) -> float:
        """计算第 leaf 位的 LLR（只读，基于当前 alpha/beta）"""
        alpha_tmp = self.alpha.copy()
        s, sz = 0, N
        for lev in range(n, 0, -1):
            h = sz >> 1
            cd = lev - 1
            a = alpha_tmp[lev][s:s+h]
            b = alpha_tmp[lev][s+h:s+sz]
            if leaf < s + h:
                alpha_tmp[cd][s:s+h] = (
                    np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
                )
                sz = h
            else:
                ul = self.beta[cd][s:s+h].astype(np.float64)
                alpha_tmp[cd][s+h:s+sz] = b + (1 - 2*ul) * a
                s += h; sz = h
        return float(alpha_tmp[0][leaf])

    def set_bit(self, leaf: int, bit: int, N: int, n: int):
        """设置叶节点判决，并向上回传 beta"""
        self.u_hat[leaf] = bit
        self.beta[0][leaf] = bit
        for lev in range(1, n + 1):
            bs     = 1 << lev
            bstart = (leaf // bs) * bs
            bh     = bs >> 1
            if leaf == bstart + bs - 1:
                ul = self.beta[lev-1][bstart:bstart+bh]
                ur = self.beta[lev-1][bstart+bh:bstart+bs]
                self.beta[lev][bstart:bstart+bh]      = ul ^ ur
                self.beta[lev][bstart+bh:bstart+bs]   = ur
            else:
                break


def _scl_decode(llr_ch: np.ndarray,
                frozen_mask: np.ndarray,
                frozen_vals: np.ndarray,
                L: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    SCL 译码，返回 L 条候选路径（按度量降序）。

    Returns:
        u_hats: (<=L, N) 各候选路径的译码结果
        pms:    (<=L,)  各路径的度量（越大越可能正确）
    """
    N = len(llr_ch)
    n = int(np.log2(N))

    paths = [_SCLPath(N, n, llr_ch)]

    for i in range(N):
        leaf_llrs = [p.get_leaf_llr(i, N, n) for p in paths]

        if frozen_mask[i]:
            # 冻结位：确定性判决
            fv = int(frozen_vals[i])
            for p, llr_i in zip(paths, leaf_llrs):
                p.pm += 0.0 if (llr_i >= 0) == (fv == 0) else -abs(llr_i)
                p.set_bit(i, fv, N, n)
        else:
            # 信息位：分裂路径
            if len(paths) * 2 <= L:
                new_paths = []
                for p, llr_i in zip(paths, leaf_llrs):
                    p1 = p.copy()
                    pm0 = p.pm + (0.0 if llr_i >= 0 else llr_i)
                    pm1 = p.pm + (0.0 if llr_i < 0 else -llr_i)
                    p.pm = pm0; p.set_bit(i, 0, N, n)
                    p1.pm = pm1; p1.set_bit(i, 1, N, n)
                    new_paths.extend([p, p1])
                paths = new_paths
            else:
                # 生成所有候选，保留最好的 L 条
                candidates = []
                for p, llr_i in zip(paths, leaf_llrs):
                    for bit in [0, 1]:
                        pc = p.copy()
                        delta = 0.0 if (llr_i >= 0) == (bit == 0) else -abs(llr_i)
                        pc.pm += delta
                        pc.set_bit(i, bit, N, n)
                        candidates.append(pc)
                candidates.sort(key=lambda x: x.pm, reverse=True)
                paths = candidates[:L]

    paths.sort(key=lambda x: x.pm, reverse=True)
    u_hats = np.array([p.u_hat for p in paths])
    pms    = np.array([p.pm    for p in paths])
    return u_hats, pms


class SSTM_PolarSCL:
    """
    极化码 + SCL 译码 + CRC 安全草图模板模块。
    接口与 SSTM_BCH / SSTM_Polar / SSTM_PolarEmbed 完全兼容。
    """

    def __init__(self, G: int = 512, k: int = 120,
                 flip_prob: float = 0.112,
                 L: int = 8,
                 crc_bits: int = 8):
        """
        Args:
            G:         码字长度，必须是 2 的幂
            k:         密钥长度（安全性 bits）
            flip_prob: BSC 信道翻转概率（per-bit 均值）
            L:         SCL 列表大小（推荐 4 或 8）
            crc_bits:  CRC 校验位数（0=不用，8=CRC-8，16=CRC-16）
        """
        assert (G & (G - 1)) == 0
        assert 0 < k
        assert k + crc_bits < G
        assert 0 < flip_prob < 0.5
        assert L >= 1
        assert crc_bits in (0, 8, 16)

        self.G        = G
        self.k        = k
        self.crc_bits = crc_bits
        self.k_total  = k + crc_bits
        self.flip_prob = flip_prob
        self.L        = L
        self.n        = int(np.log2(G))

        self._Z        = self._bhattacharyya_uniform(flip_prob, G)
        self._info_pos = np.argsort(self._Z)[:self.k_total]

        self._frozen_mask = np.ones(G, dtype=bool)
        self._frozen_mask[self._info_pos] = False

    def enroll(self, re: np.ndarray,
               embed_e: Optional[np.ndarray] = None) -> Tuple[str, bytes]:
        """
        注册。
        生物特征融入：embed_e（tanh 置信度）用于调整 p_eff，
        影响信息位的选择和认证时的 LLR 计算。
        冻结位固定为 0（标准极化码，保证 SC/SCL 译码正确性）。
        """
        re_bits = self._to_bits(re)

        rng = np.random.default_rng(
            int.from_bytes(re_bits[:4].tobytes(), 'big')
        )
        s = rng.integers(0, 2, self.k, dtype=np.uint8)
        s_with_crc = self._append_crc(s)

        # 构造 u：信息位 = s+CRC，冻结位 = 0（标准极化码）
        u = np.zeros(self.G, dtype=np.uint8)
        u[self._info_pos] = s_with_crc

        x = self._polar_encode(u)
        h = re_bits ^ x
        s_bytes = np.packbits(s).tobytes()

        p_eff   = self._get_p_eff(embed_e)
        p_eff_q = (p_eff * 10000).astype(np.uint16)

        secure_hash = hashlib.sha256(s_bytes).hexdigest()
        secure_template = (
            f"{secure_hash}|"
            f"{h.tobytes().hex()}|"
            f"{p_eff_q.tobytes().hex()}"
        )
        return secure_template, s_bytes

    def authenticate(self, rp: np.ndarray,
                     stored_template: str,
                     embed_p: Optional[np.ndarray] = None
                     ) -> Tuple[bool, bytes]:
        """认证：SCL + CRC 译码恢复密钥。"""
        rp_bits = self._to_bits(rp)

        parts = stored_template.split('|')
        stored_hash = parts[0]
        h = np.frombuffer(bytes.fromhex(parts[1]), dtype=np.uint8)
        p_eff_stored = (
            np.frombuffer(bytes.fromhex(parts[2]), dtype=np.uint16)
            .astype(np.float64) / 10000.0
        )

        x_noisy = rp_bits ^ h
        p_eff = self._get_p_eff(embed_p) if embed_p is not None else p_eff_stored
        llr = self._compute_llr(x_noisy, p_eff)

        # 冻结位先验：标准极化码用全零
        frozen_vals = np.zeros(self.G, dtype=np.uint8)

        candidates, _ = _scl_decode(llr, self._frozen_mask, frozen_vals, self.L)

        s_hat_bytes = None
        for u_hat in candidates:
            info_bits = u_hat[self._info_pos]
            s_bits    = info_bits[:self.k]
            if self.crc_bits > 0:
                crc_recv   = info_bits[self.k:self.k + self.crc_bits]
                crc_expect = self._compute_crc(s_bits, self.crc_bits)
                if np.array_equal(crc_recv, crc_expect):
                    s_hat_bytes = np.packbits(s_bits).tobytes()
                    break
            else:
                s_hat_bytes = np.packbits(s_bits).tobytes()
                break

        if s_hat_bytes is None:
            s_hat_bytes = np.packbits(candidates[0][self._info_pos][:self.k]).tobytes()

        is_genuine = (hashlib.sha256(s_hat_bytes).hexdigest() == stored_hash)
        return is_genuine, s_hat_bytes

    def get_security_bits(self) -> int:
        return self.k

    def get_code_rate(self) -> float:
        return self.k_total / self.G

    def _polar_encode(self, u: np.ndarray) -> np.ndarray:
        x = u.copy().astype(np.uint8)
        N = len(x)
        step = 1
        while step < N:
            for j in range(0, N, step * 2):
                x[j:j+step] ^= x[j+step:j+step*2]
            step *= 2
        return x

    def _compute_llr(self, y: np.ndarray, p_eff: np.ndarray) -> np.ndarray:
        p_eff = np.clip(p_eff, 1e-6, 1 - 1e-6)
        return (1 - 2 * y.astype(np.float64)) * np.log((1 - p_eff) / p_eff)

    def _get_p_eff(self, embed: Optional[np.ndarray]) -> np.ndarray:
        if embed is None:
            return np.full(self.G, self.flip_prob)
        confidence = np.clip(np.abs(embed).astype(np.float64), 0.0, 1.0)
        return np.clip(self.flip_prob * (1 - confidence * 0.8), 1e-4, 0.4999)

    def _append_crc(self, s: np.ndarray) -> np.ndarray:
        if self.crc_bits == 0:
            return s
        return np.concatenate([s, self._compute_crc(s, self.crc_bits)])

    @staticmethod
    def _compute_crc(data_bits: np.ndarray, crc_bits: int) -> np.ndarray:
        if crc_bits == 8:
            poly, init, width = 0x07, 0x00, 8
        elif crc_bits == 16:
            poly, init, width = 0x1021, 0xFFFF, 16
        else:
            raise ValueError(f"Unsupported CRC bits: {crc_bits}")
        crc = init; mask = (1 << width) - 1
        for bit in data_bits:
            crc ^= (int(bit) << (width - 1))
            for _ in range(width):
                crc = ((crc << 1) ^ poly) & mask if crc & (1 << (width-1)) \
                      else (crc << 1) & mask
        return np.array([(crc >> i) & 1 for i in range(width-1, -1, -1)],
                        dtype=np.uint8)

    @staticmethod
    def _bhattacharyya_uniform(p: float, N: int) -> np.ndarray:
        Z = np.array([2 * np.sqrt(p * (1 - p))])
        n = int(np.log2(N))
        for _ in range(n):
            Z_new = np.zeros(len(Z) * 2)
            for i, z in enumerate(Z):
                Z_new[2*i] = 2*z - z**2; Z_new[2*i+1] = z**2
            Z = Z_new
        return Z

    def _to_bits(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        bits = ((vec+1)/2).astype(np.uint8) if vec.min() < 0 else vec.astype(np.uint8)
        assert len(bits) == self.G
        return bits


def demo():
    print("=" * 65)
    print("SSTM_PolarSCL Demo (SCL L=8, CRC-8)")
    print("=" * 65)

    G, k, p, L, crc = 512, 120, 0.112, 8, 8
    sstm = SSTM_PolarSCL(G=G, k=k, flip_prob=p, L=L, crc_bits=crc)
    print(f"\nG={G}, k={k}, CRC={crc}bits, L={L}, k_total={k+crc}")

    rng = np.random.default_rng(42)
    re = rng.choice([-1, 1], G).astype(np.float32)
    embed_e = rng.uniform(-1, 1, G).astype(np.float32)
    embed_e[:G//2] = rng.uniform(0.7, 1.0, G//2) * np.sign(embed_e[:G//2])

    stored, _ = sstm.enroll(re, embed_e)
    print(f"[注册完成] 密钥 {k} bits + CRC {crc} bits")

    print("\n认证测试:")
    for n_flip in [0, 20, 40, 57, 70, 90, 110]:
        rp = re.copy()
        if n_flip:
            rp[rng.choice(G, n_flip, replace=False)] *= -1
        embed_p = np.clip(embed_e + rng.normal(0, 0.02, G), -1, 1).astype(np.float32)
        r, _ = sstm.authenticate(rp, stored, embed_p)
        print(f"  flip {n_flip:3d}: {'PASS ✓' if r else 'FAIL ✗'}")

    rp_imp = rng.choice([-1, 1], G).astype(np.float32)
    r_imp, _ = sstm.authenticate(rp_imp, stored)
    print(f"\n[冒充者] {'PASS (误报!)' if r_imp else 'FAIL ✓'}")


def benchmark(n_trials=100):
    print("\n" + "=" * 65)
    print(f"Benchmark ({n_trials} trials, G=512, k=120, p=0.112)")
    print("=" * 65)

    from sstm_polar import SSTM_Polar

    G, k, p = 512, 120, 0.112
    rng = np.random.default_rng(123)

    configs = [
        ("SC  (no CRC)",    SSTM_Polar(G=G, k=k, flip_prob=p)),
        ("SCL L=4 CRC-8",   SSTM_PolarSCL(G=G, k=k, flip_prob=p, L=4, crc_bits=8)),
        ("SCL L=8 CRC-8",   SSTM_PolarSCL(G=G, k=k, flip_prob=p, L=8, crc_bits=8)),
        ("SCL L=8 CRC-16",  SSTM_PolarSCL(G=G, k=k, flip_prob=p, L=8, crc_bits=16)),
    ]
    results = {name: [] for name, _ in configs}

    for trial in range(n_trials):
        re = rng.choice([-1, 1], G).astype(np.float32)
        embed_e = rng.uniform(-1, 1, G).astype(np.float32)
        embed_e[rng.choice(G, G//2, replace=False)] = (
            rng.uniform(0.7, 1.0, G//2) * np.sign(rng.uniform(-1, 1, G//2))
        )
        n_flip = rng.binomial(G, p)
        flip_idx = rng.choice(G, n_flip, replace=False)
        rp = re.copy(); rp[flip_idx] *= -1
        embed_p = np.clip(embed_e + rng.normal(0, 0.02, G), -1, 1).astype(np.float32)

        for name, sstm in configs:
            try:
                stored, _ = sstm.enroll(re, embed_e)
                ok, _ = sstm.authenticate(rp, stored, embed_p)
            except Exception:
                ok = False
            results[name].append(ok)

        if (trial+1) % 25 == 0:
            print(f"  {trial+1}/{n_trials} ...")

    print(f"\n{'方法':<22} {'GAR':>8}")
    print("-" * 32)
    for name, _ in configs:
        print(f"{name:<22} {np.mean(results[name])*100:>7.1f}%")


if __name__ == "__main__":
    demo()
    print("\n运行基准测试...")
    benchmark(n_trials=50)
