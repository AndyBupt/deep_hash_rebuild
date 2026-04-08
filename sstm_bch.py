"""
SSTM_BCH: Secure Sketch Template Module (BCH Code Version)
对应论文改进实验：用 BCH 码替换 RS 码，实现比特级纠错

核心区别：
  RS 码（原版 sstm.py）：按符号（8 bits）纠错
    - 随机翻转 102 bits → 约 54 个符号出错（符号错误率 84%）
    - 远超 RS 纠错能力，GAR 近似为 0

  BCH 码（本文件）：按比特纠错
    - 随机翻转 102 bits → 仅 ~35 个比特出错（在 s 的范围内）
    - 在 t=56 bits 纠错能力范围内，GAR 显著提升

方案设计（部分填充模糊承诺）：
  注册:
    re (G bits) → 生成随机密钥 s (k_bytes)
    → BCH 编码: ecc = BCH.encode(s)
    → s_padded = s || 0...0 (G bits 总长)
    → 辅助数据: h = re XOR s_padded
    → 存储: SHA256(s) + h + ecc

  认证:
    rp (G bits) → sp_padded_noisy = rp XOR h
    → 取前 k_bytes: s_noisy
    → BCH 解码纠错: s' = BCH.correct(s_noisy, ecc)
    → 比对: SHA256(s') == SHA256(s)

关键参数（G=512 bits）：
  BCH(m=9, t=56): 纠错能力 56 bits
    - k = 15 bytes = 120 bits（安全性）
    - ecc = 63 bytes（额外存储，不占 G）
    - 期望 s 内翻转数 = 102 * 120/512 ≈ 24 bits < t=56 ✓

  BCH(m=8, t=31): 纠错能力 31 bits
    - k = 6 bytes = 48 bits（安全性）
    - ecc = 31 bytes
    - 期望 s 内翻转数 = 102 * 48/512 ≈ 10 bits < t=31 ✓
"""

import hashlib
import numpy as np
from typing import Tuple, Optional

try:
    import bchlib
    _BCHLIB_AVAILABLE = True
except ImportError:
    _BCHLIB_AVAILABLE = False


# 预定义参数组合，对应不同安全级别
# (m, t) -> (k_bits, ecc_bytes, 纠错能力)
BCH_PRESETS = {
    # G=512 可用的参数（k_bytes + ecc_bytes 不超过 G/8=64）
    # 注意：ecc 单独存储，不占 G
    "low":    (8, 10),   # t=10,  k=22bytes=176bits, ecc=10bytes
    "medium": (8, 20),   # t=20,  k=14bytes=112bits, ecc=20bytes
    "high":   (8, 31),   # t=31,  k=6bytes=48bits,   ecc=31bytes
    "m9_t15": (9, 15),   # t=15,  k=47bytes=376bits, ecc=17bytes
    "m9_t28": (9, 28),   # t=28,  k=34bytes=272bits, ecc=34bytes
    "m9_t56": (9, 56),   # t=56,  k=15bytes=120bits, ecc=63bytes
}


class SSTM_BCH:
    """
    安全草图模板模块 - BCH 码版本（比特级纠错）

    与 sstm.py（RS 码版本）的对比：
    - RS 码按符号（8 bits）纠错，随机翻转的比特放大为符号错误
    - BCH 码按比特纠错，直接处理比特翻转，更适合随机翻转场景

    实现方案：部分填充模糊承诺（Partial Fuzzy Commitment）
    - 密钥 s 只占 G bits 的前 k bits，其余补零
    - 认证时只需纠正 s 范围内的翻转（约 102 * k/G bits）
    - ecc 单独存储，不占用 G bits
    """

    def __init__(self, G: int = 512, m: int = 9, t: int = 56):
        """
        Args:
            G:  可撤销模板比特数（CTM 输出），必须是 8 的倍数
            m:  BCH 有限域阶数（GF(2^m)），决定码字长度 n = 2^m - 1
            t:  纠错能力（比特数），越大纠错越强但 ecc 越长

        推荐参数（G=512）：
            m=9, t=56: 纠 56 bits，k=120 bits 安全性，ecc=63 bytes
            m=9, t=28: 纠 28 bits，k=272 bits 安全性，ecc=34 bytes
            m=8, t=31: 纠 31 bits，k=48 bits 安全性，ecc=31 bytes
        """
        assert G % 8 == 0, f"G={G} 必须是 8 的倍数"
        assert _BCHLIB_AVAILABLE, (
            "需要安装 bchlib 库: pip install bchlib\n"
            "BCH 码比特级纠错实现"
        )

        self.G = G
        self.G_bytes = G // 8

        # 初始化 BCH 编解码器
        self.bch = bchlib.BCH(t=t, m=m)
        self.m = m
        self.t = self.bch.t            # 实际纠错能力（bits）
        self.ecc_bytes = self.bch.ecc_bytes

        # 计算最大信息字节数（BCH 缩短码）
        # k_bytes 可以任意 <= (n - ecc_bits) // 8
        self.k_bytes = (self.bch.n - self.bch.ecc_bits) // 8
        self.k_bits = self.k_bytes * 8  # 安全性

        # 期望翻转数（s 范围内）
        self.expected_flips_in_s = None  # 运行时计算

        assert self.k_bytes > 0, f"k_bytes={self.k_bytes}，参数无效"
        assert self.k_bytes <= self.G_bytes, (
            f"k_bytes={self.k_bytes} > G_bytes={self.G_bytes}，"
            f"密钥超过模板长度"
        )

    def enroll(self, re: np.ndarray) -> Tuple[str, bytes]:
        """
        注册阶段（部分填充模糊承诺）

        Args:
            re: (G,) 可撤销模板，值为 {-1,+1} 或 {0,1}

        Returns:
            secure_template: 存储字符串，格式 "sha256|h_hex|ecc_hex"
            s: 密钥字节（仅用于分析）
        """
        re_bytes = self._bits_to_bytes(re)
        assert len(re_bytes) == self.G_bytes

        # 生成随机密钥 s（k_bytes 字节）
        rng = np.random.default_rng(int.from_bytes(re_bytes[:4], 'big'))
        s = bytes(rng.integers(0, 256, self.k_bytes).tolist())

        # BCH 编码
        ecc = self.bch.encode(s)

        # 部分填充：s_padded = s || 0...0 (G_bytes 字节)
        s_padded = s + bytes(self.G_bytes - self.k_bytes)

        # 辅助数据：h = re XOR s_padded
        h = bytes(a ^ b for a, b in zip(re_bytes, s_padded))

        # 存储 SHA256(s) + h + ecc
        secure_hash = hashlib.sha256(s).hexdigest()
        secure_template = f"{secure_hash}|{h.hex()}|{ecc.hex()}"
        return secure_template, s

    def authenticate(self, rp: np.ndarray,
                     stored_template: str) -> Tuple[bool, bytes]:
        """
        认证阶段（部分填充模糊承诺）

        Args:
            rp: (G,) 探针可撤销模板
            stored_template: 注册时存储的字符串

        Returns:
            is_genuine: 是否认证通过
            s_recovered: 恢复的密钥（调试用）
        """
        rp_bytes = self._bits_to_bytes(rp)
        assert len(rp_bytes) == self.G_bytes

        # 解析存储模板
        parts = stored_template.split('|')
        stored_hash = parts[0]
        h = bytes.fromhex(parts[1])
        ecc = bytes.fromhex(parts[2])

        # 恢复含噪 s_padded
        sp_padded_noisy = bytes(a ^ b for a, b in zip(rp_bytes, h))

        # 取前 k_bytes 作为含噪密钥
        s_noisy = bytearray(sp_padded_noisy[:self.k_bytes])
        ecc_work = bytearray(ecc)

        # BCH 解码纠错
        nerr = self.bch.decode(bytes(s_noisy), bytes(ecc_work))

        if nerr < 0:
            # 纠错失败：翻转位数超过 t
            return False, bytes(self.k_bytes)

        # 应用纠错
        self.bch.correct(s_noisy, ecc_work)
        s_recovered = bytes(s_noisy)

        # 比对哈希
        probe_hash = hashlib.sha256(s_recovered).hexdigest()
        is_genuine = (probe_hash == stored_hash)
        return is_genuine, s_recovered

    def get_security_bits(self) -> int:
        """返回安全性 k bits"""
        return self.k_bits

    def get_error_correction_capacity(self) -> int:
        """返回纠错能力 t bits（比特级）"""
        return self.t

    def get_effective_correction_capacity(self) -> float:
        """
        返回对 G bits 模板的有效纠错能力（等效 bits）

        由于只纠正 s 范围内（前 k bits）的翻转，
        等效于对整个 G bits 模板能容忍的翻转数：
        t_effective = t * G / k
        """
        return self.t * self.G / self.k_bits

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """将 G bits 向量打包为字节"""
        bits = np.asarray(bits)
        if bits.min() < 0:
            bits = ((bits + 1) / 2).astype(np.uint8)
        else:
            bits = bits.astype(np.uint8)
        assert len(bits) == self.G
        return bytes(np.packbits(bits).tolist())


def compare_rs_vs_bch():
    """
    对比 RS 码（sstm.py）和 BCH 码（本文件）在相同参数下的纠错能力
    """
    print("=" * 65)
    print("RS Code vs BCH Code: Error Correction Comparison (G=512)")
    print("=" * 65)

    G = 512
    N = G // 8  # = 64 symbols for RS

    print(f"\n{'Method':<20} {'Security(bits)':>15} {'ECC overhead':>14} "
          f"{'t (bits)':>10} {'t_eff (bits)':>13}")
    print("-" * 75)

    # RS 码参数（来自 sstm.py）
    try:
        from reedsolo import RSCodec
        for K in [7, 13, 20, 38]:
            nsym = N - K
            if nsym <= 0:
                continue
            t_rs = (nsym // 2) * 8  # RS 纠错 bits（符号级）
            ecc_bits = nsym * 8
            k_bits = K * 8
            print(f"{'RS(K='+str(K)+')':20} {k_bits:>15} {ecc_bits:>14} "
                  f"{t_rs:>10} {t_rs:>13}")
    except ImportError:
        print("(reedsolo not available for RS comparison)")

    print()

    # BCH 码参数
    bch_params = [
        (9, 15),   # t=15
        (9, 28),   # t=28
        (9, 56),   # t=56
        (8, 31),   # t=31
    ]
    for m, t_req in bch_params:
        try:
            bch_obj = bchlib.BCH(t=t_req, m=m)
            k_bytes = (bch_obj.n - bch_obj.ecc_bits) // 8
            k_bits = k_bytes * 8
            ecc_bits = bch_obj.ecc_bytes * 8
            t_bch = bch_obj.t
            t_eff = t_bch * G / k_bits if k_bits > 0 else 0
            label = f"BCH(m={m},t={bch_obj.t})"
            print(f"{label:20} {k_bits:>15} {ecc_bits:>14} "
                  f"{t_bch:>10} {t_eff:>13.1f}")
        except Exception as e:
            print(f"BCH(m={m},t={t_req}): failed - {e}")

    print()
    print(f"Genuine bit flip rate (observed): ~102 bits / {G} bits = 20%")
    print(f"-> Methods with t_eff >= 102 can potentially pass authentication")


def demo_bch_sstm():
    """演示 BCH SSTM 的注册和认证流程"""
    print("=" * 60)
    print("SSTM_BCH Demo (Fuzzy Commitment with BCH Codes, G=512)")
    print("=" * 60)

    G = 512
    # 用 m=9, t=56：纠 56 bits，k=120 bits 安全性
    sstm = SSTM_BCH(G=G, m=9, t=56)
    t_eff = sstm.get_effective_correction_capacity()
    print(f"\nBCH(m={sstm.m}, t={sstm.t}): "
          f"k={sstm.k_bits} bits security, "
          f"ecc={sstm.ecc_bytes} bytes")
    print(f"Effective correction capacity: {t_eff:.1f} bits "
          f"(= t={sstm.t} * G={G} / k={sstm.k_bits})")
    print(f"Observed genuine flip rate: ~102 bits")
    print(f"-> {'PASS' if t_eff >= 102 else 'FAIL'}: "
          f"t_eff={t_eff:.1f} {'≥' if t_eff >= 102 else '<'} 102")

    rng = np.random.default_rng(42)
    re = rng.choice([-1, 1], size=G)

    stored_template, s = sstm.enroll(re)
    print(f"\n[Enroll] Key: {len(s)} bytes ({len(s)*8} bits)")
    print(f"[Enroll] Template: {stored_template[:30]}...")

    print(f"\nAuthentication tests (random bit flips):")
    for n_flip in [0, 50, 80, 102, 110, 150, 200]:
        rp = re.copy()
        if n_flip > 0:
            idx = rng.choice(G, size=n_flip, replace=False)
            rp[idx] = -rp[idx]
        result, _ = sstm.authenticate(rp, stored_template)
        status = "PASS ✓" if result else "FAIL ✗"
        print(f"  flip {n_flip:3d} bits: {status}")

    # Impostor
    rp_imp = rng.choice([-1, 1], size=G)
    result_imp, _ = sstm.authenticate(rp_imp, stored_template)
    print(f"\n[Impostor] Result: "
          f"{'PASS (false accept!)' if result_imp else 'FAIL ✓ (correct)'}")


if __name__ == "__main__":
    compare_rs_vs_bch()
    print()
    demo_bch_sstm()
