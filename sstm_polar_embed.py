"""
SSTM_PolarEmbed: Secure Sketch Template Module
极化码 + 连续特征 Embedding 融合方案

核心创新：
  1. 极化码思想：将 G 个 bit 按可靠性分为"可靠信道"和"不可靠信道"
     - 可靠信道：嵌入密钥 s
     - 不可靠信道：冻结位（固定为 0）

  2. 连续特征 Embedding：用 tanh 输出的绝对值 |tanh(x)| 作为信道可靠性依据
     - |tanh(x)| 越大（越接近 ±1），该 bit 越稳定可靠
     - 实时、样本自适应（每次认证都根据当前输入计算可靠性）

  3. BCH 辅助纠错：在可靠信道恢复的密钥上额外做 BCH 纠错

与 BCH 方案的对比：
  BCH（现有）：密钥随机填充到前 k_bytes，均匀对待所有 bit
  PolarEmbed：密钥嵌入最可靠的 k_bits 位置，利用置信度信息

数据流：
  注册: re (G bits) + embed_e (G floats)
    → 按 |embed_e| 排序 → 可靠信道位置 perm[:k_bits]
    → 密钥 s 嵌入可靠位置 → BCH 编码 → 存储

  认证: rp (G bits) + embed_p (G floats)
    → 按存储的 perm 取可靠位置 → BCH 纠错 → 比对哈希
"""

import hashlib
import json
import numpy as np
from typing import Tuple, Optional

try:
    import bchlib
    _BCHLIB_AVAILABLE = True
except ImportError:
    _BCHLIB_AVAILABLE = False


class SSTM_PolarEmbed:
    """
    极化码 + 连续特征 Embedding 融合安全草图模板模块

    接口与 SSTM_BCH 完全兼容，多了可选的 embed_e/embed_p 参数。
    embed=None 时退化为随机选位（等价于 BCH 的部分填充方案）。
    """

    def __init__(self, G: int = 512, k_bits: int = 120, m: int = 9, t: int = 56):
        """
        Args:
            G:      可撤销模板长度（CTM 输出），必须是 8 的倍数
            k_bits: 可靠信道数 = 安全性（bits），必须是 8 的倍数
            m:      BCH 有限域阶数
            t:      BCH 纠错能力（bits）
        """
        assert G % 8 == 0, f"G={G} 必须是 8 的倍数"
        assert k_bits % 8 == 0, f"k_bits={k_bits} 必须是 8 的倍数"
        assert k_bits < G, f"k_bits={k_bits} 必须小于 G={G}"
        assert _BCHLIB_AVAILABLE, "需要安装 bchlib: pip install bchlib"

        self.G = G
        self.k_bits = k_bits
        self.k_bytes = k_bits // 8

        # 初始化 BCH 编解码器
        self.bch = bchlib.BCH(t=t, m=m)
        self.t = self.bch.t
        self.m = m
        self.ecc_bytes = self.bch.ecc_bytes

        # 验证 BCH 参数与 k_bytes 兼容
        max_data_bytes = (self.bch.n - self.bch.ecc_bits) // 8
        assert self.k_bytes <= max_data_bytes, (
            f"k_bytes={self.k_bytes} 超过 BCH(m={m},t={t}) 最大数据长度 {max_data_bytes}"
        )

    def enroll(self, re: np.ndarray,
               embed_e: Optional[np.ndarray] = None) -> Tuple[str, bytes]:
        """
        注册阶段：极化码 + Embedding 模糊承诺

        Args:
            re:      (G,) 可撤销模板，值为 {-1,+1} 或 {0,1}
            embed_e: (G,) tanh 连续值（置信度），None 时随机选位

        Returns:
            secure_template: 存储字符串
                             格式: "sha256|h_hex|ecc_hex|perm_json"
            s:               密钥字节（仅用于分析）
        """
        re_bits = self._to_bits(re)  # (G,) → {0,1}

        # 1. 计算信道可靠性，确定可靠/不可靠信道
        perm = self._get_reliability_order(embed_e)
        # perm[0..k_bits-1] 是最可靠的位置（放密钥）
        # perm[k_bits..G-1] 是不可靠位置（冻结位=0）

        # 2. 生成随机密钥 s（k_bytes 字节）
        rng = np.random.default_rng(int.from_bytes(re_bits[:4].tobytes(), 'big'))
        s = bytes(rng.integers(0, 256, self.k_bytes).tolist())

        # 3. BCH 编码密钥
        ecc = self.bch.encode(s)

        # 4. 构造 s_mapped（G bits）：可靠位置嵌入 s，不可靠位置为 0
        s_bits = np.unpackbits(np.frombuffer(s, dtype=np.uint8))[:self.k_bits]
        s_mapped = np.zeros(self.G, dtype=np.uint8)
        s_mapped[perm[:self.k_bits]] = s_bits

        # 5. 辅助数据 h = re_bits XOR s_mapped
        h_bits = re_bits ^ s_mapped
        h_bytes = np.packbits(h_bits).tobytes()

        # 6. 存储
        secure_hash = hashlib.sha256(s).hexdigest()
        secure_template = (
            f"{secure_hash}|"
            f"{h_bytes.hex()}|"
            f"{ecc.hex()}|"
            f"{json.dumps(perm.tolist())}"
        )
        return secure_template, s

    def authenticate(self, rp: np.ndarray,
                     stored_template: str,
                     embed_p: Optional[np.ndarray] = None) -> Tuple[bool, bytes]:
        """
        认证阶段：极化码译码 + BCH 纠错

        Args:
            rp:              (G,) 探针可撤销模板
            stored_template: 注册时的存储字符串
            embed_p:         (G,) 探针的 tanh 连续值（当前未使用，预留接口）

        Returns:
            is_genuine: 是否认证通过
            s_recovered: 恢复的密钥（调试用）
        """
        rp_bits = self._to_bits(rp)  # (G,) → {0,1}

        # 解析存储
        parts = stored_template.split('|')
        stored_hash = parts[0]
        h_bytes = bytes.fromhex(parts[1])
        ecc = bytes.fromhex(parts[2])
        perm = np.array(json.loads(parts[3]), dtype=np.int64)

        # 1. 恢复含噪的 s_mapped
        h_bits = np.unpackbits(np.frombuffer(h_bytes, dtype=np.uint8))[:self.G]
        sp_noisy = rp_bits ^ h_bits  # (G,)，含噪

        # 2. 取可靠信道位置的 k_bits → s_noisy
        s_noisy_bits = sp_noisy[perm[:self.k_bits]]
        # 补零到 BCH 要求的字节边界
        s_noisy_padded = np.zeros(self.k_bytes * 8, dtype=np.uint8)
        s_noisy_padded[:self.k_bits] = s_noisy_bits
        s_noisy = bytearray(np.packbits(s_noisy_padded).tobytes()[:self.k_bytes])

        # 3. BCH 纠错
        ecc_work = bytearray(ecc)
        nerr = self.bch.decode(bytes(s_noisy), bytes(ecc_work))

        if nerr < 0:
            # 纠错失败
            return False, bytes(self.k_bytes)

        self.bch.correct(s_noisy, ecc_work)
        s_recovered = bytes(s_noisy)

        # 4. 比对哈希
        probe_hash = hashlib.sha256(s_recovered).hexdigest()
        is_genuine = (probe_hash == stored_hash)
        return is_genuine, s_recovered

    def get_security_bits(self) -> int:
        """返回安全性 k bits"""
        return self.k_bits

    def get_effective_correction_capacity(self) -> float:
        """
        有效纠错能力（等效到 G bits 模板上的翻转容忍数）
        t_eff = t * G / k_bits
        """
        return self.t * self.G / self.k_bits

    def _get_reliability_order(self, embed: Optional[np.ndarray]) -> np.ndarray:
        """
        根据 embed 计算信道可靠性排序。

        Args:
            embed: (G,) tanh 连续值，|embed[i]| 越大越可靠
                   None 时返回随机排列（退化为随机选位）

        Returns:
            perm: (G,) 索引排列，perm[0] 是最可靠的位置
        """
        if embed is None:
            # 退化：随机排列（等价于 BCH 的随机选位）
            rng = np.random.default_rng()
            return rng.permutation(self.G)
        else:
            embed = np.asarray(embed, dtype=np.float32)
            assert len(embed) == self.G, f"embed 长度 {len(embed)} != G={self.G}"
            # |embed[i]| 越大越可靠，降序排列
            return np.argsort(-np.abs(embed))

    def _to_bits(self, vec: np.ndarray) -> np.ndarray:
        """将 {-1,+1} 或 {0,1} 向量转为 {0,1} bits 数组"""
        vec = np.asarray(vec)
        if vec.min() < 0:
            # {-1,+1} → {0,1}
            bits = ((vec + 1) / 2).astype(np.uint8)
        else:
            bits = vec.astype(np.uint8)
        assert len(bits) == self.G
        return bits


def demo_polar_embed():
    """演示 SSTM_PolarEmbed 的注册和认证流程"""
    print("=" * 65)
    print("SSTM_PolarEmbed Demo (Polar Code + Embedding, G=512)")
    print("=" * 65)

    G = 512
    k_bits = 120  # 安全性
    sstm = SSTM_PolarEmbed(G=G, k_bits=k_bits, m=9, t=56)
    t_eff = sstm.get_effective_correction_capacity()

    print(f"\n参数: G={G}, k_bits={k_bits}, BCH(m={sstm.m},t={sstm.t})")
    print(f"有效纠错能力: t_eff = {t_eff:.1f} bits")
    print(f"  = t={sstm.t} * G={G} / k={k_bits}")

    rng = np.random.default_rng(42)

    # 模拟注册：二值码 + tanh 连续值
    re = rng.choice([-1, 1], size=G).astype(np.float32)
    embed_e = rng.uniform(-1, 1, size=G).astype(np.float32)
    # 让一半 bit 有高置信度，一半低置信度
    embed_e[:G//2] = rng.uniform(0.7, 1.0, size=G//2) * np.sign(embed_e[:G//2])

    stored, s = sstm.enroll(re, embed_e)
    print(f"\n[注册] 密钥长度: {len(s)} bytes ({len(s)*8} bits)")
    print(f"[注册] 模板前30字符: {stored[:30]}...")

    # 认证测试
    print(f"\n认证测试（随机翻转）:")
    for n_flip in [0, 20, 50, 80, 102, 110, 150]:
        rp = re.copy()
        if n_flip > 0:
            idx = rng.choice(G, size=n_flip, replace=False)
            rp[idx] = -rp[idx]
        embed_p = embed_e.copy()  # 实际中应用探针图像的 tanh 输出

        result, _ = sstm.authenticate(rp, stored, embed_p)
        status = "PASS ✓" if result else "FAIL ✗"
        print(f"  flip {n_flip:3d} bits: {status}")

    # 对比：不使用 embed（退化为随机选位）
    print(f"\n对比：embed=None（退化为随机选位）")
    stored_no_embed, _ = sstm.enroll(re, embed_e=None)
    for n_flip in [0, 50, 102]:
        rp = re.copy()
        if n_flip > 0:
            idx = rng.choice(G, size=n_flip, replace=False)
            rp[idx] = -rp[idx]
        result, _ = sstm.authenticate(rp, stored_no_embed, embed_p=None)
        status = "PASS ✓" if result else "FAIL ✗"
        print(f"  flip {n_flip:3d} bits: {status}")

    # 冒充者测试
    rp_imp = rng.choice([-1, 1], size=G).astype(np.float32)
    embed_imp = rng.uniform(-1, 1, size=G).astype(np.float32)
    result_imp, _ = sstm.authenticate(rp_imp, stored, embed_imp)
    print(f"\n[冒充者] 结果: {'PASS (误报!)' if result_imp else 'FAIL ✓ (正确)'}")


if __name__ == "__main__":
    demo_polar_embed()
