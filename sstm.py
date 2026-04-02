"""
SSTM: Secure Sketch Template Module（安全草图模板模块）
对应论文 Section III-C 和 Section IV-E

流程（模糊承诺方案 / Fuzzy Commitment）:
  注册: re (G bits) → 生成随机密钥 s (K bytes) → RS编码 c → 计算辅助数据 h = re XOR c
        → SHA256(s) 存入数据库，h 作为公开辅助数据
  认证: rp (G bits) → 计算 c' = rp XOR h → RS解码 c' → 恢复 s' → 比对 SHA256(s') == SHA256(s)

论文 RS 码参数 (Section IV-E):
  - GF(2^8)，m=8，每符号8 bits（1字节）
  - 最大码字长度 N'=255 符号
  - 使用缩短 RS 码: 实际码字长度 N 符号 (N < N')
  - 信息符号数 K，安全性 k = 8*K bits
  - 纠错能力 t = (N-K)/2 符号 = (N-K)/2 * 8 bits
  - 可撤销模板长度 G = N*8 bits = N 字节
"""

import hashlib
import numpy as np
from typing import Tuple

try:
    from reedsolo import RSCodec, ReedSolomonError
    _REEDSOLO_AVAILABLE = True
except ImportError:
    _REEDSOLO_AVAILABLE = False


class SSTM:
    """
    安全草图模板模块（使用真正的 Reed-Solomon 码，模糊承诺方案）

    对应论文 Section III-C / IV-E:
    "Due to its maximum distance separable (MDS) property, we have selected
    Reed-Solomon (RS) codes and used RS decoder for FEC decoding in SSTM."

    实现方案：模糊承诺（Fuzzy Commitment）
    - 注册时生成随机密钥 s，RS编码后与模板 XOR 得到辅助数据 h
    - 认证时用探针与 h XOR 恢复含噪码字，RS解码纠错后比对哈希
    - 只要翻转位数 ≤ 纠错能力 t，就能恢复相同的 s
    """

    def __init__(self, G: int = 768, K: int = 13):
        """
        Args:
            G: 可撤销模板比特数（CTM输出），必须是8的倍数
               论文实验: G ∈ {128, 256, 512, 768}
            K: 信息符号数（每符号8bits），安全性 k = 8*K bits
               论文实验: K=13 → k=104 bits

        内部推导:
            N = G // 8        (码字符号数，即缩短RS码的实际长度)
            nsym = N - K      (纠错符号数)
            纠错能力 t = nsym // 2 符号 = nsym // 2 * 8 bits
        """
        assert G % 8 == 0, f"G={G} 必须是8的倍数（每符号8bits）"
        assert _REEDSOLO_AVAILABLE, (
            "需要安装 reedsolo 库: pip install reedsolo\n"
            "这是论文使用的真正 Reed-Solomon 码实现"
        )

        self.G = G
        self.K = K
        self.N = G // 8          # 码字符号数
        self.k_bits = K * 8      # 安全性（bits）
        self.nsym = self.N - K   # 纠错符号数

        assert self.nsym > 0, f"N={self.N} 必须大于 K={K}"
        assert self.nsym < 255, f"nsym={self.nsym} 必须小于255（GF(2^8)限制）"

        # 初始化 RS 编解码器
        # nsize=255: GF(2^8) 最大符号数
        # c_exp=8: 每符号8bits
        self.rsc = RSCodec(nsym=self.nsym, nsize=255, c_exp=8)

    def enroll(self, re: np.ndarray) -> Tuple[str, bytes]:
        """
        注册阶段: 生成安全模板（模糊承诺方案）

        Args:
            re: (G,) 可撤销模板，值为 {-1,+1} 或 {0,1}

        Returns:
            secure_template: (SHA256哈希, 辅助数据) 元组的字符串表示
                             格式: "sha256_hex|helper_data_hex"
            se_bytes: K字节密钥（仅用于分析，实际不存储）
        """
        re_bytes = self._bits_to_bytes(re)  # G bits → N 字节
        assert len(re_bytes) == self.N, \
            f"模板字节数 {len(re_bytes)} 与 N={self.N} 不匹配"

        # 生成随机密钥 s (K 字节)
        rng = np.random.default_rng(int.from_bytes(re_bytes[:4], 'big'))
        s = bytes(rng.integers(0, 256, self.K).tolist())

        # RS 编码: s → c (N 字节码字)
        c = bytes(self.rsc.encode(s))
        assert len(c) == self.N, f"RS编码输出长度 {len(c)} != N={self.N}"

        # 辅助数据: h = re XOR c（公开存储，不泄露 re 或 s）
        h = bytes(a ^ b for a, b in zip(re_bytes, c))

        # 加密哈希: SHA256(s) 存入数据库
        secure_hash = hashlib.sha256(s).hexdigest()

        # 返回格式: "sha256_hex|helper_data_hex"
        secure_template = f"{secure_hash}|{h.hex()}"
        return secure_template, s

    def authenticate(self, rp: np.ndarray,
                     stored_template: str) -> Tuple[bool, bytes]:
        """
        认证阶段: 验证探针是否匹配（模糊承诺方案）

        Args:
            rp: (G,) 探针可撤销模板
            stored_template: 注册时存储的模板字符串（"sha256|helper_hex"）

        Returns:
            is_genuine: 是否认证通过
            sp_bytes: 恢复的密钥字节（调试用）
        """
        rp_bytes = self._bits_to_bytes(rp)
        assert len(rp_bytes) == self.N

        # 解析存储的模板
        stored_hash, h_hex = stored_template.split('|')
        h = bytes.fromhex(h_hex)

        # 恢复含噪码字: c' = rp XOR h
        c_noisy = bytes(a ^ b for a, b in zip(rp_bytes, h))

        # RS 解码: 纠错后恢复密钥 s'
        try:
            decoded, _, _ = self.rsc.decode(c_noisy)
            sp_bytes = bytes(decoded)
        except ReedSolomonError:
            # RS 解码失败：翻转位数超过纠错能力，直接拒绝认证
            return False, b'\x00' * self.K

        # 比对哈希: SHA256(s') == SHA256(s)
        probe_hash = hashlib.sha256(sp_bytes).hexdigest()
        is_genuine = (probe_hash == stored_hash)
        return is_genuine, sp_bytes

    def get_security_bits(self) -> int:
        """返回系统安全性 k = 8*K bits"""
        return self.k_bits

    def get_error_correction_capacity(self) -> int:
        """返回可纠错符号数 t = nsym//2，对应 t*8 bits"""
        return self.nsym // 2

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """将 G bits 的二值向量打包为 N=G//8 字节"""
        bits = np.asarray(bits)
        # 统一转为 {0,1}
        if bits.min() < 0:
            bits = ((bits + 1) / 2).astype(np.uint8)
        else:
            bits = bits.astype(np.uint8)
        assert len(bits) == self.G
        return bytes(np.packbits(bits).tolist())


def demo_sstm():
    """演示 SSTM 的注册和认证流程"""
    print("=" * 55)
    print("SSTM Demo (Fuzzy Commitment with Reed-Solomon Codes)")
    print("=" * 55)

    # 论文参数: G=768, K=13 → k=104 bits, 纠错能力 t=41 符号
    sstm = SSTM(G=768, K=13)
    print(f"Codeword length:  N={sstm.N} symbols ({sstm.G} bits)")
    print(f"Info length:      K={sstm.K} symbols ({sstm.k_bits} bits) <- security")
    print(f"Error correction: t={sstm.get_error_correction_capacity()} symbols "
          f"= {sstm.get_error_correction_capacity()*8} bits")

    rng = np.random.default_rng(42)
    re = rng.choice([-1, 1], size=768)

    # 注册
    stored_template, se = sstm.enroll(re)
    print(f"\n[Enroll] Key length: {len(se)} bytes ({len(se)*8} bits)")
    print(f"[Enroll] Stored hash: {stored_template[:20]}...")

    # 认证（genuine，不同翻转量）
    # Note: RS corrects symbol errors; contiguous bit flips map to fewer symbol errors
    print("\nGenuine authentication tests (contiguous bit flips):")
    t_bits = sstm.get_error_correction_capacity() * 8
    for n_flip in [50, 200, 328, 329]:
        flip_mask = np.zeros(768, dtype=bool)
        flip_mask[:n_flip] = True
        rp = re.copy()
        rp[flip_mask] = -rp[flip_mask]
        is_genuine, _ = sstm.authenticate(rp, stored_template)
        status = "PASS ✓" if is_genuine else "FAIL ✗"
        within = "within" if n_flip <= t_bits else "exceeds"
        print(f"  Flip {n_flip:3d} bits: {status} [{within} t={t_bits} bits]")

    # 认证（impostor）
    rp_imp = rng.choice([-1, 1], size=768)
    is_imp, _ = sstm.authenticate(rp_imp, stored_template)
    print(f"\n[Impostor] Result: {'PASS (false accept!)' if is_imp else 'FAIL ✓ (correct)'}")


if __name__ == "__main__":
    demo_sstm()
