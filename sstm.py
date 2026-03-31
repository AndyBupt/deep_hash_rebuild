"""
SSTM: Secure Sketch Template Module（安全草图模板模块）
对应论文 Section III-C 和 Section IV-E

流程:
  注册: re (G bits) → RS解码 → 安全草图 se (k bits) → SHA256 → 存储 hash(se)
  认证: rp (G bits) → RS解码 → 探针草图 sp → SHA256 → 比对 hash(sp) == hash(se)

论文 RS 码参数 (Section IV-E):
  - GF(2^8)，m=8，每符号8 bits（1字节）
  - 最大码字长度 N'=255 符号
  - 使用缩短 RS 码: 实际码字长度 N 符号 (N < N')
  - 信息符号数 K，安全性 k = 8*K bits
  - 纠错能力 t = (N-K)/2 符号 = (n-k)/2 bits (n=8N, k=8K)
  - 可撤销模板长度 G = n bits = N 字节
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
    安全草图模板模块（使用真正的 Reed-Solomon 码）

    对应论文 Section III-C / IV-E:
    "Due to its maximum distance separable (MDS) property, we have selected
    Reed-Solomon (RS) codes and used RS decoder for FEC decoding in SSTM."
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
            纠错能力 t = nsym // 2 符号 = G//2 - K*4 bits
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
        注册阶段: 生成安全模板

        Args:
            re: (G,) 可撤销模板，值为 {-1,+1} 或 {0,1}

        Returns:
            secure_template: SHA256 哈希字符串（存入数据库）
            se_bytes: K字节安全草图（仅用于分析，实际不存储）
        """
        re_bytes = self._bits_to_bytes(re)  # G bits → N 字节（含噪码字）
        assert len(re_bytes) == self.N, \
            f"模板字节数 {len(re_bytes)} 与 N={self.N} 不匹配"

        # FEC 解码: 将可撤销模板视为含噪 RS 码字，解码得到最近码字的信息部分
        # 论文: "This noisy codeword is decoded with a FEC decoder and the output
        #        of the decoder is the multimodal secure sketch se"
        try:
            decoded, _, _ = self.rsc.decode(re_bytes)
            se_bytes = bytes(decoded)
        except ReedSolomonError:
            # 错误超过纠错能力，解码失败
            se_bytes = b'\x00' * self.K

        # 加密哈希: SHA256(se) 存入数据库
        secure_template = hashlib.sha256(se_bytes).hexdigest()
        return secure_template, se_bytes

    def authenticate(self, rp: np.ndarray,
                     stored_hash: str) -> Tuple[bool, bytes]:
        """
        认证阶段: 验证探针是否匹配

        Args:
            rp: (G,) 探针可撤销模板
            stored_hash: 注册时存储的 SHA256 哈希

        Returns:
            is_genuine: 是否认证通过
            sp_bytes: 探针安全草图
        """
        rp_bytes = self._bits_to_bytes(rp)
        assert len(rp_bytes) == self.N

        try:
            decoded, _, _ = self.rsc.decode(rp_bytes)
            sp_bytes = bytes(decoded)
        except ReedSolomonError:
            sp_bytes = b'\x00' * self.K

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
    print("SSTM 安全草图模板模块演示（真正 Reed-Solomon 码）")
    print("=" * 55)

    # 论文参数: G=768, K=13 → k=104 bits, 纠错能力 t=41 符号
    sstm = SSTM(G=768, K=13)
    print(f"码字长度:   N={sstm.N} 符号 ({sstm.G} bits)")
    print(f"信息长度:   K={sstm.K} 符号 ({sstm.k_bits} bits) ← 安全性")
    print(f"纠错能力:   t={sstm.get_error_correction_capacity()} 符号")

    rng = np.random.default_rng(42)
    re = rng.choice([-1, 1], size=768)

    # 注册
    stored_hash, se = sstm.enroll(re)
    print(f"\n[注册] 安全草图: {len(se)} 字节 ({len(se)*8} bits)")
    print(f"[注册] 存储哈希: {stored_hash[:20]}...")

    # 认证（genuine，少量噪声）
    for flip_rate in [0.05, 0.20, 0.40]:
        noise_mask = rng.random(768) < flip_rate
        rp = re.copy()
        rp[noise_mask] = -rp[noise_mask]
        flip_count = noise_mask.sum()
        is_genuine, _ = sstm.authenticate(rp, stored_hash)
        status = "通过 ✓" if is_genuine else "拒绝 ✗"
        print(f"[Genuine] 翻转 {flip_count:3d} bits ({flip_rate*100:.0f}%): {status}")

    # 认证（impostor）
    rp_imp = rng.choice([-1, 1], size=768)
    is_imp, _ = sstm.authenticate(rp_imp, stored_hash)
    print(f"\n[Impostor] 认证结果: {'通过（误报！）' if is_imp else '拒绝 ✓'}")


if __name__ == "__main__":
    demo_sstm()
