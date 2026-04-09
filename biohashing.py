"""
BioHashing: 经典可撤销生物特征保护方法

参考论文:
  Teoh et al., "BioHashing: two factor authentication featuring fingerprint
  data and tokenised random number", Pattern Recognition, 2004.

原理：
  用用户专属的随机矩阵（由 token/密钥生成）对特征向量做随机投影，
  再二值化得到 BioHash token。

  注册: x (hash_dim) → R (hash_dim × G, 由 key 生成) → x·R → sign → token (G bits)
  认证: x_probe → 相同的 R → sign → token_probe → 汉明距离比对

与 CTM 的接口保持一致：
  enroll(binary_vec, key=None, seed=None) → (token, key)
  authenticate(binary_vec, key)           → token_probe
  hamming_distance(a, b)                  → int
"""

import numpy as np
from typing import Tuple, Optional


class BioHashing:
    """
    BioHashing 可撤销模板方法

    接口与 CTM/StableCTM 完全一致，可直接替换进评估流程。
    """

    def __init__(self, hash_dim: int = 1024, G: int = 512):
        """
        Args:
            hash_dim: 输入特征维度（VGG-19 哈希层输出，1024）
            G:        输出 BioHash token 长度（bits）
        """
        self.hash_dim = hash_dim
        self.G = G

    def enroll(self, binary_vec: np.ndarray,
               key: Optional[int] = None,
               seed: Optional[int] = None
               ) -> Tuple[np.ndarray, int]:
        """
        注册阶段：生成 BioHash token 和用户密钥

        Args:
            binary_vec: (hash_dim,) 特征向量，值为 {-1,+1} 或 {0,1}
            key:  用户密钥（整数随机种子），None 则随机生成
            seed: 备用随机种子（与 CTM 接口兼容）

        Returns:
            token: (G,) BioHash token，值为 {-1,+1}
            key:   用户密钥（整数）
        """
        x = self._to_float(binary_vec)

        # 生成密钥
        if key is None:
            rng_seed = seed if seed is not None else np.random.randint(0, 2**31)
            key = int(rng_seed)

        # 用密钥生成随机投影矩阵 R (hash_dim × G)，列正交化
        R = self._make_projection(key)

        # 随机投影 + 二值化
        projected = x @ R          # (G,)
        token = np.sign(projected)  # {-1, 0, +1}
        token[token == 0] = 1       # 处理零值

        return token, key

    def authenticate(self, binary_vec: np.ndarray,
                     key: int) -> np.ndarray:
        """
        认证阶段：用注册时的密钥生成探针 token

        Args:
            binary_vec: (hash_dim,) 探针特征向量
            key:        注册时的用户密钥

        Returns:
            token_probe: (G,) 探针 BioHash token
        """
        x = self._to_float(binary_vec)
        R = self._make_projection(key)
        projected = x @ R
        token = np.sign(projected)
        token[token == 0] = 1
        return token

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """计算两个 {-1,+1} token 的汉明距离"""
        return int(np.sum(a != b))

    def _make_projection(self, key: int) -> np.ndarray:
        """
        用密钥生成随机正交投影矩阵 R (hash_dim × G)

        使用 Gram-Schmidt 正交化确保列向量正交，
        提升 BioHashing 的区分能力（原论文要求）。
        """
        rng = np.random.default_rng(int(key))
        # 生成随机矩阵
        R = rng.standard_normal((self.hash_dim, self.G))
        # 列正交化（QR 分解）
        if self.G <= self.hash_dim:
            Q, _ = np.linalg.qr(R)
            R = Q[:, :self.G]
        else:
            # G > hash_dim 时无法完全正交，退而求其次做列归一化
            R = R / np.linalg.norm(R, axis=0, keepdims=True)
        return R

    def _to_float(self, vec: np.ndarray) -> np.ndarray:
        """将输入统一转为 float，{0,1} 转为 {-1,+1}"""
        vec = np.asarray(vec, dtype=np.float32)
        if vec.min() >= 0:
            vec = 2 * vec - 1   # {0,1} → {-1,+1}
        return vec


def demo_biohashing():
    """演示 BioHashing 的注册和认证流程"""
    print("=" * 50)
    print("BioHashing Demo")
    print("=" * 50)

    bh = BioHashing(hash_dim=1024, G=512)
    rng = np.random.default_rng(42)

    # 模拟注册特征
    x_enroll = rng.choice([-1.0, 1.0], size=1024)
    token_e, key = bh.enroll(x_enroll)
    print(f"Token length: {len(token_e)} bits")
    print(f"Key (seed): {key}")

    # 同一用户认证（少量噪声）
    print("\nGenuine authentication:")
    for noise in [0.0, 0.05, 0.10, 0.20]:
        x_probe = x_enroll.copy()
        flip_mask = rng.random(1024) < noise
        x_probe[flip_mask] = -x_probe[flip_mask]
        token_p = bh.authenticate(x_probe, key)
        dist = bh.hamming_distance(token_e, token_p)
        print(f"  noise={noise:.0%}: hamming={dist}/{bh.G} "
              f"({dist/bh.G*100:.1f}%)")

    # 冒充者（不同用户，相同 key）
    x_imp = rng.choice([-1.0, 1.0], size=1024)
    token_imp = bh.authenticate(x_imp, key)
    dist_imp = bh.hamming_distance(token_e, token_imp)
    print(f"\nImpostor (stolen key): hamming={dist_imp}/{bh.G} "
          f"({dist_imp/bh.G*100:.1f}%)")

    # 冒充者（不同 key）
    _, key2 = bh.enroll(x_imp)
    token_imp2 = bh.authenticate(x_imp, key2)
    dist_imp2 = bh.hamming_distance(token_e, token_imp2)
    print(f"Impostor (unknown key): hamming={dist_imp2}/{bh.G} "
          f"({dist_imp2/bh.G*100:.1f}%)")


if __name__ == "__main__":
    demo_biohashing()
