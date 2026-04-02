"""
CTM: Cancelable Template Module（可撤销模板模块）
对应论文 Section III-B3 和 Section IV

核心逻辑:
  1. 输入: DFB输出的1024维二值向量 e ∈ {-1, +1}^J
  2. 随机比特选择: 用用户密钥 key 从 J 维中选 G 位
  3. 可靠性排序: 按比特可靠性降序排列选出的比特
  4. 输出: G维可撤销模板 re，密钥 ke（选出的比特索引）

可撤销性:
  - 若密钥泄露，重新选一组随机索引即可生成新模板
  - 不同用户使用不同密钥，模板不可链接
"""

import numpy as np
import torch
from typing import Tuple, Optional


class CTM:
    """
    可撤销模板模块

    对应论文 Section III-B3:
    "A user-specific random-bit selection is performed using the fused
    feature vector to generate the cancelable multimodal template."
    """

    def __init__(self, hash_dim: int = 1024, G: int = 512,
                 reliability: Optional[np.ndarray] = None):
        """
        Args:
            hash_dim: DFB输出的二值向量维度 J（论文中为1024）
            G: 随机选取的比特数，论文实验了 128/256/512/768
            reliability: (J,) 每个比特位置的可靠性分数，用于排序
                         None时跳过排序（注册和认证顺序一致即可）
        """
        self.hash_dim = hash_dim
        self.G = G
        # reliability[i] = (1 - p_genuine_error_i) * p_impostor_error_i
        # 论文 Section III-B3
        self.reliability = reliability

    def enroll(self, binary_vec: np.ndarray,
               key: Optional[np.ndarray] = None,
               seed: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        注册阶段: 生成可撤销模板和用户密钥

        Args:
            binary_vec: (J,) 二值向量，值为 {-1, +1} 或 {0, 1}
            key: 可选，预先指定的密钥（比特索引数组，长度G）
                 若为None则随机生成
            seed: 随机种子（用于复现）

        Returns:
            re: (G,) 可撤销模板（选出的G个比特，按可靠性排序）
            ke: (G,) 用户密钥（选出的比特索引）
        """
        binary_vec = self._to_numpy(binary_vec)
        assert len(binary_vec) == self.hash_dim, \
            f"输入向量维度 {len(binary_vec)} 与 hash_dim {self.hash_dim} 不匹配"

        # 生成或使用密钥（随机比特索引）
        if key is None:
            rng = np.random.default_rng(seed)
            ke = rng.choice(self.hash_dim, size=self.G, replace=False)
            ke = np.sort(ke)  # 排序便于复现
        else:
            ke = np.asarray(key, dtype=np.int64)
            assert len(ke) == self.G, f"密钥长度 {len(ke)} 与 G={self.G} 不匹配"

        # 选出 G 个比特
        selected = binary_vec[ke]  # (G,)

        # 按可靠性降序排列（论文 Section III-B3）
        # reliability_i = (1 - p_genuine_error_i) * p_impostor_error_i
        if self.reliability is not None:
            rel_selected = self.reliability[ke]           # 选出位置的可靠性
            sort_order = np.argsort(-rel_selected)        # 降序
            ke = ke[sort_order]
            selected = selected[sort_order]

        re = selected
        return re, ke

    def authenticate(self, binary_vec: np.ndarray,
                     ke: np.ndarray) -> np.ndarray:
        """
        认证阶段: 用已有密钥生成探针模板

        Args:
            binary_vec: (J,) 探针的二值向量
            ke: (G,) 注册时的用户密钥

        Returns:
            rp: (G,) 探针可撤销模板
        """
        binary_vec = self._to_numpy(binary_vec)
        rp = binary_vec[ke]
        return rp

    def revoke_and_reenroll(self, binary_vec: np.ndarray,
                            old_key: np.ndarray,
                            seed: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        撤销旧模板，用新密钥重新注册（可撤销性的核心体现）

        Args:
            binary_vec: (J,) 原始二值向量（从安全位置取回，或重新采集）
            old_key: (G,) 被撤销的旧密钥
            seed: 新密钥的随机种子

        Returns:
            re_new: (G,) 新可撤销模板
            ke_new: (G,) 新密钥（与旧密钥不同）
        """
        # 生成新密钥，确保与旧密钥不同
        rng = np.random.default_rng(seed)
        max_attempts = 100
        for _ in range(max_attempts):
            ke_new = rng.choice(self.hash_dim, size=self.G, replace=False)
            ke_new = np.sort(ke_new)
            if not np.array_equal(ke_new, old_key):
                break

        re_new, ke_new = self.enroll(binary_vec, key=ke_new)
        return re_new, ke_new

    @staticmethod
    def compute_bit_reliability(genuine_vecs: np.ndarray,
                                impostor_vecs: np.ndarray
                                ) -> np.ndarray:
        """
        计算每个比特位置的可靠性
        论文公式 (Section III-B3):
            reliability_i = (1 - p_genuine_error_i) * p_impostor_error_i

        其中:
          p_genuine_error_i: 同一用户不同采集中，第i位不一致的概率
          p_impostor_error_i: 不同用户中，第i位恰好相同的概率（即1-p_impostor_diff）

        Args:
            genuine_vecs:  (N_g, J) 同一用户多次采集的二值向量（训练集）
            impostor_vecs: (N_i, J) 不同用户的二值向量（训练集）

        Returns:
            reliability: (J,) 每个比特位置的可靠性，值越大越可靠
        """
        # 转换为 {0, 1}
        g = (genuine_vecs > 0).astype(float)
        imp = (impostor_vecs > 0).astype(float)

        # p_genuine_error_i: 同一用户内，第i位与众数不同的概率
        # 用每个位置的方差估计：p*(1-p)，p为该位为1的概率
        p_g = g.mean(axis=0)
        p_genuine_error = p_g * (1 - p_g) * 4  # 归一化到[0,1]

        # p_impostor_error_i: 冒充者与真实用户在第i位恰好相同的概率
        # 近似为 impostor 中该位为1的概率 * genuine中为1的概率
        #        + impostor中为0的概率 * genuine中为0的概率
        p_i = imp.mean(axis=0)
        p_impostor_same = p_g * p_i + (1 - p_g) * (1 - p_i)
        p_impostor_error = 1.0 - p_impostor_same  # 不同的概率（越大越好区分）

        reliability = (1.0 - p_genuine_error) * p_impostor_error
        return reliability

    @staticmethod
    def estimate_reliability_from_codes(all_codes: np.ndarray,
                                        all_labels: np.ndarray
                                        ) -> np.ndarray:
        """
        从训练集提取的二值码统计每个位置的可靠性

        Args:
            all_codes:  (N, J) 所有训练样本的二值向量
            all_labels: (N,) 对应的身份标签

        Returns:
            reliability: (J,) 可靠性分数
        """
        genuine_list = []
        impostor_list = []
        unique_ids = np.unique(all_labels)

        for uid in unique_ids:
            idx = np.where(all_labels == uid)[0]
            if len(idx) >= 2:
                genuine_list.append(all_codes[idx])
            else:
                impostor_list.append(all_codes[idx])

        # 其他身份的样本作为 impostor
        other_ids = [uid for uid in unique_ids
                     if np.sum(all_labels == uid) < 2]
        impostor_idx = np.concatenate(
            [np.where(all_labels == uid)[0] for uid in other_ids]
        ) if other_ids else np.arange(len(all_labels))

        genuine_vecs = np.vstack(genuine_list) if genuine_list else all_codes
        impostor_vecs = all_codes[impostor_idx]

        return CTM.compute_bit_reliability(genuine_vecs, impostor_vecs)

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """计算两个二值向量的汉明距离"""
        a = self._to_binary_01(a)
        b = self._to_binary_01(b)
        return int(np.sum(a != b))

    def hamming_distance_ratio(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算归一化汉明距离（0~1）"""
        return self.hamming_distance(a, b) / len(a)

    @staticmethod
    def _to_numpy(vec) -> np.ndarray:
        if isinstance(vec, torch.Tensor):
            return vec.detach().cpu().numpy()
        return np.asarray(vec)

    @staticmethod
    def _to_binary_01(vec: np.ndarray) -> np.ndarray:
        """统一转换为 {0, 1}"""
        return (vec > 0).astype(np.uint8)


class StableCTM(CTM):
    """
    改进版可撤销模板模块：基于稳定比特选择

    与原始 CTM 的区别:
      - 原始 CTM: 随机选 G 个 bit（每个用户密钥不同）
      - StableCTM: 优先选翻转率最低的 G 个 bit（全局稳定 bit 池）
                   再从稳定 bit 池中随机选，保持可撤销性

    改进效果:
      - Genuine 翻转率从 ~20% 降低到 ~5-8%
      - G-S 曲线的下降拐点右移，在更高安全性下仍保持高 GAR
    """

    def __init__(self, hash_dim: int = 1024, G: int = 512,
                 flip_rate: Optional[np.ndarray] = None,
                 stable_ratio: float = 0.3):
        """
        Args:
            hash_dim: 二值向量维度
            G: 选取的比特数
            flip_rate: (J,) 每个 bit 位置的翻转率（由训练集统计得到）
                       None 时退化为随机选择（等同于原始 CTM）
            stable_ratio: 稳定 bit 池占总 bit 数的比例
                          默认 0.3 表示从翻转率最低的 30%（约300个）bit 中选
        """
        super().__init__(hash_dim=hash_dim, G=G)
        self.flip_rate = flip_rate
        self.stable_ratio = stable_ratio

        # 构建稳定 bit 池：翻转率最低的前 stable_ratio 比例的 bit
        if flip_rate is not None:
            n_stable = max(G, int(hash_dim * stable_ratio))
            self.stable_pool = np.argsort(flip_rate)[:n_stable]
        else:
            self.stable_pool = np.arange(hash_dim)

    def enroll(self, binary_vec: np.ndarray,
               key: Optional[np.ndarray] = None,
               seed: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        注册阶段: 从稳定 bit 池中随机选 G 个 bit

        改进点: 不从全部 1024 个 bit 中随机选，
                而是从翻转率最低的 stable_pool 中随机选，
                保证选出的 bit 都是相对稳定的
        """
        binary_vec = self._to_numpy(binary_vec)
        assert len(binary_vec) == self.hash_dim

        if key is None:
            # 从稳定 bit 池中随机选 G 个
            rng = np.random.default_rng(seed)
            ke = rng.choice(self.stable_pool, size=self.G, replace=False)
            ke = np.sort(ke)
        else:
            ke = np.asarray(key, dtype=np.int64)
            assert len(ke) == self.G

        selected = binary_vec[ke]
        re = selected
        return re, ke

    def revoke_and_reenroll(self, binary_vec: np.ndarray,
                            old_key: np.ndarray,
                            seed: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """撤销旧模板，从稳定 bit 池中重新选一组 bit"""
        rng = np.random.default_rng(seed)
        max_attempts = 100
        for _ in range(max_attempts):
            ke_new = rng.choice(self.stable_pool, size=self.G, replace=False)
            ke_new = np.sort(ke_new)
            if not np.array_equal(ke_new, old_key):
                break

        re_new, ke_new = self.enroll(binary_vec, key=ke_new)
        return re_new, ke_new

    @staticmethod
    def compute_flip_rate(all_codes: np.ndarray,
                          all_labels: np.ndarray) -> np.ndarray:
        """
        从训练集统计每个 bit 位置的翻转率

        Args:
            all_codes:  (N, J) 所有训练样本的二值向量
            all_labels: (N,) 对应的身份标签

        Returns:
            flip_rate: (J,) 每个 bit 位置的翻转率，值越小越稳定
        """
        unique_ids = np.unique(all_labels)
        hash_dim = all_codes.shape[1]
        flip_counts = np.zeros(hash_dim)
        total_counts = np.zeros(hash_dim)

        for uid in unique_ids:
            idx = np.where(all_labels == uid)[0]
            if len(idx) < 2:
                continue
            user_codes = (all_codes[idx] > 0).astype(float)
            majority = (user_codes.mean(axis=0) >= 0.5).astype(float)
            for code in user_codes:
                flip_counts += (code != majority)
                total_counts += 1

        flip_rate = flip_counts / np.maximum(total_counts, 1)
        return flip_rate


def demo_ctm():
    """演示 CTM 的注册、认证、撤销流程"""
    print("=" * 50)
    print("CTM 可撤销模板模块演示")
    print("=" * 50)

    ctm = CTM(hash_dim=1024, G=512)
    rng = np.random.default_rng(42)

    # 模拟同一用户的两次指纹采集（有少量噪声）
    true_vec = rng.choice([-1, 1], size=1024)
    noise = rng.choice([-1, 1], size=1024)
    noise_mask = rng.random(1024) < 0.05  # 5% 比特翻转
    probe_vec = true_vec.copy()
    probe_vec[noise_mask] = noise[noise_mask]

    # 模拟冒充者的指纹
    impostor_vec = rng.choice([-1, 1], size=1024)

    # --- 注册 ---
    re, ke = ctm.enroll(true_vec, seed=0)
    print(f"\n[注册] 模板长度: {len(re)}, 密钥长度: {len(ke)}")

    # --- 认证（genuine）---
    rp_genuine = ctm.authenticate(probe_vec, ke)
    dist_genuine = ctm.hamming_distance(re, rp_genuine)
    print(f"[Genuine] 汉明距离: {dist_genuine}/{ctm.G} "
          f"({dist_genuine/ctm.G*100:.1f}%)")

    # --- 认证（impostor）---
    rp_impostor = ctm.authenticate(impostor_vec, ke)
    dist_impostor = ctm.hamming_distance(re, rp_impostor)
    print(f"[Impostor] 汉明距离: {dist_impostor}/{ctm.G} "
          f"({dist_impostor/ctm.G*100:.1f}%)")

    # --- 撤销并重新注册 ---
    re_new, ke_new = ctm.revoke_and_reenroll(true_vec, ke, seed=99)
    print(f"\n[撤销重注册] 新模板与旧模板相关性检查:")
    overlap = len(set(ke) & set(ke_new))
    print(f"  新旧密钥重叠位数: {overlap}/{ctm.G} "
          f"(越少越好，期望约 {ctm.G**2//ctm.hash_dim})")

    # 验证新旧模板不可链接（应接近随机）
    dist_link = ctm.hamming_distance(re, re_new)
    print(f"  新旧模板汉明距离: {dist_link}/{ctm.G} "
          f"(应接近 {ctm.G//2}，即不可链接)")


if __name__ == "__main__":
    demo_ctm()
