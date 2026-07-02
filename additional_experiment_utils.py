"""
additional_experiment_utils.py — Shared helpers for supplementary experiments

These helpers keep the new supplementary scripts close to the style of the
existing evaluation files while avoiding repeated boilerplate.
"""

import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_curve

from dataset import build_dataloaders
from model import FingerprintHashNet


DEFAULT_MODEL_PATH = os.environ.get("RGSS_MODEL_PATH", "checkpoints/final_model.pth")
DEFAULT_DATA_ROOT = os.environ.get("FVC2004_ROOT", "/root/autodl-tmp/FVC2004")
DEFAULT_DB_NAMES = [
    "DB1_A/image", "DB1_B/image",
    "DB2_A/image", "DB2_B/image",
    "DB3_A/image", "DB3_B/image",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    db_names=None,
    model_path: str = DEFAULT_MODEL_PATH,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    beta: float = 32,
):
    """Load train/test dataloaders and the trained FingerprintHashNet."""
    if db_names is None:
        db_names = DEFAULT_DB_NAMES

    device = load_device()
    train_loader, test_loader, num_classes = build_dataloaders(
        data_root, db_names, train_ratio=train_ratio, batch_size=batch_size
    )

    model = FingerprintHashNet(num_classes=num_classes, hash_dim=1024, pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    else:
        print(f"WARNING: model not found, using random weights: {model_path}")
    model = model.to(device)
    model.set_beta(beta)
    return device, model, train_loader, test_loader, num_classes


def extract_codes_with_embed(model, loader, device):
    model.eval()
    all_binary, all_hash, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, hash_c, binary_c = model(imgs.to(device))
            all_binary.append(binary_c.cpu().numpy())
            all_hash.append(hash_c.cpu().numpy())
            all_labels.append(lbs.numpy())
    return np.vstack(all_binary), np.vstack(all_hash), np.concatenate(all_labels)


def extract_codes(model, loader, device):
    model.eval()
    all_binary, all_labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            _, _, binary_c = model(imgs.to(device))
            all_binary.append(binary_c.cpu().numpy())
            all_labels.append(lbs.numpy())
    return np.vstack(all_binary), np.concatenate(all_labels)


def hamming_ratio(a: np.ndarray, b: np.ndarray) -> float:
    a = (np.asarray(a) > 0).astype(np.uint8)
    b = (np.asarray(b) > 0).astype(np.uint8)
    return float(np.mean(a != b))


def bits01(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    if vec.min() < 0:
        return ((vec + 1) / 2).astype(np.uint8)
    return vec.astype(np.uint8)


def signs_pm1(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    return bits.astype(np.int8) * 2 - 1


def parse_rgss_template(stored_template: str) -> Dict[str, np.ndarray]:
    parts = stored_template.split('|')
    secure_hash = parts[0]
    h_bytes = bytes.fromhex(parts[1])
    ecc_bytes = bytes.fromhex(parts[2])
    perm = np.array(json.loads(parts[3]), dtype=np.int64)
    h_bits = np.unpackbits(np.frombuffer(h_bytes, dtype=np.uint8))
    return {
        "hash": secure_hash,
        "h_bytes": h_bytes,
        "h_bits": h_bits,
        "ecc_bytes": ecc_bytes,
        "perm": perm,
    }


def compute_eer_from_scores(mated_scores: np.ndarray, non_mated_scores: np.ndarray,
                            higher_score_more_mated: bool = True) -> float:
    mated_scores = np.asarray(mated_scores, dtype=np.float64)
    non_mated_scores = np.asarray(non_mated_scores, dtype=np.float64)
    scores = np.concatenate([mated_scores, non_mated_scores])
    if not higher_score_more_mated:
        scores = -scores
    labels = np.concatenate([
        np.ones(len(mated_scores), dtype=np.uint8),
        np.zeros(len(non_mated_scores), dtype=np.uint8),
    ])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def find_bch_params_for_k(k_bits: int, m: int = 9, max_t: int = 60) -> Tuple[int, int]:
    import bchlib

    for t in range(1, max_t + 1):
        try:
            b = bchlib.BCH(t=t, m=m)
            k_b = (b.n - b.ecc_bits) // 8 * 8
            if k_b == k_bits:
                return m, b.t
        except Exception:
            break
    raise RuntimeError(f"Cannot find BCH(m={m}, t) for k_bits={k_bits}")


def summarise_k50(k_bits_list, gars, threshold: float = 50.0):
    valid = [k for k, g in zip(k_bits_list, gars) if g >= threshold]
    return valid[-1] if valid else None
