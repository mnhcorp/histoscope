import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
try:
    from scipy.spatial.distance import pdist, squareform
except ImportError:  # pragma: no cover - scipy optional in some environments
    pdist = squareform = None
from spider_embedding_dataset import SpiderEmbeddingDataset
from torch.utils.data import DataLoader
from ignite_embedding_dataset import IgniteEmbeddingDataset
from kather100k_embedding_dataset import Kather100kEmbeddingDataset
from simple_sae_spider import SparseAutoEncoder
from tqdm import tqdm

# report_file can be set by calling module
report_file = None

def load_label_map(cache_root="cache"):
    """Load label map from cache directory, with fallback to SPIDER labels"""
    label_map_path = os.path.join(cache_root, "label_map.json")

    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map_data = json.load(f)
        # Convert string keys to integers and create the mapping
        label2desc = {int(k): v for k, v in label_map_data.items()}
        print(f"Loaded label map from {label_map_path}")
        return label2desc
    else:
        # Fallbacks for known caches. Prefer a static mapping for Kather (NCT-CRC-HE-100K)
        print(f"Label map not found at {label_map_path}, using fallback mapping based on cache root '{cache_root}'")
        if "nctcrche100k" in cache_root or cache_root.endswith("cache-nctcrche100k"):
            # Kather / NCT-CRC-HE-100K static mapping (9 classes)
            return {
                0: "ADI - Adipose",
                1: "BACK - Background",
                2: "DEB - Debris",
                3: "LYM - Lymphocytes",
                4: "MUC - Mucus",
                5: "MUS - Smooth muscle",
                6: "NORM - Normal colon mucosa",
                7: "STR - Cancer-associated stroma",
                8: "TUM - Colorectal adenocarcinoma epithelium"
            }
        else:
            # Generic fallback: SPIDER-like mapping (13 classes) if nothing else applicable
            return {
                0: 'Adenocarcinoma HG', 1: 'Adenocarcinoma LG', 2: 'Adenoma HG',
                3: 'Adenoma LG', 4: 'Fat', 5: 'Hyperplastic polyp',
                6: 'Inflammation', 7: 'Mucus', 8: 'Muscle', 9: 'Necrosis',
                10: 'Sessile serrated lesion', 11: 'Stroma healthy', 12: 'Vessels'
            }

def extract_baseline_features(dataloader, device="cuda"):
    """Extract original UNI embeddings (1024-dim) without SAE processing"""
    feature_matrix = []
    image_paths = []
    labels = []

    # No model needed - just extract the embeddings directly from dataloader
    for emb, lab, path in dataloader:
        # emb is already the UNI embedding (1024-dim)
        feature_matrix.append(emb.cpu())
        image_paths.extend(path)
        labels.extend(lab)

    feature_matrix = torch.cat(feature_matrix, dim=0)
    return feature_matrix, image_paths, labels

def calculate_baseline_monosemanticity_scores(feature_matrix_train, labels_train,
                                            feature_matrix_test, labels_test,
                                            feature_indices, eps=1e-6, delta=1e-6, robust_trim=0.05, cache_root="cache"):
    """Calculate monosemanticity scores for baseline UNI features"""
    label2desc = load_label_map(cache_root)

    labels_train = torch.tensor(labels_train) if not isinstance(labels_train, torch.Tensor) else labels_train
    labels_test = torch.tensor(labels_test) if not isinstance(labels_test, torch.Tensor) else labels_test

    mono_scores = {}

    print(f"Computing monosemanticity for {len(feature_indices)} baseline features...")
    for i, feature_idx in enumerate(tqdm(feature_indices, desc="mono-baseline")):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(feature_indices)}")

        # Extract feature activations (can be negative for UNI embeddings)
        train_activations = feature_matrix_train[:, feature_idx]
        test_activations = feature_matrix_test[:, feature_idx]

        # For UNI embeddings, we use absolute values for class means since they can be negative
        # This measures "feature importance" rather than "activation"
        train_abs = torch.abs(train_activations)
        test_abs = torch.abs(test_activations)

        # --- robust class means (train) ---
        mu_train = torch.zeros(13)
        for class_idx in range(13):
            class_mask = (labels_train == class_idx)
            if class_mask.any():
                class_vals = train_abs[class_mask]
                # Use all values since UNI embeddings don't have sparsity like SAE
                if class_vals.numel() > 0:
                    mu_train[class_idx] = robust_mean(class_vals.detach().cpu().numpy(), trim=robust_trim)

        # --- robust class means (test) ---
        mu_test = torch.zeros(13)
        for class_idx in range(13):
            class_mask = (labels_test == class_idx)
            if class_mask.any():
                class_vals = test_abs[class_mask]
                if class_vals.numel() > 0:
                    mu_test[class_idx] = robust_mean(class_vals.detach().cpu().numpy(), trim=robust_trim)

        # Normalize to get probability distributions (original method)
        sum_train = mu_train.sum() + delta
        sum_test = mu_test.sum() + delta
        p_train = mu_train / sum_train
        p_test = mu_test / sum_test

        # Softmax-based probability distributions (new method)
        p_train_softmax = torch.softmax(mu_train, dim=0)
        p_test_softmax = torch.softmax(mu_test, dim=0)

        # Lock dominant class on train (using original normalization)
        c_star_train = torch.argmax(p_train).item()
        c_star_test = torch.argmax(p_test).item()
        flipped = c_star_train != c_star_test

        # Lock dominant class on train (using softmax)
        c_star_train_softmax = torch.argmax(p_train_softmax).item()
        c_star_test_softmax = torch.argmax(p_test_softmax).item()
        flipped_softmax = c_star_train_softmax != c_star_test_softmax

        # Calculate margins (train-locked)
        def second_best(p, k):
            """Get second highest probability excluding class k"""
            p_copy = p.clone()
            p_copy[k] = -float('inf')  # Exclude the dominant class
            if torch.all(p_copy == -float('inf')):
                return 0.0  # Only one class has non-zero probability
            return torch.max(p_copy).item()

        # Handle edge cases (original method)
        if sum_train <= delta:  # No active samples in train
            m_train = 0.0
        else:
            m_train = p_train[c_star_train].item() - second_best(p_train, c_star_train)

        if sum_test <= delta:  # No active samples in test
            m_test = 0.0
        else:
            m_test = p_test[c_star_train].item() - second_best(p_test, c_star_train)

        # Softmax-based margins
        m_train_softmax = p_train_softmax[c_star_train_softmax].item() - second_best(p_train_softmax, c_star_train_softmax)
        m_test_softmax = p_test_softmax[c_star_train_softmax].item() - second_best(p_test_softmax, c_star_train_softmax)

        # Final monosemanticity scores (clamp negatives to 0)
        M = max(0.0, min(m_train, m_test))
        M_softmax = max(0.0, min(m_train_softmax, m_test_softmax))

        mono_scores[feature_idx] = {
            'M': M,  # Final monosemanticity score [0,1] (normalized)
            'M_softmax': M_softmax,  # Final monosemanticity score [0,1] (softmax)
            'm_train': m_train,  # Raw train margin [-1,1] (normalized)
            'm_test': m_test,   # Raw test margin [-1,1] (normalized)
            'm_train_softmax': m_train_softmax,  # Raw train margin [-1,1] (softmax)
            'm_test_softmax': m_test_softmax,   # Raw test margin [-1,1] (softmax)
            'c_star_train': c_star_train,
            'c_star_test': c_star_test,
            'c_star_train_softmax': c_star_train_softmax,
            'c_star_test_softmax': c_star_test_softmax,
            'c_star_train_name': label2desc[c_star_train],
            'c_star_test_name': label2desc[c_star_test],
            'c_star_train_name_softmax': label2desc[c_star_train_softmax],
            'c_star_test_name_softmax': label2desc[c_star_test_softmax],
            'flipped': flipped,
            'flipped_softmax': flipped_softmax,
            'sum_train_activations': sum_train - delta,  # For diagnostics
            'sum_test_activations': sum_test - delta
        }

    return mono_scores

def compute_ms_scores(feature_matrix: torch.Tensor,
                      embedding_matrix: torch.Tensor,
                      feature_indices: list,
                      verify_sample: int = 0) -> dict:
    """Efficient computation of Monosemanticity Score (MS_score) for features.

    Original definition (naive O(F*N^2)):
        MS^k = 1 / (N (N-1)) * sum_{n!=m} (ã_n^k ã_m^k) * s_{nm}
        where ã are min-max normalized activations for feature k, s_{nm} cosine similarity.

    Optimization:
        Let w_n = ã_n^k and e_n unit-normalized embedding rows.
        Σ_{n!=m} w_n w_m (e_n·e_m) = (Σ_n w_n e_n)^2_{L2} - Σ_n w_n^2 (since ||e_n||=1).
        So:
            MS^k = ( || Σ_n w_n e_n ||^2 - Σ_n w_n^2 ) / ( N (N-1) )

    Complexity becomes O(F * N * D) instead of O(F * N^2 + N^2 * D).

    Args:
        feature_matrix: (N, F) activations (can be signed).
        embedding_matrix: (N, D) semantic embeddings (will be L2-normalized per row).
        feature_indices: list of feature indices to score.
        verify_sample: if >0, runs naive formula for that many first features to assert closeness.

    Returns:
        dict feature_index -> MS_score in [-1,1].
    """
    if not isinstance(feature_matrix, torch.Tensor):
        raise ValueError("feature_matrix must be a torch.Tensor")
    if not isinstance(embedding_matrix, torch.Tensor):
        raise ValueError("embedding_matrix must be a torch.Tensor")
    N, F = feature_matrix.shape
    if embedding_matrix.shape[0] != N:
        raise ValueError("Row count of embedding_matrix must match feature_matrix")
    if N < 2:
        return {fid: 0.0 for fid in feature_indices}

    device = feature_matrix.device
    emb = embedding_matrix.to(device).float()
    with torch.no_grad():
        emb_norm = emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        emb_unit = emb / emb_norm  # (N,D)

    feats = feature_matrix.to(device).float()
    ms_scores = {}
    denom = float(N * (N - 1))

    # Pre-extract activation mins & maxs in a vectorized pass for requested indices (still per-feature loop for w-weighted sum)
    # Could batch but per-feature loop now O(N*D) each.
    for fid in tqdm(feature_indices, desc="MS_score(opt)", leave=False):
        # Min-max normalize activations for feature fid
        a = feats[:, fid]
        a_min = a.min()
        a_max = a.max()
        if a_max <= a_min:
            ms_scores[fid] = 0.0
            continue
        w = (a - a_min) / (a_max - a_min)  # [0,1]
        # Weighted sum
        ws = (w.unsqueeze(1) * emb_unit).sum(dim=0)
        numerator = (ws * ws).sum() - (w * w).sum()
        ms_scores[fid] = (numerator / denom).clamp_min(0).item()
    return ms_scores


def compute_ms_scores_slow(feature_matrix: torch.Tensor,
                          embedding_matrix: torch.Tensor,
                          feature_indices: list,
                          verify_sample: int = 0) -> dict:
    """Naive O(N²) implementation of MS_score computation for performance comparison.

    Original definition:
        MS^k = 1 / (N (N-1)) * sum_{n!=m} (ã_n^k ã_m^k) * s_{nm}
        where ã are min-max normalized activations for feature k, s_{nm} cosine similarity.

    This implementation directly computes the double sum, making it O(F * N² * D) complexity.
    Use this only for performance testing against the optimized version.

    Args:
        feature_matrix: (N, F) activations (can be signed).
        embedding_matrix: (N, D) semantic embeddings (will be L2-normalized per row).
        feature_indices: list of feature indices to score.
        verify_sample: ignored (kept for API compatibility).

    Returns:
        dict feature_index -> MS_score in [0,1].
    """
    if not isinstance(feature_matrix, torch.Tensor):
        raise ValueError("feature_matrix must be a torch.Tensor")
    if not isinstance(embedding_matrix, torch.Tensor):
        raise ValueError("embedding_matrix must be a torch.Tensor")
    N, F = feature_matrix.shape
    if embedding_matrix.shape[0] != N:
        raise ValueError("Row count of embedding_matrix must match feature_matrix")
    if N < 2:
        return {fid: 0.0 for fid in feature_indices}

    device = feature_matrix.device
    emb = embedding_matrix.to(device).float()

    # L2 normalize embeddings
    with torch.no_grad():
        emb_norm = emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        emb_unit = emb / emb_norm  # (N, D)

    feats = feature_matrix.to(device).float()
    ms_scores = {}
    denom = float(N * (N - 1))

    print(f"[MS_slow] Computing MS_score for {len(feature_indices)} features using O(N²) algorithm...")
    print(f"[MS_slow] This computes {len(feature_indices)} pairwise similarity matrices of size {N}x{N}")

    for fid in tqdm(feature_indices, desc="MS_score(slow)", leave=False):
        # Min-max normalize activations for feature fid
        a = feats[:, fid]
        a_min = a.min()
        a_max = a.max()
        if a_max <= a_min:
            ms_scores[fid] = 0.0
            continue
        w = (a - a_min) / (a_max - a_min)  # [0,1]

        # Compute the double sum: sum_{n!=m} w_n * w_m * (e_n · e_m)
        # Vectorized approach: compute full pairwise cosine similarity matrix
        with torch.no_grad():
            # Compute all pairwise cosine similarities: S[n,m] = e_n · e_m
            S = torch.mm(emb_unit, emb_unit.t())  # (N, N)

            # Compute weighted sum excluding diagonal
            # sum_{n!=m} w_n * w_m * S[n,m] = sum_all w_n * w_m * S[n,m] - sum_diag w_n * w_n * S[n,n]
            # Since S[n,n] = 1 for unit vectors: = (w^T S w) - sum(w^2)
            weighted_sum = torch.dot(w, torch.mv(S, w)) - torch.sum(w * w)
            total_sum = weighted_sum.item()

        ms_scores[fid] = max(0.0, total_sum / denom)

    return ms_scores


def compute_ms_scores_ratio(feature_matrix: torch.Tensor, embeddings: torch.Tensor, feature_indices=None):
    """Ratio-form MS_score: (||Σ w_i e_i||^2 - Σ w_i^2) / Σ_{i!=j} w_i w_j.

    Uses raw (non-normalized) activations with min-max scaling per feature to [0,1] like compute_ms_scores.
    Embeddings are L2-normalized row-wise internally.
    """
    if feature_indices is None:
        feature_indices = range(feature_matrix.shape[1])
    if not isinstance(feature_matrix, torch.Tensor):
        raise ValueError("feature_matrix must be a torch.Tensor")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("embeddings must be a torch.Tensor")
    device = feature_matrix.device
    emb = embeddings.to(device).float()
    emb = emb / (emb.norm(dim=1, keepdim=True).clamp_min(1e-12))
    feats = feature_matrix.to(device).float()
    scores = {}
    with torch.no_grad():
        for fid in feature_indices:
            a = feats[:, fid]
            a_min = a.min(); a_max = a.max()
            if a_max <= a_min:
                scores[int(fid)] = 0.0
                continue
            w = (a - a_min) / (a_max - a_min)
            ws = (w.unsqueeze(1) * emb).sum(dim=0)
            sum_sq = (w * w).sum()
            numerator = (ws * ws).sum() - sum_sq
            w_sum = w.sum()
            pair_mass = (w_sum * w_sum) - sum_sq
            if pair_mass <= 0:
                scores[int(fid)] = 0.0
            else:
                scores[int(fid)] = (numerator / pair_mass).clamp(min=0.0, max=1.0).item()
    return scores


def compute_activation_thresholds(feature_matrix: torch.Tensor, percentile: float = 95.0) -> torch.Tensor:
    """Compute per-feature activation thresholds using the specified percentile."""
    if not isinstance(feature_matrix, torch.Tensor):
        raise TypeError("feature_matrix must be a torch.Tensor")
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2-dimensional")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be within [0, 100]")
    if feature_matrix.numel() == 0:
        return torch.empty(feature_matrix.shape[1], device=feature_matrix.device)

    q = percentile / 100.0
    # Use float32 to ensure quantile stability even if original activations are half precision.
    fm = feature_matrix.to(dtype=torch.float32)
    return torch.quantile(fm, q, dim=0)


def compute_recall_by_class(feature_matrix: torch.Tensor,
                            labels: torch.Tensor,
                            thresholds: torch.Tensor,
                            num_classes: int) -> torch.Tensor:
    """
    Compute recall per class for each feature using precomputed activation thresholds.

    Args:
        feature_matrix: Tensor of shape [N, F] containing activations.
        labels: Tensor of shape [N] with integer class ids.
        thresholds: Tensor of shape [F] with activation thresholds per feature.
        num_classes: Number of distinct classes to evaluate.

    Returns:
        Tensor of shape [num_classes, F] with recall values in [0, 1].
    """
    if not isinstance(feature_matrix, torch.Tensor):
        raise TypeError("feature_matrix must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if not isinstance(thresholds, torch.Tensor):
        raise TypeError("thresholds must be a torch.Tensor")
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2-dimensional")
    if labels.ndim != 1:
        raise ValueError("labels must be 1-dimensional")
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be 1-dimensional")
    if feature_matrix.shape[0] != labels.shape[0]:
        raise ValueError("feature_matrix and labels must have matching row counts")
    if feature_matrix.shape[1] != thresholds.shape[0]:
        raise ValueError("feature count of feature_matrix and thresholds must match")
    if num_classes < 0:
        raise ValueError("num_classes must be non-negative")

    if feature_matrix.numel() == 0 or num_classes == 0:
        return torch.zeros((num_classes, thresholds.shape[0]), device=feature_matrix.device)

    labels = labels.to(dtype=torch.long, device=feature_matrix.device)
    thresholds = thresholds.to(device=feature_matrix.device)

    # Broadcast thresholds and compute activation mask.
    threshold_view = thresholds.view(1, -1)
    active_mask = feature_matrix > threshold_view  # [N, F] boolean

    recall = torch.zeros((num_classes, thresholds.shape[0]), device=feature_matrix.device, dtype=torch.float32)
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        class_count = int(class_mask.sum().item())
        if class_count == 0:
            continue
        class_active = active_mask[class_mask]
        if class_active.numel() == 0:
            continue
        # Convert to float for averaging; ensure division uses float32 to avoid integer truncation.
        recall[class_idx] = class_active.float().sum(dim=0) / float(class_count)
    return recall


def compute_precision_at_threshold(feature_matrix: torch.Tensor,
                                   labels: torch.Tensor,
                                   thresholds: torch.Tensor,
                                   num_classes: int) -> torch.Tensor:
    """Compute precision per class at the given activation thresholds."""
    if not isinstance(feature_matrix, torch.Tensor):
        raise TypeError("feature_matrix must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if not isinstance(thresholds, torch.Tensor):
        raise TypeError("thresholds must be a torch.Tensor")
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2-dimensional")
    if labels.ndim != 1:
        raise ValueError("labels must be 1-dimensional")
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be 1-dimensional")
    if feature_matrix.shape[0] != labels.shape[0]:
        raise ValueError("feature_matrix and labels must have matching row counts")
    if feature_matrix.shape[1] != thresholds.shape[0]:
        raise ValueError("feature count of feature_matrix and thresholds must match")
    if num_classes < 0:
        raise ValueError("num_classes must be non-negative")

    if feature_matrix.numel() == 0 or num_classes == 0:
        return torch.zeros((num_classes, thresholds.shape[0]), device=feature_matrix.device)

    device = feature_matrix.device
    labels = labels.to(dtype=torch.long, device=device)
    thresholds = thresholds.to(device=device)

    active_mask = feature_matrix > thresholds.view(1, -1)  # [N, F]
    counts_per_feature = active_mask.float().sum(dim=0).clamp_min(1.0)  # avoid divide-by-zero

    precision = torch.zeros((num_classes, thresholds.shape[0]), device=device, dtype=torch.float32)
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if not class_mask.any():
            continue
        class_active = active_mask[class_mask]
        if class_active.numel() == 0:
            continue
        precision[class_idx] = class_active.float().sum(dim=0) / counts_per_feature
    # Zero-out columns that had no activations at all
    zero_fire = (counts_per_feature == 1.0) & (~active_mask.any(dim=0))
    if zero_fire.any():
        precision[:, zero_fire] = 0.0
    return precision


def compute_auprc_by_class(feature_matrix: torch.Tensor,
                           labels: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
    """Compute AUPRC per class for each feature using activations as scores."""
    # Mirrors sklearn.metrics.average_precision_score: integrate precision–recall curve
    # generated by sweeping a descending activation threshold per feature/class.
    if not isinstance(feature_matrix, torch.Tensor):
        raise TypeError("feature_matrix must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2-dimensional")
    if labels.ndim != 1:
        raise ValueError("labels must be 1-dimensional")
    if feature_matrix.shape[0] != labels.shape[0]:
        raise ValueError("feature_matrix and labels must have matching row counts")
    if num_classes < 0:
        raise ValueError("num_classes must be non-negative")

    if feature_matrix.numel() == 0 or num_classes == 0:
        return torch.zeros((num_classes, feature_matrix.shape[1]), device=feature_matrix.device)

    device = feature_matrix.device
    labels = labels.to(dtype=torch.long, device=device)
    fm = feature_matrix.to(dtype=torch.float32, device=device)

    class_counts = torch.bincount(labels, minlength=num_classes).to(dtype=torch.float32, device=device)
    results = torch.zeros((num_classes, fm.shape[1]), dtype=torch.float32, device=device)

    if (class_counts > 0).sum() == 0:
        return results

    count_safe = class_counts.clamp_min(1.0)

    for feat_idx in range(fm.shape[1]):
        scores = fm[:, feat_idx]
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        sorted_labels = labels[sorted_idx]
        one_hot = F.one_hot(sorted_labels, num_classes=num_classes).to(dtype=torch.float32)
        cum_tp = torch.cumsum(one_hot, dim=0)

        denom = torch.arange(1, scores.shape[0] + 1, device=device, dtype=torch.float32).unsqueeze(1)
        precision = cum_tp / denom
        recall = cum_tp / count_safe.unsqueeze(0)

        precision_aug = torch.cat([torch.ones(1, num_classes, device=device), precision], dim=0)
        recall_aug = torch.cat([torch.zeros(1, num_classes, device=device), recall], dim=0)

        delta_recall = recall_aug[1:] - recall_aug[:-1]
        auprc_vec = torch.sum(delta_recall * precision_aug[1:], dim=0)

        auprc_vec = torch.where(class_counts > 0, auprc_vec, torch.zeros_like(auprc_vec))
        results[:, feat_idx] = auprc_vec.clamp(min=0.0, max=1.0)

    return results

def generate_baseline_interactive_cache(feature_matrix: torch.Tensor,
                                       labels,
                                       image_paths,
                                       split: str,
                                       feature_indices: list,
                                       mono_scores: dict,
                                       topk_per_cell: int = 50,
                                       out_root: str = "interactive_cache",
                                       cache_root: str = "cache"):
    """
    Generate interactive cache for baseline UNI features:
    - All 1024 UNI features
    - Heatmap matrix: mean absolute activation per (class, feature)
    - For each (class, feature): top-K sample indices and activation values
    - Include monosemanticity scores for sorting
    """
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    classes = torch.unique(labels_tensor).tolist()

    # All 1024 features
    F = len(feature_indices)
    feat_ids = np.array(feature_indices, dtype=np.int32)

    # Label names from cache
    label2desc = load_label_map(cache_root)
    class_names = [label2desc.get(int(c), f"Label_{int(c)}") for c in classes]

    heat_rows = []
    topk_idx_cube = []
    topk_val_cube = []
    class_counts = []

    for c in classes:
        mask = (labels_tensor == c)                                       # [N]
        cls_rows = torch.where(mask)[0]                                   # [Nc]
        cls_feats = feature_matrix[mask]                                   # [Nc, 1024]
        Nc = int(cls_feats.shape[0])
        class_counts.append(Nc)

        if Nc == 0:
            heat_rows.append(torch.zeros(F))
            topk_idx_cube.append(torch.full((F, topk_per_cell), -1, dtype=torch.long))
            topk_val_cube.append(torch.zeros(F, topk_per_cell))
            continue

        # Use absolute values for UNI embeddings (can be negative)
        cls_feats_abs = torch.abs(cls_feats)

        # Mean per class (normalizes class size)
        cls_mean = cls_feats_abs.mean(dim=0)                              # [1024]
        heat_rows.append(cls_mean[feature_indices])                       # [F]

        # Top-K samples for each feature within this class (use absolute values)
        cls_sel = cls_feats_abs[:, feature_indices]                       # [Nc, F]
        k_local = min(topk_per_cell, Nc)
        vals, idx_local = torch.topk(cls_sel, k=k_local, dim=0)           # [k_local, F]
        # Map class-local indices back to global sample indices
        global_idx = cls_rows[idx_local]                                   # [k_local, F]

        # Pad to fixed K
        if k_local < topk_per_cell:
            pad = topk_per_cell - k_local
            pad_idx = torch.full((pad, F), -1, dtype=global_idx.dtype)
            pad_val = torch.zeros(pad, F, dtype=vals.dtype)
            global_idx = torch.cat([global_idx, pad_idx], dim=0)          # [K, F]
            vals = torch.cat([vals, pad_val], dim=0)                       # [K, F]

        topk_idx_cube.append(global_idx.T)                                 # [F, K]
        topk_val_cube.append(vals.T)                                       # [F, K]

    H = torch.stack(heat_rows, dim=0).cpu().numpy().astype(np.float32)     # [C, F]
    topk_indices = torch.stack(topk_idx_cube, dim=0).cpu().numpy()         # [C, F, K]
    topk_values = torch.stack(topk_val_cube, dim=0).cpu().numpy().astype(np.float32)

    # Extract monosemanticity scores in feature order
    mono_values = np.array([mono_scores[fid]['M'] for fid in feature_indices], dtype=np.float32)
    mono_values_softmax = np.array([mono_scores[fid]['M_softmax'] for fid in feature_indices], dtype=np.float32)

    # Save arrays including monosemanticity scores
    np.savez_compressed(
        os.path.join(out_dir, "cache.npz"),
        classes=np.array(classes, dtype=np.int32),
        class_names=np.array(class_names, dtype=object),
        class_counts=np.array(class_counts, dtype=np.int32),
        top_feature_indices=feat_ids,
        heatmap=H,
        topk_indices=topk_indices,
        topk_values=topk_values,
        monosemanticity_scores=mono_values,  # Add monosemanticity scores (normalized)
        monosemanticity_scores_softmax=mono_values_softmax  # Add monosemanticity scores (softmax)
    )

    # Save image paths (JSON for readability)
    with open(os.path.join(out_dir, "image_paths.json"), "w") as f:
        json.dump(image_paths, f)

    print(f"[baseline_cache] Saved: {out_dir}/cache.npz and image_paths.json")
    print(f"[baseline_cache] Features: {F}, Classes: {len(classes)}")
    print(f"[baseline_cache] Monosemanticity range (normalized): {mono_values.min():.3f} - {mono_values.max():.3f}")
    print(f"[baseline_cache] Monosemanticity range (softmax): {mono_values_softmax.min():.3f} - {mono_values_softmax.max():.3f}")

# === NEW HELPERS: per-patch activations + metadata (place near other utility funcs) ===
def parse_slide_and_patch(image_path: str):
    """
    Heuristic extraction of slide_id / patch_id from path.
    Adjust if your naming differs.
    slide_id = parent directory name
    patch_id = filename stem
    """
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)
    parent = os.path.basename(os.path.dirname(image_path))
    return parent, stem

def save_locked_patch_activations(feature_matrix, locked_feature_indices, split_name, analysis_base_dir):
    """
    Save per-patch activations for locked features as numpy array:
      shape = [num_patches, K] where K = len(locked_feature_indices)
    File: analysis/patch-activations/{split}_locked_patch_activations.npy
    Also saves locked_feature_order.json once (the feature indices used, in order).
    """
    if not locked_feature_indices:
        print(f"[patch-activations] No locked features for split={split_name}; skipping.")
        return
    if not isinstance(feature_matrix, torch.Tensor):
        raise TypeError("feature_matrix must be a torch.Tensor")

    subdir = os.path.join(analysis_base_dir, "patch-activations")
    os.makedirs(subdir, exist_ok=True)

    fm_slice = feature_matrix[:, locked_feature_indices]  # [N, K]
    npy_path = os.path.join(subdir, f"{split_name}_locked_patch_activations.npy")
    np.save(npy_path, fm_slice.cpu().numpy())
    print(f"[patch-activations] Saved {split_name} activations: {npy_path} shape={tuple(fm_slice.shape)}")

    order_path = os.path.join(subdir, "locked_feature_order.json")
    if not os.path.exists(order_path):
        with open(order_path, "w") as f:
            json.dump({"locked_feature_indices": locked_feature_indices}, f, indent=2)
        print(f"[patch-activations] Saved feature order: {order_path}")

def save_patch_metadata(image_paths, labels, split_name: str, analysis_base_dir: str, cache_root: str = "cache"):
    """
    Save per-patch metadata aligned with feature_matrix row order.
    JSONL file: one record per line with fields:
      row_index, image_path, class_label_idx, class_label_name, slide_id, patch_id, split
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    label2desc = load_label_map(cache_root)

    out_dir = os.path.join(analysis_base_dir, "patch-activations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split_name}_patch_metadata.jsonl")

    with open(out_path, "w") as f:
        for idx, (p, lab_idx) in enumerate(zip(image_paths, labels_np)):
            slide_id, patch_id = parse_slide_and_patch(p)
            record = {
                "row_index": idx,
                "image_path": p,
                "class_label_idx": int(lab_idx),
                "class_label_name": label2desc.get(int(lab_idx), f"Label_{int(lab_idx)}"),
                "slide_id": slide_id,
                "patch_id": patch_id,
                "split": split_name
            }
            f.write(json.dumps(record) + "\n")
    print(f"[patch-metadata] Saved {split_name} metadata: {out_path} ({len(image_paths)} rows)")


def save_feature_visualizations_fast(
    feature_matrix,                  # [N, F] torch.Tensor
    image_paths,                     # list[str], len N
    labels,                          # torch.Tensor|np.ndarray|list of ints, len N
    locked_feature_indices,          # iterable[int]
    split_name,
    feature_analysis,                # dict[int] -> {...}
    base_dir="top-k-features-v0.2",
    top_k=25,
    grid_images=100,
    cols=10,                         # fixed columns for square 10x10 layout
    tile_size=(160, 160),            # per-tile image size (smaller for dense grids)
    pad=6,                           # spacing between tiles
    style='mpl',                     # 'mpl' = titles ABOVE tiles (classic look), 'compact' = overlay inside tiles
    png=True,                        # True: PNG (like before). False: JPEG (faster/smaller)
    cache_root="cache"               # cache directory for label map
):
    # ---- labels to numpy ----
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    # ---- label mapping ----
    label2desc = load_label_map(cache_root)
    label2desc_list = [label2desc.get(i, f"Label_{i}") for i in range(max(label2desc.keys()) + 1)]

    # ---- output/skip ----
    split_dir = Path(base_dir) / split_name
    if split_dir.exists() and any(split_dir.iterdir()):
        print(f"Skipping {split_dir}, already exists and is not empty.")
        return
    split_dir.mkdir(parents=True, exist_ok=True)

    # ---- batched top-k across ALL features ----
    viz_k = min(max(top_k, grid_images), feature_matrix.size(0))
    # Use absolute values so we capture strongest magnitude activations regardless of sign
    top_vals, top_idx = torch.topk(torch.abs(feature_matrix), k=viz_k, dim=0, largest=True, sorted=True)

    top_vals = top_vals.cpu()
    top_idx  = top_idx.cpu()

    # ---- fonts ----
    try:
        font = ImageFont.load_default()
        bold = font
    except Exception:
        font = bold = None

    # ---- geometry ----
    rows = math.ceil(grid_images / cols)
    tile_w, tile_h = tile_size
    # title area per tile (two lines like matplotlib titles)
    line_h = 10  # slightly smaller text for denser grid
    title_h = 2 * line_h + 4  # two lines + small gap

    grid_w = cols * tile_w + (cols - 1) * pad
    grid_h = rows * (title_h + tile_h) + (rows - 1) * pad

    # margins roughly match your previous figure padding
    M_L, M_T, M_R, M_B = 26, 26, 26, 42
    header_h = 24  # "Feature {id}" main title row
    info_h   = 18  # bottom subtitle row

    canvas_w = M_L + grid_w + M_R
    canvas_h = M_T + header_h + grid_h + info_h + M_B

    # ---- thumbnail cache ----
    @lru_cache(maxsize=2048)
    def load_thumb(sample_idx: int) -> Image.Image:
        p = image_paths[sample_idx]
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return im.resize(tile_size, Image.BILINEAR)

    made_dirs = set()
    for feat_idx in tqdm(locked_feature_indices, desc=f"viz:{split_name}", unit="feat"):
        info = feature_analysis[feat_idx]
        maj_lab  = info['majority_label']
        maj_name = info['majority_label_name']
        purity   = info['purity']
        purity_k = info['purity_k']

        out_dir = split_dir / maj_name
        if out_dir not in made_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            made_dirs.add(out_dir)

        n_disp = min(grid_images, viz_k)
        idxs = top_idx[:n_disp, feat_idx].numpy()
        vals = top_vals[:n_disp, feat_idx].numpy()
        # Get original signed values for display
        original_vals = feature_matrix[idxs, feat_idx].cpu().numpy()


        # canvas
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)

        # main title
        main_title = f"Feature {feat_idx}"
        # Try to use a larger font or create bold effect
        try:
            title_font = ImageFont.truetype("arial.ttf", 25)
        except (OSError, IOError):
            try:
                title_font = ImageFont.load_default().font_variant(size=25)
            except:
                title_font = bold if bold else font

        # Calculate text width for centering
        bbox = draw.textbbox((0, 0), main_title, font=title_font)
        text_width = bbox[2] - bbox[0]
        center_x = M_L + (grid_w - text_width) // 2

        draw.text((center_x, M_T), main_title, fill=(0,0,0), font=title_font)

        # grid origin
        gx = M_L
        gy = M_T + header_h + 40  # leave space for subtitle block

        # paste tiles
        for i in range(n_disp):
            r = i // cols
            c = i % cols
            x0 = gx + c * (tile_w + pad)
            y0 = gy + r * (tile_h + title_h + pad)

            idx_i = int(idxs[i])
            val_i = float(original_vals[i])
            lab_i = int(labels_np[idx_i])
            lab_name = label2desc_list[lab_i] if lab_i < len(label2desc_list) else f"Label_{lab_i}"

            # "mpl" style: text above tile, two lines
            if style == 'mpl':
                # Line 1: Value
                t1 = f"Value: {val_i:.2f}"
                # Line 2: Label
                t2 = f"Label: {lab_name}"
                color = (192, 0, 0) if lab_i != maj_lab else (0, 0, 0)
                draw.text((x0, y0), t1, fill=color, font=font if font else None)
                draw.text((x0, y0 + line_h + 2), t2, fill=color, font=font if font else None)
                img_y = y0 + title_h
            else:
                # "compact" overlay: text at bottom-left inside the tile
                img_y = y0
            # image
            img = load_thumb(idx_i)
            canvas.paste(img, (x0, img_y))

            # extras for 'compact' overlay
            if style != 'mpl':
                text = f"{val_i:.2f} · {lab_name}"
                color = (192, 0, 0) if lab_i != maj_lab else (0, 0, 0)
                draw.text((x0 + 6, img_y + tile_h - 16), text, fill=color, font=font if font else None)

            # thin border like an axis frame
            draw.rectangle([x0, img_y, x0 + tile_w - 1, img_y + tile_h - 1], outline=(0,0,0), width=1)

        # bottom info line moved to below title with 2 blank lines gap
        try:
            info_font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            try:
                info_font = ImageFont.load_default().font_variant(size=20)
            except:
                info_font = bold if bold else font

        meta = f"Purity@{purity_k}={purity:.2f} | Majority={maj_name} | Split={split_name}"
        bbox = draw.textbbox((0, 0), meta, font=info_font)
        meta_width = bbox[2] - bbox[0]
        meta_center_x = M_L + (grid_w - meta_width) // 2
        meta_y = M_T + header_h + 6

        draw.text((meta_center_x, meta_y), meta, fill=(0,0,0), font=info_font)

        # adjust grid origin to account for moved info line
        gy = M_T + header_h + 34  # additional space for the moved info line

        out_path = out_dir / f"feature_{feat_idx}.{'png' if png else 'jpg'}"
        if png:
            canvas.save(out_path, format="PNG", compress_level=1)  # faster PNG
        else:
            canvas.save(out_path, format="JPEG", quality=85, optimize=True)

    load_thumb.cache_clear()


def compute_coverage_stats(feature_matrix: torch.Tensor, eps: float = 1e-6):
    """
    Compute coverage statistics per feature.
    coverage(j) = mean(z[:, j] > eps) - fraction of samples where feature j is active
    """
    num_samples, hidden_dim = feature_matrix.shape
    active_mask = (feature_matrix > eps)

    # Coverage per feature: fraction of samples where each feature is active
    coverage_per_feature = active_mask.float().mean(dim=0)  # [hidden_dim]

    stats = {
        "num_samples": int(num_samples),
        "hidden_dim": int(hidden_dim),
        "mean_coverage": float(coverage_per_feature.mean()),
        "median_coverage": float(coverage_per_feature.median()),
        "std_coverage": float(coverage_per_feature.std()),
        "min_coverage": float(coverage_per_feature.min()),
        "max_coverage": float(coverage_per_feature.max()),
        "coverage_per_feature": coverage_per_feature.tolist()  # For detailed analysis
    }

    return stats

def analyze_dead_and_near_dead_features(train_coverage_stats, test_coverage_stats,
                                       tau=1e-5, locked_feature_indices=None):
    """
    Analyze dead and near-dead features across train/test splits.

    Args:
        train_coverage_stats: Coverage stats from train split
        test_coverage_stats: Coverage stats from test split
        tau: Threshold for near-dead classification
        locked_feature_indices: Optional list to focus analysis on specific features
    """
    train_coverage = torch.tensor(train_coverage_stats["coverage_per_feature"])
    test_coverage = torch.tensor(test_coverage_stats["coverage_per_feature"])

    # Handle streaming mode: if train/test have different sizes, test is locked features only
    # In this case, we can only analyze the locked features
    if len(train_coverage) != len(test_coverage):
        if locked_feature_indices is not None and len(test_coverage) == len(locked_feature_indices):
            # Test coverage is for locked features only (streaming mode)
            # Extract corresponding train coverage for locked features
            locked_indices_tensor = torch.tensor(locked_feature_indices, dtype=torch.long)
            train_coverage = train_coverage[locked_indices_tensor]
        else:
            # Size mismatch but can't resolve - skip analysis
            print(f"Warning: Coverage tensor size mismatch (train={len(train_coverage)}, test={len(test_coverage)}). Skipping dead feature analysis.")
            return {
                "tau_threshold": tau,
                "total_features": 0,
                "dead_features": {"count": 0, "percentage": 0.0, "indices": []},
                "near_dead_features": {"count": 0, "percentage": 0.0, "indices": []},
                "active_features": {"count": 0, "percentage": 0.0},
                "split_discrepancy": {"train_only_count": 0, "test_only_count": 0, "train_only_indices": [], "test_only_indices": []},
                "coverage_correlation": 0.0,
                "locked_feature_analysis_note": "Analysis skipped due to tensor size mismatch"
            }

    # Dead features: zero coverage on both splits
    dead_mask = (train_coverage == 0) & (test_coverage == 0)
    dead_features = torch.where(dead_mask)[0].tolist()

    # Near-dead features: both coverages below threshold
    near_dead_mask = (train_coverage < tau) & (test_coverage < tau) & ~dead_mask
    near_dead_features = torch.where(near_dead_mask)[0].tolist()

    # Coverage discrepancy: features active on one split but not the other
    train_only_mask = (train_coverage > tau) & (test_coverage < tau)
    test_only_mask = (test_coverage > tau) & (train_coverage < tau)
    train_only_features = torch.where(train_only_mask)[0].tolist()
    test_only_features = torch.where(test_only_mask)[0].tolist()

    # Robustness checks
    total_features = len(train_coverage)
    dead_percentage = len(dead_features) / total_features * 100 if total_features > 0 else 0.0
    near_dead_percentage = len(near_dead_features) / total_features * 100 if total_features > 0 else 0.0
    active_features = total_features - len(dead_features) - len(near_dead_features)
    active_percentage = active_features / total_features * 100 if total_features > 0 else 0.0

    # Coverage correlation between splits
    if len(train_coverage) > 1 and len(test_coverage) > 1:
        coverage_corr = torch.corrcoef(torch.stack([train_coverage, test_coverage]))[0, 1].item()
    else:
        coverage_corr = 0.0

    analysis = {
        "tau_threshold": tau,
        "total_features": total_features,
        "dead_features": {
            "count": len(dead_features),
            "percentage": dead_percentage,
            "indices": dead_features[:20]  # Show first 20 for brevity
        },
        "near_dead_features": {
            "count": len(near_dead_features),
            "percentage": near_dead_percentage,
            "indices": near_dead_features[:20]
        },
        "active_features": {
            "count": active_features,
            "percentage": active_percentage
        },
        "split_discrepancy": {
            "train_only_count": len(train_only_features),
            "test_only_count": len(test_only_features),
            "train_only_indices": train_only_features[:10],
            "test_only_indices": test_only_features[:10]
        },
        "robustness_metrics": {
            "coverage_correlation": coverage_corr,
            "dead_ratio_is_concerning": dead_percentage > 50,  # Flag if >50% dead
            "near_dead_ratio_is_concerning": near_dead_percentage > 30,  # Flag if >30% near-dead
            "split_consistency_good": coverage_corr > 0.7  # Flag if correlation is good
        }
    }

    # If locked features provided, analyze them specifically
    if locked_feature_indices is not None:
        locked_dead = [f for f in locked_feature_indices if f in dead_features]
        locked_near_dead = [f for f in locked_feature_indices if f in near_dead_features]
        locked_train_only = [f for f in locked_feature_indices if f in train_only_features]
        locked_test_only = [f for f in locked_feature_indices if f in test_only_features]

        analysis["locked_features_analysis"] = {
            "total_locked": len(locked_feature_indices),
            "locked_dead": {
                "count": len(locked_dead),
                "percentage": len(locked_dead) / len(locked_feature_indices) * 100,
                "indices": locked_dead
            },
            "locked_near_dead": {
                "count": len(locked_near_dead),
                "percentage": len(locked_near_dead) / len(locked_feature_indices) * 100,
                "indices": locked_near_dead
            },
            "locked_split_discrepancy": {
                "train_only": locked_train_only,
                "test_only": locked_test_only
            }
        }

    return analysis

def print_coverage_analysis(train_coverage_stats, test_coverage_stats, dead_analysis, file=None):
    """Print comprehensive coverage analysis"""
    def pprint(*args, **kwargs):
        print(*args, **kwargs, file=file)

    pprint("\n=== COVERAGE ANALYSIS ===")
    pprint(f"Coverage = fraction of samples where feature is active (> eps)")
    pprint(f"Threshold τ = {dead_analysis['tau_threshold']} for near-dead classification")

    pprint(f"\nTrain split coverage:")
    pprint(f"  Mean coverage: {train_coverage_stats['mean_coverage']:.6f}")
    pprint(f"  Median coverage: {train_coverage_stats['median_coverage']:.6f}")
    pprint(f"  Std coverage: {train_coverage_stats['std_coverage']:.6f}")
    pprint(f"  Min coverage: {train_coverage_stats['min_coverage']:.6f}")
    pprint(f"  Max coverage: {train_coverage_stats['max_coverage']:.6f}")

    pprint(f"\nTest split coverage:")
    pprint(f"  Mean coverage: {test_coverage_stats['mean_coverage']:.6f}")
    pprint(f"  Median coverage: {test_coverage_stats['median_coverage']:.6f}")
    pprint(f"  Std coverage: {test_coverage_stats['std_coverage']:.6f}")
    pprint(f"  Min coverage: {test_coverage_stats['min_coverage']:.6f}")
    pprint(f"  Max coverage: {test_coverage_stats['max_coverage']:.6f}")

    pprint(f"\n=== DEAD & NEAR-DEAD FEATURE ANALYSIS ===")
    pprint(f"Total features: {dead_analysis['total_features']:,}")

    dead = dead_analysis['dead_features']
    pprint(f"Dead features (zero coverage both splits): {dead['count']:,} ({dead['percentage']:.1f}%)")
    if dead['count'] > 0:
        pprint(f"  First 20 dead feature indices: {dead['indices']}")

    near_dead = dead_analysis['near_dead_features']
    pprint(f"Near-dead features (coverage < τ both splits): {near_dead['count']:,} ({near_dead['percentage']:.1f}%)")
    if near_dead['count'] > 0:
        pprint(f"  First 20 near-dead feature indices: {near_dead['indices']}")

    active = dead_analysis['active_features']
    pprint(f"Active features (not dead or near-dead): {active['count']:,} ({active['percentage']:.1f}%)")

    disc = dead_analysis['split_discrepancy']
    pprint(f"\nSplit discrepancy:")
    pprint(f"  Train-only active (test < τ): {disc['train_only_count']:,}")
    pprint(f"  Test-only active (train < τ): {disc['test_only_count']:,}")
    if disc['train_only_count'] > 0:
        pprint(f"    First 10 train-only indices: {disc['train_only_indices']}")
    if disc['test_only_count'] > 0:
        pprint(f"    First 10 test-only indices: {disc['test_only_indices']}")

    robust = dead_analysis['robustness_metrics']
    pprint(f"\n=== ROBUSTNESS ASSESSMENT ===")
    pprint(f"Coverage correlation (train vs test): {robust['coverage_correlation']:.4f}")
    pprint(f"Split consistency: {'✓ GOOD' if robust['split_consistency_good'] else '✗ POOR'} (correlation > 0.7)")
    pprint(f"Dead ratio: {'⚠ CONCERNING' if robust['dead_ratio_is_concerning'] else '✓ ACCEPTABLE'} ({dead['percentage']:.1f}% dead)")
    pprint(f"Near-dead ratio: {'⚠ CONCERNING' if robust['near_dead_ratio_is_concerning'] else '✓ ACCEPTABLE'} ({near_dead['percentage']:.1f}% near-dead)")

    # Locked features analysis if available
    if 'locked_features_analysis' in dead_analysis:
        locked = dead_analysis['locked_features_analysis']
        pprint(f"\n=== LOCKED FEATURES COVERAGE ===")
        pprint(f"Total locked features: {locked['total_locked']}")

        locked_dead = locked['locked_dead']
        pprint(f"Locked dead features: {locked_dead['count']}/{locked['total_locked']} ({locked_dead['percentage']:.1f}%)")
        if locked_dead['count'] > 0:
            pprint(f"  Dead locked indices: {locked_dead['indices']}")

        locked_near = locked['locked_near_dead']
        pprint(f"Locked near-dead features: {locked_near['count']}/{locked['total_locked']} ({locked_near['percentage']:.1f}%)")
        if locked_near['count'] > 0:
            pprint(f"  Near-dead locked indices: {locked_near['indices']}")

        locked_disc = locked['locked_split_discrepancy']
        if locked_disc['train_only'] or locked_disc['test_only']:
            pprint(f"Locked split discrepancy:")
            if locked_disc['train_only']:
                pprint(f"  Train-only locked: {locked_disc['train_only']}")
            if locked_disc['test_only']:
                pprint(f"  Test-only locked: {locked_disc['test_only']}")

def compute_activation_stats(feature_matrix: torch.Tensor, eps: float = 1e-6):
    """
    Compute activation sparsity stats across the dataset.
    eps: threshold to consider a unit 'active' (use >0 for ReLU codes; 1e-6 is safer).
    """
    num_samples, hidden_dim = feature_matrix.shape
    active_mask = (feature_matrix > eps)

    active_per_sample = active_mask.sum(dim=1)          # number of active units per sample
    active_per_feature = active_mask.sum(dim=0)         # number of samples that activate each feature

    stats = {
        "num_samples": int(num_samples),
        "hidden_dim": int(hidden_dim),
        "mean_active_units_per_sample": float(active_per_sample.float().mean()),
        "median_active_units_per_sample": float(active_per_sample.float().median()),
        "features_ever_active": int((active_per_feature > 0).sum()),
        "features_active_>0.1%_samples": int((active_per_feature >= max(1, int(0.001*num_samples))).sum()),
        "features_active_>1%_samples": int((active_per_feature >= int(0.01*num_samples)).sum()),
        "features_active_>5%_samples": int((active_per_feature >= int(0.05*num_samples)).sum()),
        "features_active_>10%_samples": int((active_per_feature >= int(0.10*num_samples)).sum()),
        "mean_activation_rate_per_feature": float((active_per_feature.float()/num_samples).mean()),
        "median_activation_rate_per_feature": float((active_per_feature.float()/num_samples).median()),
    }
    return stats

def generate_class_feature_heatmap(feature_matrix: torch.Tensor, labels, top_features: int = 20,
                                   save_dir: str = "class-label-hists", flip_axes: bool = False, sort_features: bool = True,
                                   feature_indices: Optional[Sequence[int]] = None, cache_root: str = "cache"):
    """
    Heatmap:
    - Values = mean activation per class (normalizes class imbalance via mean over samples).
    - Columns = features, Rows = classes. Set flip_axes=True to swap.
    - Annotates each cell with value and outlines top-3 classes per feature.
    - If sort_features=True, sorts feature indices (x-axis) numerically.
    """
    os.makedirs(save_dir, exist_ok=True)

    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    unique_labels = torch.unique(labels_tensor)

    # Limit heatmaps to max 25 features
    top_features = min(top_features, 25)

    # Global top-N features by overall mean activation
    overall_mean_activation = feature_matrix.mean(dim=0)  # [hidden_dim]
    k = min(top_features, overall_mean_activation.shape[0])
    _, top_feature_indices = torch.topk(overall_mean_activation, k=k)
    feat_labels = [int(i) for i in top_feature_indices]

    # Select features: lock to provided indices if given, else pick global top-N from this matrix
    if feature_indices is not None and len(feature_indices) > 0:
        top_feature_indices = torch.as_tensor(feature_indices, dtype=torch.long, device=feature_matrix.device)
        k = int(top_feature_indices.numel())
        feat_labels = [int(i) for i in top_feature_indices.tolist()]
    else:
        overall_mean_activation = feature_matrix.mean(dim=0)  # [hidden_dim]
        k = min(top_features, overall_mean_activation.shape[0])
        _, top_feature_indices = torch.topk(overall_mean_activation, k=k)
        feat_labels = [int(i) for i in top_feature_indices]

    # Sort feature indices and update top_feature_indices and feat_labels
    if sort_features and k > 0:
        sorted_pairs = sorted(zip(feat_labels, top_feature_indices.tolist()))
        if sorted_pairs:
            feat_labels_sorted, indices_sorted = zip(*sorted_pairs)
            feat_labels = [str(i) for i in feat_labels_sorted]
            top_feature_indices = torch.as_tensor(list(indices_sorted), dtype=torch.long, device=feature_matrix.device)

    label2desc = load_label_map(cache_root)

    rows = []
    class_names = []
    class_counts = []

    for lbl in sorted(unique_labels.tolist()):
        class_mask = (labels_tensor == lbl)
        class_samples = feature_matrix[class_mask]  # [n_class, hidden_dim]
        n_class = int(class_samples.shape[0])
        if n_class == 0:
            continue

        # Mean over samples in class (normalizes for class-size imbalance)
        class_mean = class_samples.mean(dim=0)  # [hidden_dim]
        class_top = class_mean[top_feature_indices].cpu().numpy()  # [k]

        rows.append(class_top)
        cls_name = label2desc.get(int(lbl), f"Label_{int(lbl)}")
        class_names.append(f"{cls_name} ({int(lbl)})")
        class_counts.append(n_class)

    if not rows:
        print("No classes found for heatmap.")
        return

    H = np.vstack(rows)  # [num_classes, k]

    # Optional axis flip (features on Y, classes on X)
    if flip_axes:
        H = H.T  # [k, num_classes]

    fig_h = 1 + 0.5 * (H.shape[0] if flip_axes else len(class_names))
    plt.figure(figsize=(max(12, k), fig_h))
    ax = plt.gca()
    im = ax.imshow(H, cmap='viridis', aspect='auto')

    plt.colorbar(im, ax=ax, label="Mean activation strength")

    if flip_axes:
        ax.set_yticks(range(k))
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels([f"{n} (n={c})" for n, c in zip(class_names, class_counts)],
                           rotation=45, ha='right', fontsize=9)
        ax.set_xlabel("Class label")
        ax.set_ylabel("Feature index (global top, sorted)")
        title_suffix = " (features on Y, classes on X)"
    else:
        ax.set_xticks(range(k))
        ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels([f"{n} (n={c})" for n, c in zip(class_names, class_counts)], fontsize=9)
        ax.set_xlabel("Feature index (global top, sorted)")
        ax.set_ylabel("Class label")
        title_suffix = " (classes on Y, features on X)"

    ax.set_title(f"Class vs Top {k} Features Heatmap • values are class means{title_suffix}")

    # Annotate each cell with its value (choose text color for contrast)
    for i in range(H.shape[0]):          # row
        for j in range(H.shape[1]):      # col
            v = H[i, j]
            txt_color = 'white' if im.norm(v) > 0.5 else 'black'
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', color=txt_color, fontsize=8)

    # Outline top-3 classes per feature (or top-3 features per class if flipped)
    if flip_axes:
        # Rows = features, cols = classes
        for i in range(H.shape[0]):  # feature row
            top_js = np.argsort(H[i, :])[-3:]
            for j in top_js:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor='white', linewidth=2.2))
    else:
        # Cols = features, rows = classes
        for j in range(H.shape[1]):  # feature column
            top_is = np.argsort(H[:, j])[-3:]
            for i in top_is:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor='white', linewidth=2.2))

    plt.tight_layout()
    suffix = "-flipped" if flip_axes else ""
    out_path = os.path.join(save_dir, f"class_vs_top{k}_features_heatmap{suffix}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def generate_interactive_cache(feature_matrix: torch.Tensor,
                               labels,
                               image_paths,
                               split: str,
                               top_features: int = 20,
                               topk_per_cell: int = 50,
                               eps: float = 1e-6,
                               out_root: str = "interactive_cache",
                               feature_indices: Optional[Sequence[int]] = None,
                               cache_root: str = "cache"):
    """
    Precompute cache for interactive viz:
    - Global top-N features by overall mean activation
    - Heatmap matrix: mean activation per (class, feature)
    - For each (class, feature): top-K sample indices and activation values
    """
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    classes = torch.unique(labels_tensor).tolist()

    # Global top-N features
    overall_mean = feature_matrix.mean(dim=0)                            # [H]
    F = min(int(top_features), overall_mean.shape[0])
    _, top_feat_idx = torch.topk(overall_mean, k=F)                      # [F]
    feat_ids = top_feat_idx.cpu().numpy().astype(np.int32)

    # Feature set: lock to provided indices if given, else pick global top-N
    if feature_indices is not None and len(feature_indices) > 0:
        # feature_indices are global indices, but feature_matrix is already locked in streaming mode
        # Check if matrix is already locked (columns match feature_indices length)
        if feature_matrix.shape[1] == len(feature_indices):
            # Matrix is locked: use local indices [0, 1, 2, ...]
            top_feat_idx = torch.arange(len(feature_indices), dtype=torch.long, device=feature_matrix.device)
            feat_ids = np.array(feature_indices, dtype=np.int32)  # Store global IDs for metadata
        else:
            # Matrix is full: use global indices to slice
            top_feat_idx = torch.as_tensor(feature_indices, dtype=torch.long, device=feature_matrix.device)
            feat_ids = top_feat_idx.detach().cpu().numpy().astype(np.int32)
        F = int(top_feat_idx.numel())
    else:
        overall_mean = feature_matrix.mean(dim=0)                         # [H]
        F = min(int(top_features), overall_mean.shape[0])
        _, top_feat_idx = torch.topk(overall_mean, k=F)                   # [F]
        feat_ids = top_feat_idx.detach().cpu().numpy().astype(np.int32)

    # Optional label names
    label2desc = load_label_map(cache_root)
    class_names = [label2desc.get(int(c), f"Label_{int(c)}") for c in classes]

    heat_rows = []
    topk_idx_cube = []
    topk_val_cube = []
    class_counts = []

    for c in classes:
        mask = (labels_tensor == c)                                       # [N]
        cls_rows = torch.where(mask)[0]                                   # [Nc]
        cls_feats = feature_matrix[mask]                                   # [Nc, H]
        Nc = int(cls_feats.shape[0])
        class_counts.append(Nc)

        if Nc == 0:
            heat_rows.append(torch.zeros(F))
            topk_idx_cube.append(torch.full((F, topk_per_cell), -1, dtype=torch.long))
            topk_val_cube.append(torch.zeros(F, topk_per_cell))
            continue

        # Mean per class (normalizes class size)
        cls_mean = cls_feats.mean(dim=0)                                  # [H]
        heat_rows.append(cls_mean[top_feat_idx])                           # [F]

        # Top-K samples for each selected feature within this class
        cls_sel = cls_feats[:, top_feat_idx]                               # [Nc, F]
        k_local = min(topk_per_cell, Nc)
        vals, idx_local = torch.topk(cls_sel, k=k_local, dim=0)           # [k_local, F]
        # Map class-local indices back to global sample indices
        global_idx = cls_rows[idx_local]                                   # [k_local, F]

        # Pad to fixed K
        if k_local < topk_per_cell:
            pad = topk_per_cell - k_local
            pad_idx = torch.full((pad, F), -1, dtype=global_idx.dtype)
            pad_val = torch.zeros(pad, F, dtype=vals.dtype)
            global_idx = torch.cat([global_idx, pad_idx], dim=0)          # [K, F]
            vals = torch.cat([vals, pad_val], dim=0)                       # [K, F]

        topk_idx_cube.append(global_idx.T)                                 # [F, K]
        topk_val_cube.append(vals.T)                                       # [F, K]

    H = torch.stack(heat_rows, dim=0).cpu().numpy().astype(np.float32)     # [C, F]
    topk_indices = torch.stack(topk_idx_cube, dim=0).cpu().numpy()         # [C, F, K]
    topk_values = torch.stack(topk_val_cube, dim=0).cpu().numpy().astype(np.float32)

    # Save arrays
    np.savez_compressed(
        os.path.join(out_dir, "cache.npz"),
        classes=np.array(classes, dtype=np.int32),
        class_names=np.array(class_names, dtype=object),
        class_counts=np.array(class_counts, dtype=np.int32),
        top_feature_indices=feat_ids,
        heatmap=H,
        topk_indices=topk_indices,
        topk_values=topk_values,
    )
    # Save image paths (JSON for readability)
    with open(os.path.join(out_dir, "image_paths.json"), "w") as f:
        json.dump(image_paths, f)

    print(f"[interactive_cache] Saved: {out_dir}/cache.npz and image_paths.json")


def compute_class_activation_means(
    feature_matrix: torch.Tensor,
    labels,
    feature_indices: Optional[Sequence[int]] = None,
    cache_root: str = "cache",
    use_absolute: bool = False,
) -> tuple[np.ndarray, List[int], List[str], List[int]]:
    """
    Compute mean activation per class for each selected feature.

    Returns:
        class_means: np.ndarray shape [num_features, num_classes]
        selected_features: list of feature indices (ints)
        class_names: list of class names aligned with columns
        class_ids: list of class indices aligned with columns
    """
    if feature_matrix.numel() == 0:
        raise ValueError("feature_matrix is empty; cannot compute class means.")

    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.long)
    labels_tensor = labels_tensor.to(dtype=torch.long, device=feature_matrix.device)
    unique_classes = torch.unique(labels_tensor).tolist()

    if feature_indices is not None and len(feature_indices) > 0:
        feat_idx_tensor = torch.as_tensor(feature_indices, dtype=torch.long, device=feature_matrix.device)
    else:
        feat_idx_tensor = torch.arange(feature_matrix.shape[1], device=feature_matrix.device, dtype=torch.long)
        feature_indices = feat_idx_tensor.tolist()

    label2desc = load_label_map(cache_root)
    class_names = [label2desc.get(int(c), f"Label_{int(c)}") for c in unique_classes]

    mean_rows: List[torch.Tensor] = []
    for cls in unique_classes:
        mask = labels_tensor == cls
        if mask.sum() == 0:
            mean_rows.append(torch.zeros(feat_idx_tensor.numel(), device=feature_matrix.device))
            continue
        cls_feats = feature_matrix[mask][:, feat_idx_tensor]
        if use_absolute:
            cls_feats = cls_feats.abs()
        mean_rows.append(cls_feats.mean(dim=0))

    class_means = torch.stack(mean_rows, dim=0).cpu().numpy().astype(np.float32)  # [num_classes, num_features]
    class_means = np.transpose(class_means)  # [num_features, num_classes]
    class_ids = [int(c) for c in unique_classes]
    selected_features = [int(idx) for idx in feature_indices]
    return class_means, selected_features, class_names, class_ids


def compute_top_patches_for_features(
    feature_matrix: torch.Tensor,
    image_paths: Sequence[str],
    labels,
    feature_indices: Sequence[int],
    top_k: int = 3,
    cache_root: str = "cache",
    use_absolute: bool = False,
) -> Dict[int, List[Dict[str, object]]]:
    """
    For each neuron (feature), return metadata for top-k samples by activation.
    """
    if len(image_paths) != feature_matrix.shape[0]:
        raise ValueError("Number of image_paths must match number of rows in feature_matrix.")

    labels_tensor = None
    label2desc = None
    if labels is not None:
        labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.long)
        labels_tensor = labels_tensor.to(dtype=torch.long, device=feature_matrix.device)
        label2desc = load_label_map(cache_root)

    feature_dict: Dict[int, List[Dict[str, object]]] = {}
    fm = feature_matrix.to(device=feature_matrix.device)
    for fid in feature_indices:
        idx_tensor = torch.as_tensor(fid, device=fm.device, dtype=torch.long)
        activations = fm[:, idx_tensor]
        if use_absolute:
            activations = activations.abs()
        k = min(int(top_k), activations.shape[0])
        if k <= 0:
            feature_dict[int(fid)] = []
            continue
        values, indices = torch.topk(activations, k=k)
        records: List[Dict[str, object]] = []
        for rank, (value, row_idx) in enumerate(zip(values.tolist(), indices.tolist())):
            entry: Dict[str, object] = {
                "rank": int(rank),
                "activation": float(value),
                "row_index": int(row_idx),
                "image_path": image_paths[row_idx],
            }
            if labels_tensor is not None:
                label_idx = int(labels_tensor[row_idx].item())
                entry["label_index"] = label_idx
                if label2desc is not None:
                    entry["label_name"] = label2desc.get(label_idx, f"Label_{label_idx}")
            records.append(entry)
        feature_dict[int(fid)] = records
    return feature_dict


def _normalize_class_means_to_probabilities(class_means: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert per-feature class mean activations into probability distributions.

    Ensures non-negative values by shifting rows if needed and adds epsilon for numerical stability.
    """
    probs = np.array(class_means, dtype=np.float64, copy=True)
    if probs.ndim != 2:
        raise ValueError("class_means must be a 2D array")

    # Shift each row to be non-negative if negatives are present
    row_min = probs.min(axis=1, keepdims=True)
    negative_mask = row_min < 0.0
    if np.any(negative_mask):
        probs[negative_mask] -= row_min[negative_mask]

    probs = np.clip(probs, 0.0, None)
    probs += eps

    row_sums = probs.sum(axis=1, keepdims=True)
    zero_mask = row_sums <= eps * probs.shape[1]
    if np.any(zero_mask):
        probs[zero_mask] = 1.0
        row_sums[zero_mask] = probs.shape[1]

    probs /= row_sums
    return probs


def _pairwise_sym_kl_distance(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute a symmetric KL-divergence distance matrix between probability vectors.
    """
    if pdist is not None and squareform is not None:
        def _sym_kl(u, v):
            u = np.clip(u, eps, None)
            v = np.clip(v, eps, None)
            forward = np.sum(u * (np.log(u) - np.log(v)))
            backward = np.sum(v * (np.log(v) - np.log(u)))
            return max(0.0, 0.5 * (forward + backward))

        condensed = pdist(probs, metric=_sym_kl)
        dist = squareform(condensed)
    else:
        n = probs.shape[0]
        dist = np.zeros((n, n), dtype=np.float64)
        log_probs = np.log(np.clip(probs, eps, None))
        for i in range(n):
            for j in range(i + 1, n):
                u = probs[i]
                v = probs[j]
                forward = np.sum(u * (log_probs[i] - log_probs[j]))
                backward = np.sum(v * (log_probs[j] - log_probs[i]))
                val = max(0.0, 0.5 * (forward + backward))
                dist[i, j] = dist[j, i] = val
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def _pairwise_wasserstein_distance(probs: np.ndarray) -> np.ndarray:
    """
    Compute Wasserstein-1 (EMD) distance matrix between 1D probability histograms over classes.
    """
    cdf = np.cumsum(probs, axis=1)
    num_bins = probs.shape[1]
    if pdist is not None and squareform is not None:
        condensed = pdist(cdf, metric="cityblock")
        dist = squareform(condensed) / float(num_bins)
    else:
        n = probs.shape[0]
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                val = np.sum(np.abs(cdf[i] - cdf[j])) / float(num_bins)
                dist[i, j] = dist[j, i] = val
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def _agglomerative_from_distance(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Perform agglomerative clustering on a precomputed distance matrix.
    """
    n_samples = distance_matrix.shape[0]
    if n_samples == 0:
        return np.array([], dtype=np.int32)
    if n_clusters <= 1 or n_samples == 1:
        return np.zeros(n_samples, dtype=np.int32)

    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    distance_matrix = np.maximum(distance_matrix, 0.0)
    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(distance_matrix, 0.0)

    try:
        agg = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    except TypeError:  # compatibility with older sklearn versions
        agg = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    return agg.fit_predict(distance_matrix).astype(np.int32)


def cluster_neurons_by_class_means(
    feature_matrix: torch.Tensor,
    labels,
    image_paths: Sequence[str],
    feature_indices: Sequence[int],
    out_dir: str,
    split_name: str = "train",
    algorithm: str = "kmeans",
    n_clusters: Optional[int] = None,
    random_state: int = 0,
    top_k_patches: int = 3,
    cache_root: str = "cache",
    use_absolute: bool = False,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> Dict[str, object]:
    """
    Cluster neurons using their class-wise mean activation vectors.

    Saves two artifacts under out_dir:
      - <split_name>_clusters.npz with numeric matrices
      - <split_name>_clusters.json with metadata and patch references
    """
    os.makedirs(out_dir, exist_ok=True)

    class_means, selected_features, class_names, class_ids = compute_class_activation_means(
        feature_matrix,
        labels,
        feature_indices=feature_indices,
        cache_root=cache_root,
        use_absolute=use_absolute,
    )

    feature_count = class_means.shape[0]
    if feature_count == 0:
        raise ValueError("No features provided for clustering.")

    norms = np.linalg.norm(class_means, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = class_means / norms

    algorithm = algorithm.lower()
    algorithm_params: Dict[str, object] = {"use_absolute": bool(use_absolute)}

    if algorithm in {"kmeans", "k-means"}:
        if n_clusters is None:
            if feature_count == 1:
                n_clusters = 1
            else:
                n_clusters = max(2, min(12, feature_count // 5 or 1))
        n_clusters = max(1, min(int(n_clusters), feature_count))
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_ids = kmeans.fit_predict(normalized)
        algorithm_params.update(
            {
                "algorithm": "kmeans",
                "n_clusters": int(n_clusters),
                "random_state": int(random_state),
                "inertia": float(kmeans.inertia_),
            }
        )
    elif algorithm in {"agglomerative", "ward"}:
        if n_clusters is None:
            if feature_count == 1:
                n_clusters = 1
            else:
                n_clusters = max(2, min(12, feature_count // 5 or 1))
        n_clusters = max(1, min(int(n_clusters), feature_count))
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_ids = agg.fit_predict(normalized)
        algorithm_params.update(
            {
                "algorithm": "agglomerative",
                "n_clusters": int(n_clusters),
                "linkage": getattr(agg, "linkage", "ward"),
            }
        )
    elif algorithm in {"kl", "wasserstein"}:
        if n_clusters is None:
            if feature_count == 1:
                n_clusters = 1
            else:
                n_clusters = max(2, min(12, feature_count // 5 or 1))
        n_clusters = max(1, min(int(n_clusters), feature_count))
        prob_matrix = _normalize_class_means_to_probabilities(class_means)
        if algorithm == "kl":
            distance_matrix = _pairwise_sym_kl_distance(prob_matrix)
        else:
            distance_matrix = _pairwise_wasserstein_distance(prob_matrix)
        cluster_ids = _agglomerative_from_distance(distance_matrix, n_clusters)
        algorithm_params.update(
            {
                "algorithm": algorithm,
                "n_clusters": int(n_clusters),
                "linkage": "average",
                "distance_metric": algorithm,
            }
        )
    elif algorithm == "dbscan":
        eps = float(dbscan_eps)
        min_samples = max(1, int(dbscan_min_samples))
        db = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_ids = db.fit_predict(normalized)
        algorithm_params.update(
            {
                "algorithm": "dbscan",
                "eps": eps,
                "min_samples": int(min_samples),
            }
        )
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    # PCA for 2D visualization
    if normalized.shape[0] >= 2 and normalized.shape[1] >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(normalized).astype(np.float32)
        explained = pca.explained_variance_ratio_.astype(np.float32)
        components = pca.components_.astype(np.float32)
    elif normalized.shape[0] >= 1 and normalized.shape[1] >= 1:
        coords = np.zeros((normalized.shape[0], 2), dtype=np.float32)
        coords[:, 0] = normalized[:, 0].astype(np.float32)
        explained = np.zeros(2, dtype=np.float32)
        components = np.zeros((2, normalized.shape[1]), dtype=np.float32)
    else:
        coords = np.zeros((normalized.shape[0], 2), dtype=np.float32)
        explained = np.zeros(2, dtype=np.float32)
        components = np.zeros((2, max(1, normalized.shape[1])), dtype=np.float32)

    feature_strength = norms.squeeze(-1).astype(np.float32)

    # Representative patches per feature
    top_patch_info = compute_top_patches_for_features(
        feature_matrix,
        image_paths,
        labels,
        selected_features,
        top_k=top_k_patches,
        cache_root=cache_root,
        use_absolute=use_absolute,
    )

    # Summaries per cluster
    cluster_summary: List[Dict[str, object]] = []
    unique_clusters = sorted(set(int(c) for c in cluster_ids))
    for cid in unique_clusters:
        mask = cluster_ids == cid
        size = int(np.sum(mask))
        if size == 0:
            continue
        centroid = class_means[mask].mean(axis=0) if size > 1 else class_means[mask][0]
        top_class_idx = int(np.argmax(centroid))
        summary_entry = {
            "cluster_id": int(cid),
            "size": size,
            "top_class_id": int(class_ids[top_class_idx]),
            "top_class_name": class_names[top_class_idx],
            "top_class_score": float(centroid[top_class_idx]),
        }
        cluster_summary.append(summary_entry)

    npz_path = os.path.join(out_dir, f"{split_name}_clusters.npz")
    np.savez_compressed(
        npz_path,
        feature_indices=np.array(selected_features, dtype=np.int32),
        class_ids=np.array(class_ids, dtype=np.int32),
        class_means=class_means.astype(np.float32),
        class_means_normalized=normalized.astype(np.float32),
        cluster_ids=np.array(cluster_ids, dtype=np.int32),
        pca_coords=coords,
        pca_components=components,
        pca_explained=explained,
        feature_strength=feature_strength,
        split=np.array([split_name]),
    )

    metadata = {
        "split": split_name,
        "algorithm": algorithm_params.get("algorithm", algorithm),
        "algorithm_params": algorithm_params,
        "top_patch_k": int(top_k_patches),
        "class_names": class_names,
        "class_ids": class_ids,
        "feature_indices": selected_features,
        "cluster_summary": cluster_summary,
        "representative_patches": {str(fid): info for fid, info in top_patch_info.items()},
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_features": feature_count,
        "num_classes": len(class_ids),
    }

    meta_path = os.path.join(out_dir, f"{split_name}_clusters.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    summary_line = (
        f"[cluster] {split_name} | algorithm={metadata['algorithm']} | "
        f"features={feature_count} | clusters={len(cluster_summary)}"
    )
    print(summary_line)
    if report_file:
        print(summary_line, file=report_file)

    return {
        "npz_path": npz_path,
        "metadata_path": meta_path,
        "metadata": metadata,
    }

def extract_features(model, dataloader, device="cuda", return_inputs: bool = False):
    """Extract feature activations from SAE model.

    Args:
        model: SparseAutoEncoder
        dataloader: yields (embedding, label, path)
        device: torch device
        return_inputs: if True also return stacked (possibly normalized/z-scored) input embeddings used as model inputs; useful for MS_score.

    Returns:
        feature_matrix, image_paths, labels (, input_embeddings if return_inputs)
    """
    feature_matrix = []
    image_paths = []
    labels = []
    input_embeddings = [] if return_inputs else None

    model.eval()
    with torch.no_grad():
        for emb, lab, path in dataloader:
            input_batch = emb.to(device)
            _, h = model(input_batch)
            feature_matrix.append(h.cpu())
            image_paths.extend(path)
            labels.extend(lab)
            if return_inputs:
                input_embeddings.append(emb.cpu())

    feature_matrix = torch.cat(feature_matrix, dim=0)
    if return_inputs:
        input_embeddings = torch.cat(input_embeddings, dim=0)
        # ensure row-wise unit norm (dataset may already L2-normalize, but be safe)
        norms = input_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-6)
        input_embeddings = input_embeddings / norms
        return feature_matrix, image_paths, labels, input_embeddings
    return feature_matrix, image_paths, labels


def extract_features_streaming(model, dataloader, percentile=95.0, num_features=50,
                               device="cuda", return_inputs: bool = False, eps=1e-6):
    """Two-pass streaming feature extraction for large datasets.

    Pass 1: Compute per-feature statistics (mean activation, coverage) without materializing full matrix
    Pass 2: Extract only locked features (top num_features by mean activation)

    This reduces memory from O(N × D) to O(N × k) where k << D.

    Args:
        model: SparseAutoEncoder
        dataloader: yields (embedding, label, path)
        percentile: IGNORED - kept for API compatibility (use magnitude-based selection)
        num_features: Number of top features to select (default: 50)
        device: torch device
        return_inputs: if True also return stacked input embeddings for MS_score
        eps: Threshold for coverage computation (default: 1e-6)

    Returns:
        feature_matrix_locked: [N, num_features] tensor with only selected features
        image_paths: list of image paths
        labels: list of labels
        locked_feature_indices: list of selected feature indices
        coverage_stats: dict with coverage statistics for all features
        (input_embeddings if return_inputs)
    """
    print(f"[Streaming] Pass 1: Computing per-feature statistics (magnitude-based selection)...")

    model.eval()
    hidden_dim = None
    total_samples = 0

    # Accumulators for Pass 1 - O(D) memory only!
    sum_abs = None  # Sum of absolute activations per feature
    coverage_counts = None  # Count of active samples per feature

    with torch.no_grad():
        for batch_idx, (emb, lab, path) in enumerate(tqdm(dataloader, desc="Pass 1: Statistics")):
            input_batch = emb.to(device)
            _, h = model(input_batch)
            h_cpu = h.cpu()

            batch_size = h_cpu.shape[0]
            if hidden_dim is None:
                hidden_dim = h_cpu.shape[1]
                sum_abs = torch.zeros(hidden_dim, dtype=torch.float32)
                coverage_counts = torch.zeros(hidden_dim, dtype=torch.long)

            # Accumulate mean activation (magnitude-based)
            sum_abs += h_cpu.abs().sum(dim=0)

            # Update coverage counts
            active_mask = (h_cpu > eps)
            coverage_counts += active_mask.sum(dim=0).long()

            total_samples += batch_size

    print(f"[Streaming] Selecting top {num_features} features by mean magnitude...")

    # Compute mean magnitude scores for all features
    mean_magnitude = sum_abs / max(1, total_samples)

    # Select top features by mean magnitude
    _, top_indices = torch.topk(mean_magnitude, k=num_features)
    locked_feature_indices = sorted(top_indices.tolist())

    print(f"[Streaming] Selected features: {locked_feature_indices[:10]}... (showing first 10)")
    print(f"[Streaming] Mean magnitude range: {mean_magnitude.min():.6f} to {mean_magnitude.max():.6f}")

    # Compute coverage statistics
    coverage_per_feature = coverage_counts.float() / max(1, total_samples)
    coverage_stats = {
        "num_samples": total_samples,
        "hidden_dim": hidden_dim,
        "mean_coverage": float(coverage_per_feature.mean()),
        "median_coverage": float(coverage_per_feature.median()),
        "std_coverage": float(coverage_per_feature.std()),
        "min_coverage": float(coverage_per_feature.min()),
        "max_coverage": float(coverage_per_feature.max()),
        "coverage_per_feature": coverage_per_feature.tolist()
    }

    # Free memory before Pass 2
    del sum_abs
    del coverage_counts
    del mean_magnitude

    print(f"[Streaming] Pass 2: Extracting locked features only ({num_features} features)...")

    # Pass 2: Extract only locked features
    feature_matrix_locked = []
    image_paths = []
    labels_list = []
    input_embeddings = [] if return_inputs else None

    locked_indices_tensor = torch.tensor(locked_feature_indices, dtype=torch.long)

    with torch.no_grad():
        for emb, lab, path in tqdm(dataloader, desc="Pass 2: Extraction"):
            input_batch = emb.to(device)
            _, h = model(input_batch)
            # Extract only locked columns
            h_locked = h.cpu()[:, locked_indices_tensor]
            feature_matrix_locked.append(h_locked)
            image_paths.extend(path)
            labels_list.extend(lab)
            if return_inputs:
                input_embeddings.append(emb.cpu())

    feature_matrix_locked = torch.cat(feature_matrix_locked, dim=0)

    print(f"[Streaming] Extracted feature matrix shape: {feature_matrix_locked.shape}")

    if return_inputs:
        input_embeddings = torch.cat(input_embeddings, dim=0)
        norms = input_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-6)
        input_embeddings = input_embeddings / norms
        return feature_matrix_locked, image_paths, labels_list, locked_feature_indices, coverage_stats, input_embeddings

    return feature_matrix_locked, image_paths, labels_list, locked_feature_indices, coverage_stats

def save_feature_matrices(feature_matrix, image_paths, labels, split_name, model_dir):
    """Save feature matrices, paths, and labels to disk for caching within model directory"""
    cache_dir = os.path.join(model_dir, "cache", "feature-matrix", split_name)
    os.makedirs(cache_dir, exist_ok=True)

    torch.save(feature_matrix, os.path.join(cache_dir, "feature_matrix.pt"))

    with open(os.path.join(cache_dir, "image_paths.json"), 'w') as f:
        json.dump(image_paths, f)

    torch.save(labels, os.path.join(cache_dir, "labels.pt"))
    print(f"Saved feature matrix cache for {split_name} split in {cache_dir}")

def load_feature_matrices(split_name, model_dir):
    """Load cached feature matrices, paths, and labels from disk within model directory"""
    cache_dir = os.path.join(model_dir, "cache", "feature-matrix", split_name)

    required_files = ["feature_matrix.pt", "image_paths.json", "labels.pt"]
    if not all(os.path.exists(os.path.join(cache_dir, f)) for f in required_files):
        return None, None, None

    feature_matrix = torch.load(os.path.join(cache_dir, "feature_matrix.pt"))

    with open(os.path.join(cache_dir, "image_paths.json"), 'r') as f:
        image_paths = json.load(f)

    labels = torch.load(os.path.join(cache_dir, "labels.pt"))
    print(f"Loaded cached feature matrix for {split_name} split from {cache_dir}")

    return feature_matrix, image_paths, labels

def load_model_from_dir(model_dir):
    """Load SAE model and metadata from model directory"""
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Extract model parameters
    hp = metadata["hyperparameters"]
    input_dim = hp["input_dim"]
    hidden_dim = hp["hidden_dim"]
    tie_weights = hp.get("tie_weights", False)  # Default to False for backward compatibility
    use_pre_bias = hp.get("use_pre_bias", False)  # Default to False for backward compatibility
    activation = hp.get("activation", "relu")
    topk_k = hp.get("topk_k")
    if activation == "topk" and (topk_k is None or topk_k <= 0):
        print(f"Warning: metadata for {model_dir} missing valid topk_k; defaulting to hidden_dim ({hidden_dim})")
        topk_k = hidden_dim

    # Load model
    model = SparseAutoEncoder(input_dim, hidden_dim, tie_weights=tie_weights, use_pre_bias=use_pre_bias,
                              activation=activation, topk_k=topk_k)
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path)

    # Handle DataParallel models: strip 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()

    # Extract model name from directory, removing any trailing slashes
    model_name = os.path.basename(model_dir.rstrip("/"))

    return model, metadata, model_name

def create_datasets_from_metadata(metadata):
    """Create datasets using metadata configuration

    Returns (train_dataset, test_dataset). The function attempts to resolve
    common split folder names (train/test/validation) from the provided
    emb_dir. For datasets without a distinct test split, the test dataset
    may point to the same directory as train (caller should handle that case).
    """
    dataset_args = metadata["dataset_args"]
    dataset_type = metadata["dataset_type"]

    emb_dir = dataset_args.get("emb_dir", "cache/train")
    zscore = dataset_args.get("zscore", True)
    l2_normalize = dataset_args.get("l2_normalize", True)

    def resolve_split_dir(root_dir, preferred_splits=("train", "validation", "test")):
        # If root_dir already exists, prefer it. Otherwise try root_dir/{split}
        if os.path.isdir(root_dir):
            return root_dir
        parent = os.path.dirname(root_dir)
        base = os.path.basename(root_dir)
        for s in preferred_splits:
            candidate = os.path.join(parent, base + ("" if base.endswith(s) else f"_{s}"))
            if os.path.isdir(candidate):
                return candidate
            # Try just the split name directly under parent (new naming: cache-kather100k/train)
            candidate2 = os.path.join(parent, s)
            if os.path.isdir(candidate2):
                return candidate2
        # fallback to root_dir even if missing
        return root_dir

    # Resolve train/test paths
    train_path = resolve_split_dir(emb_dir, preferred_splits=("train", "train_nonorm", "validation"))
    # For test, prefer a validation or test folder sibling
    test_candidate = emb_dir.replace("train", "test") if "train" in emb_dir else emb_dir
    test_path = resolve_split_dir(test_candidate, preferred_splits=("validation", "test", "train"))

    if dataset_type == 'ignite':
        TrainClass = IgniteEmbeddingDataset
        TestClass = IgniteEmbeddingDataset
    elif dataset_type == 'kather100k':
        TrainClass = Kather100kEmbeddingDataset
        TestClass = Kather100kEmbeddingDataset
    else:
        TrainClass = SpiderEmbeddingDataset
        TestClass = SpiderEmbeddingDataset

    train_stats_path = os.path.join(train_path, "emb_stats.pt")
    train_dataset = TrainClass(
        emb_dir=train_path,
        include_paths=True,
        zscore=zscore,
        l2_normalize=l2_normalize,
        stats_path=train_stats_path,
    )

    if zscore and not os.path.exists(train_stats_path):
        raise FileNotFoundError(
            f"Training normalization stats were not created at {train_stats_path}"
        )

    # Evaluation must reuse the training-split normalization transform rather than
    # reading split-local test statistics from test_path/emb_stats.pt.
    test_dataset = TestClass(
        emb_dir=test_path,
        include_paths=True,
        zscore=zscore,
        l2_normalize=l2_normalize,
        stats_path=train_stats_path if zscore else None,
    )

    return train_dataset, test_dataset

def select_top_features_from_train(feature_matrix_train, selection_k=50, num_features=50):
    """Select top features based on TRAIN data only - this roster is locked"""
    top_k_activations, _ = torch.topk(feature_matrix_train, k=selection_k, dim=0)
    score_per_feature = top_k_activations.sum(dim=0)
    _, top_feature_indices = torch.topk(score_per_feature, k=num_features)

    return top_feature_indices.tolist(), score_per_feature

def select_top_features_percentile(feature_matrix_train, percentile=95, num_features=50):
    """Select top features using percentile-based scoring (alternative to topk)"""
    score_per_feature = torch.quantile(feature_matrix_train, percentile/100, dim=0)
    _, top_feature_indices = torch.topk(score_per_feature, k=num_features)

    print(f"Selected features based on {percentile}th percentile activations")
    print(f"Score range: {score_per_feature.min():.3f} to {score_per_feature.max():.3f}")

    return top_feature_indices.tolist(), score_per_feature

def analyze_locked_features(feature_matrix, labels, locked_feature_indices, top_k=25, debug=False, cache_root="cache", already_locked=False):
    """Analyze purity for locked feature set (no reranking).

    Args:
        feature_matrix: Feature activations, either full [N, D] or locked [N, k]
        labels: Sample labels
        locked_feature_indices: List of feature indices (global indices if not already_locked)
        top_k: Number of top activations to analyze
        debug: Include detailed debugging info
        cache_root: Path to label map cache
        already_locked: If True, feature_matrix is already filtered to locked columns only
                       and locked_feature_indices are just ordinal positions [0, 1, 2, ...]

    If debug=True, include detailed index/label lists for purity@100 and top_k visualization subset
    to aid in diagnosing discrepancies.
    """
    label2desc = load_label_map(cache_root)

    feature_analysis = {}
    for local_idx, feature_idx in enumerate(locked_feature_indices):
        # In streaming mode, feature_matrix is already locked, so use local_idx
        col_idx = local_idx if already_locked else feature_idx
        feature_activations = feature_matrix[:, col_idx]

        # Use absolute values for baseline features (UNI embeddings can be negative)
        # For SAE features, this should have no effect since they're ReLU-activated
        abs_activations = torch.abs(feature_activations)

        purity_k = min(100, feature_activations.shape[0])
        purity_values, purity_indices = torch.topk(abs_activations, purity_k)
        purity_indices_list = purity_indices.tolist()
        purity_labels = [int(labels[i]) for i in purity_indices_list]
        purity_maj_label, purity_maj_count = Counter(purity_labels).most_common(1)[0]
        purity = purity_maj_count / purity_k if purity_k > 0 else 0.0

        values, indices = torch.topk(abs_activations, top_k)
        indices_list = indices.tolist()
        top_labels = [int(labels[i]) for i in indices_list]
        maj_label, maj_count = Counter(top_labels).most_common(1)[0]

        # Store the original (signed) activation values for the selected indices
        original_values = feature_activations[indices]

        entry = {
            'majority_label': purity_maj_label,
            'majority_label_name': label2desc[purity_maj_label],
            'purity': purity,
            'purity_k': purity_k,
            'top_activations': original_values.tolist(),  # Keep original signed values for display
            'label_distribution': dict(Counter(top_labels)),
            'max_activation': float(feature_activations.max()),
            'mean_activation': float(feature_activations.mean())
        }
        if debug:
            # Store original signed values for purity samples too
            original_purity_values = feature_activations[purity_indices]
            entry.update({
                'purity_sample_indices': purity_indices_list,
                'purity_sample_labels': purity_labels,
                'top_sample_indices': indices_list,
                'top_sample_labels': top_labels,
                'viz_majority_label': maj_label,
                'viz_majority_label_name': label2desc[maj_label],
                'viz_majority_label_fraction_topk': maj_count / len(top_labels) if top_labels else 0.0,
                'purity_sample_values': original_purity_values.tolist()
            })
        feature_analysis[feature_idx] = entry
    return feature_analysis

def calculate_grid_size(num_images):
    """Calculate optimal grid size for given number of images"""
    sqrt_n = math.sqrt(num_images)
    rows = math.ceil(sqrt_n)
    cols = math.ceil(num_images / rows)
    return rows, cols

def save_feature_visualizations(
    feature_matrix,
    image_paths,
    labels,
    locked_feature_indices,
    split_name,
    feature_analysis,
    base_dir="top-k-features-v0.2",
    top_k=25,
    grid_images=100,
    cache_root="cache",
):
    """Save visualization plots for locked features"""
    label2desc = load_label_map(cache_root)

    split_dir = f"{base_dir}/{split_name}"

    # skip if split_dir exists and is not empty
    if os.path.exists(split_dir) and os.listdir(split_dir):
        print(f"Skipping {split_dir}, already exists and is not empty.")
        return

    # Use pre-calculated purities
    all_purities = [feature_analysis[idx]['purity'] for idx in locked_feature_indices]

    # Progress bar
    pbar = tqdm(locked_feature_indices, desc=f"viz:{split_name}", unit="feat")
    for feature_index in pbar:
        pbar.set_postfix({"purity": f"{feature_analysis[feature_index]['purity']:.3f}"})
        feature_activations = feature_matrix[:, feature_index]

        # Use pre-calculated values from analysis
        analysis_info = feature_analysis[feature_index]
        purity = analysis_info['purity']
        purity_k = analysis_info['purity_k']
        maj_label = analysis_info['majority_label']
        maj_name = analysis_info['majority_label_name']

        # Get top_k samples for visualization
        viz_k = min(max(top_k, grid_images), feature_activations.shape[0])
        values, indices = torch.topk(feature_activations, viz_k)
        top_labels = [int(labels[i]) for i in indices.tolist()]

        # Calculate grid dimensions: default to 10 columns for denser layout
        cols = 10
        rows = math.ceil(grid_images / cols)

        # Avoid Matplotlib backend auto-close warnings / resource leaks
        plt.close('all')
        fig_width = 18
        fig_height = max(6, rows * 2.4)
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        # Show only the requested number of images
        display_indices = indices[:grid_images]
        display_values = values[:grid_images]

        for i, (ax, idx, val) in enumerate(zip(axes[:grid_images], display_indices, display_values)):
            img_path = image_paths[idx]
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            img = img.resize((160, 160), Image.BILINEAR)
            ax.imshow(img)

            label = int(labels[idx])
            label_text = label2desc[label]

            # Color minority labels in red
            if label != maj_label:
                ax.set_title(f"Value: {val:.2f}\nLabel: {label_text}", color='red', fontsize=8)
            else:
                ax.set_title(f"Value: {val:.2f}\nLabel: {label_text}", fontsize=8)
            ax.axis("off")

        # Hide unused subplots
        for i in range(grid_images, len(axes)):
            axes[i].axis("off")

        os.makedirs(f"{split_dir}/{maj_name}", exist_ok=True)

        # Feature number as main title
        plt.suptitle(f"Feature {feature_index}", fontsize=16, fontweight='bold')

        # Other info as subtitle at bottom
        plt.figtext(0.5, 0.94, f"Purity@{purity_k}={purity:.2f} | Majority={maj_name} | Split={split_name}",
                   ha='center', fontsize=10)

        plt.savefig(f"{split_dir}/{maj_name}/feature_{feature_index}.png", bbox_inches="tight", dpi=180)
        plt.close(fig)
    pbar.close()

def save_flipper_visualizations_fast(
    feature_matrix_train, image_paths_train, labels_train,
    feature_matrix_test,  image_paths_test,  labels_test,
    flipper_features, base_dir="top-k-features-v0.2",
    top_k=25, grid_images=25, png=True, cache_root="cache"
):
    """
    Fast flipper viz:
      - Precompute top-k for all features (train & test)
      - Render 'matplotlib-style' grids with two-line titles ABOVE each tile
      - Save train/test grids + combined side-by-side image
    Layout preserved: 5 columns (or your calculate_grid_size), titles per tile, minority labels in red.
    """

    # --- label mapping ---
    label2desc = load_label_map(cache_root)
    max_label = max(label2desc.keys())
    label2desc_list = [label2desc.get(i, f"Label_{i}") for i in range(max_label + 1)]
    reverse_label_map = {v: i for i, v in enumerate(label2desc_list)}

    # --- to numpy for fast indexing ---
    labels_train_np = labels_train.cpu().numpy() if isinstance(labels_train, torch.Tensor) else np.asarray(labels_train)
    labels_test_np  = labels_test.cpu().numpy()  if isinstance(labels_test,  torch.Tensor) else np.asarray(labels_test)

    # --- output dir ---
    flipper_dir = Path(base_dir) / "flippers"
    flipper_dir.mkdir(parents=True, exist_ok=True)

    # --- grid geometry (keep identical look) ---
    # Try user's function if defined; else default to 5 columns.
    try:
        rows, cols = calculate_grid_size(grid_images)  # noqa: F821 (may be user-defined)
    except NameError:
        cols = 5
        rows = math.ceil(grid_images / cols)

    # per-tile image size & spacing (tuned to match your old look)
    tile_w, tile_h = 224, 224
    pad = 10
    line_h = 12         # ~fontsize=8
    title_h = 2*line_h + 6  # "Value: ..." + "Label: ..." above each tile

    grid_w = cols * tile_w + (cols - 1) * pad
    grid_h = rows * (title_h + tile_h) + (rows - 1) * pad

    # combined canvas margins & headers
    OUT_M_L, OUT_M_T, OUT_M_R, OUT_M_B = 30, 30, 30, 50
    panel_gap = 60
    sup_h = 28          # "Feature X - FLIPPER"
    panel_title_h = 24  # "TRAIN | ...", "TEST | ..."

    # total combined dims
    combined_w = OUT_M_L + grid_w + panel_gap + grid_w + OUT_M_R
    combined_h = OUT_M_T + sup_h + panel_title_h + grid_h + OUT_M_B

    # --- fonts ---
    try:
        font = ImageFont.load_default()
        bold = font
    except Exception:
        font = bold = None

    # --- batched top-k for train & test ---
    K_train = min(top_k, feature_matrix_train.size(0))
    K_test  = min(top_k,  feature_matrix_test.size(0))
    train_vals_all, train_idx_all = torch.topk(feature_matrix_train, k=K_train, dim=0, largest=True, sorted=True)
    test_vals_all,  test_idx_all  = torch.topk(feature_matrix_test,  k=K_test,  dim=0, largest=True, sorted=True)
    train_vals_all = train_vals_all.cpu()
    train_idx_all  = train_idx_all.cpu()
    test_vals_all  = test_vals_all.cpu()
    test_idx_all   = test_idx_all.cpu()

    # --- thumbnail caches (separate for train/test) ---
    @lru_cache(maxsize=4096)
    def load_thumb_train(sample_idx: int) -> Image.Image:
        p = image_paths_train[sample_idx]
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return im.resize((tile_w, tile_h), Image.BILINEAR)

    @lru_cache(maxsize=4096)
    def load_thumb_test(sample_idx: int) -> Image.Image:
        p = image_paths_test[sample_idx]
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return im.resize((tile_w, tile_h), Image.BILINEAR)

    def render_panel_grid(idxs, vals, labels_np, maj_label_int, is_train=True):
        """
        Render a single grid (no outer margins, no panel title).
        Two-line title ABOVE each tile, minority labels in red, thin border around tile.
        Returns PIL.Image (size grid_w x grid_h).
        """
        n_disp = min(grid_images, len(idxs))
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(canvas)

        for i in range(n_disp):
            r = i // cols
            c = i % cols
            x0 = c * (tile_w + pad)
            y0 = r * (tile_h + title_h + pad)

            idx_i = int(idxs[i])
            val_i = float(vals[i])
            lab_i = int(labels_np[idx_i])
            lab_name = label2desc_list[lab_i]

            # two-line "matplotlib" style captions above tile
            t1 = f"Value: {val_i:.2f}"
            t2 = f"{lab_name}"
            color = (192, 0, 0) if lab_i != maj_label_int else (0, 0, 0)
            draw.text((x0, y0), t1, fill=color, font=font if font else None)
            draw.text((x0, y0 + line_h + 2), t2, fill=color, font=font if font else None)

            # paste image below captions
            img_y = y0 + title_h
            img = load_thumb_train(idx_i) if is_train else load_thumb_test(idx_i)
            canvas.paste(img, (x0, img_y))
            # thin border around tile area
            draw.rectangle([x0, img_y, x0 + tile_w - 1, img_y + tile_h - 1], outline=(0,0,0), width=1)

        # hide unused tiles (no-op visually, grid is white)
        return canvas

    # ---- main loop ----
    for feature_info in tqdm(flipper_features, desc="flipper-viz", unit="flipper"):
        feat_idx = feature_info['feature_index']

        # majority labels & purities
        train_maj_name = feature_info['train_majority']
        test_maj_name  = feature_info['test_majority']
        train_maj_lab  = reverse_label_map[train_maj_name]
        test_maj_lab   = reverse_label_map[test_maj_name]
        train_purity   = feature_info.get('train_purity', feature_info.get('purity', 0.0))
        test_purity    = feature_info.get('test_purity',  feature_info.get('purity', 0.0))

        # indexes/values for this feature (precomputed)
        tK = min(grid_images, K_train)
        sK = min(grid_images, K_test)
        train_idxs = train_idx_all[:tK, feat_idx].numpy()
        train_vals = train_vals_all[:tK, feat_idx].numpy()
        test_idxs  = test_idx_all[:sK,  feat_idx].numpy()
        test_vals  = test_vals_all[:sK,  feat_idx].numpy()

        # render panels
        train_grid = render_panel_grid(train_idxs, train_vals, labels_train_np, train_maj_lab, is_train=True)
        test_grid  = render_panel_grid(test_idxs,  test_vals,  labels_test_np,  test_maj_lab,  is_train=False)

        # save standalone train/test grids (keeps your previous outputs)
        out_train = flipper_dir / f"feature_{feat_idx}_train.{ 'png' if png else 'jpg' }"
        out_test  = flipper_dir / f"feature_{feat_idx}_test.{  'png' if png else 'jpg' }"
        if png:
            train_grid.save(out_train, format="PNG", compress_level=1)
            test_grid.save(out_test,  format="PNG", compress_level=1)
        else:
            train_grid.save(out_train, format="JPEG", quality=85, optimize=True)
            test_grid.save(out_test,  format="JPEG", quality=85, optimize=True)

        # build combined side-by-side (with suptitle + panel titles)
        combined = Image.new("RGB", (combined_w, combined_h), "white")
        draw = ImageDraw.Draw(combined)

        # Draw vertical separator line between panels
        sep_x = OUT_M_L + grid_w + panel_gap // 2
        line_top = OUT_M_T + sup_h + panel_title_h
        line_bottom = line_top + grid_h
        draw.line([(sep_x, line_top), (sep_x, line_bottom)], fill=(0, 0, 0), width=2)

        # suptitle - bigger and bold
        sup_text = f"Feature {feat_idx} - FLIPPER"
        # Use a larger font if available, otherwise fallback to default
        try:
            title_font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            try:
                title_font = ImageFont.load_default().font_variant(size=20)
            except:
                title_font = bold if bold else font

        # Draw the suptitle with bold styling
        sup_x = OUT_M_L + (combined_w - OUT_M_L - OUT_M_R) // 2 - (len(sup_text) * 6)  # adjust centering for larger text
        # Try to create a bold effect by drawing the text multiple times with slight offsets
        for dx in [0, 1]:
            for dy in [0, 1]:
                draw.text((max(OUT_M_L, sup_x) + dx, OUT_M_T + dy), sup_text, fill=(0,0,0), font=title_font)

        # panel titles
        left_title  = f"TRAIN | {train_maj_name} | Purity@100: {train_purity:.2f}"
        right_title = f"TEST  | {test_maj_name}  | Purity@100: {test_purity:.2f}"
        draw.text((OUT_M_L, OUT_M_T + sup_h + 10), left_title,  fill=(0,0,0), font=title_font)
        right_x = OUT_M_L + grid_w + panel_gap
        draw.text((right_x,  OUT_M_T + sup_h + 10), right_title, fill=(0,0,0), font=title_font)

        # paste panels
        panel_y = OUT_M_T + sup_h + panel_title_h + 30
        combined.paste(train_grid, (OUT_M_L, panel_y))
        combined.paste(test_grid,  (right_x,  panel_y))

        out_combined = flipper_dir / f"feature_{feat_idx}_combined.{ 'png' if png else 'jpg' }"
        if png:
            combined.save(out_combined, format="PNG", compress_level=1)
        else:
            combined.save(out_combined, format="JPEG", quality=85, optimize=True)

    # clear caches if you’ll call again
    load_thumb_train.cache_clear()
    load_thumb_test.cache_clear()


def save_flipper_visualizations(feature_matrix_train, image_paths_train, labels_train,
                               feature_matrix_test, image_paths_test, labels_test,
                               flipper_features, base_dir="top-k-features-v0.2", top_k=25, grid_images=25, cache_root="cache"):
    """Save side-by-side visualizations for flipper features"""
    label2desc = load_label_map(cache_root)

    flipper_dir = f"{base_dir}/flippers"
    os.makedirs(flipper_dir, exist_ok=True)

    pbar = tqdm(flipper_features, desc="flipper-viz", unit="flipper")
    for feature_info in pbar:
        feature_index = feature_info['feature_index']
        pbar.set_postfix({"feat": feature_index})

        # Get train data
        train_activations = feature_matrix_train[:, feature_index]
        train_values, train_indices = torch.topk(train_activations, top_k)
        train_labels = [int(labels_train[i]) for i in train_indices.tolist()]

        # Get test data
        test_activations = feature_matrix_test[:, feature_index]
        test_values, test_indices = torch.topk(test_activations, top_k)
        test_labels = [int(labels_test[i]) for i in test_indices.tolist()]

        # Use pre-calculated majority labels from feature_info (calculated from top 100)
        train_maj_name = feature_info['train_majority']
        test_maj_name = feature_info['test_majority']

        # Convert string labels back to integer codes for highlighting logic
        reverse_label_map = {v: k for k, v in label2desc.items()}
        train_maj_label = reverse_label_map[train_maj_name]
        test_maj_label = reverse_label_map[test_maj_name]

        train_purity = feature_info.get('train_purity', feature_info.get('purity', 0.0))
        test_purity = feature_info.get('test_purity', feature_info.get('purity', 0.0))

        # Calculate grid size
        rows, cols = calculate_grid_size(grid_images)

        # Create side-by-side figure
        # Close previous figures to avoid backend warnings
        plt.close('all')
        fig_width = cols * 6  # Double width for side-by-side
        fig_height = rows * 3.5

        fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        # Train subplot
        plt.close('all')
        train_fig, train_axes = plt.subplots(rows, cols, figsize=(fig_width//2, fig_height))
        if grid_images == 1:
            train_axes = [train_axes]
        else:
            train_axes = train_axes.flatten()

        display_train_indices = train_indices[:grid_images]
        display_train_values = train_values[:grid_images]

        for i, (ax, idx, val) in enumerate(zip(train_axes[:grid_images], display_train_indices, display_train_values)):
            img_path = image_paths_train[idx]
            img = Image.open(img_path)
            ax.imshow(img)

            label = int(labels_train[idx])
            label_text = label2desc[label]

            if label != train_maj_label:
                ax.set_title(f"Value: {val:.2f}\n{label_text}", color='red', fontsize=8)
            else:
                ax.set_title(f"Value: {val:.2f}\n{label_text}", fontsize=8)
            ax.axis("off")

        # Hide unused train subplots
        for i in range(grid_images, len(train_axes)):
            train_axes[i].axis("off")

        plt.tight_layout()
        train_fig.savefig(f"{flipper_dir}/feature_{feature_index}_train.png", bbox_inches="tight", dpi=180)
        plt.close(train_fig)

        # Test subplot
        plt.close('all')
        test_fig, test_axes = plt.subplots(rows, cols, figsize=(fig_width//2, fig_height))
        if grid_images == 1:
            test_axes = [test_axes]
        else:
            test_axes = test_axes.flatten()

        display_test_indices = test_indices[:grid_images]
        display_test_values = test_values[:grid_images]

        for i, (ax, idx, val) in enumerate(zip(test_axes[:grid_images], display_test_indices, display_test_values)):
            img_path = image_paths_test[idx]
            img = Image.open(img_path)
            ax.imshow(img)

            label = int(labels_test[idx])
            label_text = label2desc[label]

            if label != test_maj_label:
                ax.set_title(f"Value: {val:.2f}\n{label_text}", color='red', fontsize=8)
            else:
                ax.set_title(f"Value: {val:.2f}\n{label_text}", fontsize=8)
            ax.axis("off")

        # Hide unused test subplots
        for i in range(grid_images, len(test_axes)):
            test_axes[i].axis("off")

        plt.tight_layout()
        test_fig.savefig(f"{flipper_dir}/feature_{feature_index}_test.png", bbox_inches="tight", dpi=180)
        plt.close(test_fig)

        # Create combined side-by-side visualization
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Load and display the saved images
        train_img = Image.open(f"{flipper_dir}/feature_{feature_index}_train.png")
        test_img = Image.open(f"{flipper_dir}/feature_{feature_index}_test.png")

        ax1.imshow(train_img)
        ax1.set_title(f"TRAIN | {label2desc[train_maj_label]} | Purity@100: {train_purity:.2f}", fontsize=14)
        ax1.axis("off")

        ax2.imshow(test_img)
        ax2.set_title(f"TEST | {label2desc[test_maj_label]} | Purity@100: {test_purity:.2f}", fontsize=14)
        ax2.axis("off")

        plt.suptitle(f"Feature {feature_index} - FLIPPER", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{flipper_dir}/feature_{feature_index}_combined.png", bbox_inches="tight", dpi=180)
        plt.close(fig)
    pbar.close()

def robust_mean(x, trim=0.05):
    """
    Trimmed mean to reduce influence of outliers.
    x: 1D numpy array
    trim: fraction to trim from each tail (0.05 = 5%)
    """
    if x.size == 0:
        return 0.0
    if trim <= 0.0:
        return float(x.mean())
    n = x.size
    k = int(n * trim)
    if k == 0 or 2 * k >= n:
        return float(x.mean())
    xs = np.sort(x)
    return float(xs[k: n - k].mean())

def calculate_monosemanticity_scores(feature_matrix_train, labels_train,
                                   feature_matrix_test, labels_test,
                                   locked_feature_indices, eps=1e-6, delta=1e-6, robust_trim=0.05, cache_root="cache",
                                   already_locked=False):
    """Calculate per-feature monosemanticity scores based on class-activation margin

    Args:
        already_locked: If True, matrices are already filtered to locked columns
    """
    label2desc = load_label_map(cache_root)

    labels_train = torch.tensor(labels_train) if not isinstance(labels_train, torch.Tensor) else labels_train
    labels_test = torch.tensor(labels_test) if not isinstance(labels_test, torch.Tensor) else labels_test

    mono_scores = {}

    for local_idx, feature_idx in enumerate(locked_feature_indices):
        # Extract feature activations - use local_idx if already locked
        col_idx = local_idx if already_locked else feature_idx
        train_activations = feature_matrix_train[:, col_idx]
        test_activations = feature_matrix_test[:, col_idx]

        # --- robust class means (train) ---
        mu_train = torch.zeros(13)
        for class_idx in range(13):
            class_mask = (labels_train == class_idx)
            if class_mask.any():
                class_vals = train_activations[class_mask]
                active_vals = class_vals[class_vals > eps]
                if active_vals.numel() > 0:
                    mu_train[class_idx] = robust_mean(active_vals.detach().cpu().numpy(), trim=robust_trim)

        # --- robust class means (test) ---
        mu_test = torch.zeros(13)
        for class_idx in range(13):
            class_mask = (labels_test == class_idx)
            if class_mask.any():
                class_vals = test_activations[class_mask]
                active_vals = class_vals[class_vals > eps]
                if active_vals.numel() > 0:
                    mu_test[class_idx] = robust_mean(active_vals.detach().cpu().numpy(), trim=robust_trim)

        # Normalize to get probability distributions (original method)
        sum_train = mu_train.sum() + delta
        sum_test = mu_test.sum() + delta
        p_train = mu_train / sum_train
        p_test = mu_test / sum_test

        # Softmax-based probability distributions (new method)
        p_train_softmax = torch.softmax(mu_train, dim=0)
        p_test_softmax = torch.softmax(mu_test, dim=0)

        # Lock dominant class on train (using original normalization)
        c_star_train = torch.argmax(p_train).item()
        c_star_test = torch.argmax(p_test).item()
        flipped = c_star_train != c_star_test

        # Lock dominant class on train (using softmax)
        c_star_train_softmax = torch.argmax(p_train_softmax).item()
        c_star_test_softmax = torch.argmax(p_test_softmax).item()
        flipped_softmax = c_star_train_softmax != c_star_test_softmax

        # Calculate margins (train-locked)
        def second_best(p, k):
            """Get second highest probability excluding class k"""
            p_copy = p.clone()
            p_copy[k] = -float('inf')  # Exclude the dominant class
            if torch.all(p_copy == -float('inf')):
                return 0.0  # Only one class has non-zero probability
            return torch.max(p_copy).item()

        # Handle edge cases (original method)
        if sum_train <= delta:  # No active samples in train
            m_train = 0.0
        else:
            m_train = p_train[c_star_train].item() - second_best(p_train, c_star_train)

        if sum_test <= delta:  # No active samples in test
            m_test = 0.0
        else:
            m_test = p_test[c_star_train].item() - second_best(p_test, c_star_train)

        # Softmax-based margins
        m_train_softmax = p_train_softmax[c_star_train_softmax].item() - second_best(p_train_softmax, c_star_train_softmax)
        m_test_softmax = p_test_softmax[c_star_train_softmax].item() - second_best(p_test_softmax, c_star_train_softmax)

        # Final monosemanticity scores (clamp negatives to 0)
        M = max(0.0, min(m_train, m_test))
        M_softmax = max(0.0, min(m_train_softmax, m_test_softmax))

        mono_scores[feature_idx] = {
            'M': M,  # Final monosemanticity score [0,1] (normalized)
            'M_softmax': M_softmax,  # Final monosemanticity score [0,1] (softmax)
            'm_train': m_train,  # Raw train margin [-1,1] (normalized)
            'm_test': m_test,   # Raw test margin [-1,1] (normalized)
            'm_train_softmax': m_train_softmax,  # Raw train margin [-1,1] (softmax)
            'm_test_softmax': m_test_softmax,   # Raw test margin [-1,1] (softmax)
            'c_star_train': c_star_train,
            'c_star_test': c_star_test,
            'c_star_train_softmax': c_star_train_softmax,
            'c_star_test_softmax': c_star_test_softmax,
            'c_star_train_name': label2desc[c_star_train],
            'c_star_test_name': label2desc[c_star_test],
            'c_star_train_name_softmax': label2desc[c_star_train_softmax],
            'c_star_test_name_softmax': label2desc[c_star_test_softmax],
            'flipped': flipped,
            'flipped_softmax': flipped_softmax,
            'sum_train_activations': sum_train - delta,  # For diagnostics
            'sum_test_activations': sum_test - delta
        }

    return mono_scores

def calculate_monosemanticity_summary_stats(mono_scores):
    """Calculate summary statistics for monosemanticity scores"""
    if not mono_scores:
        return {}

    M_values = [score['M'] for score in mono_scores.values()]
    M_softmax_values = [score['M_softmax'] for score in mono_scores.values()]
    m_train_values = [score['m_train'] for score in mono_scores.values()]
    m_test_values = [score['m_test'] for score in mono_scores.values()]
    m_train_softmax_values = [score['m_train_softmax'] for score in mono_scores.values()]
    m_test_softmax_values = [score['m_test_softmax'] for score in mono_scores.values()]

    # Count failed cases
    failed_train = sum(1 for score in mono_scores.values() if score['sum_train_activations'] <= 0)
    failed_test = sum(1 for score in mono_scores.values() if score['sum_test_activations'] <= 0)
    flipped_count = sum(1 for score in mono_scores.values() if score['flipped'])
    flipped_softmax_count = sum(1 for score in mono_scores.values() if score['flipped_softmax'])

    # High monosemanticity features (M > 0.8)
    high_mono_count = sum(1 for m in M_values if m > 0.8)
    high_mono_softmax_count = sum(1 for m in M_softmax_values if m > 0.8)

    summary = {
        'total_features': len(mono_scores),
        # Normalized method statistics
        'mean_monosemanticity': float(np.mean(M_values)),
        'std_monosemanticity': float(np.std(M_values)),
        'median_monosemanticity': float(np.median(M_values)),
        'min_monosemanticity': float(np.min(M_values)),
        'max_monosemanticity': float(np.max(M_values)),
        'percentile_25': float(np.percentile(M_values, 25)),
        'percentile_75': float(np.percentile(M_values, 75)),
        'high_monosemanticity_count': high_mono_count,
        'high_monosemanticity_percentage': (high_mono_count / len(M_values)) * 100,
        'mean_train_margin': float(np.mean(m_train_values)),
        'mean_test_margin': float(np.mean(m_test_values)),
        'flipped_dominant_class_count': flipped_count,
        'flipped_dominant_class_percentage': (flipped_count / len(mono_scores)) * 100,
        # Softmax method statistics
        'mean_monosemanticity_softmax': float(np.mean(M_softmax_values)),
        'std_monosemanticity_softmax': float(np.std(M_softmax_values)),
        'median_monosemanticity_softmax': float(np.median(M_softmax_values)),
        'min_monosemanticity_softmax': float(np.min(M_softmax_values)),
        'max_monosemanticity_softmax': float(np.max(M_softmax_values)),
        'percentile_25_softmax': float(np.percentile(M_softmax_values, 25)),
        'percentile_75_softmax': float(np.percentile(M_softmax_values, 75)),
        'high_monosemanticity_count_softmax': high_mono_softmax_count,
        'high_monosemanticity_percentage_softmax': (high_mono_softmax_count / len(M_softmax_values)) * 100,
        'mean_train_margin_softmax': float(np.mean(m_train_softmax_values)),
        'mean_test_margin_softmax': float(np.mean(m_test_softmax_values)),
        'flipped_dominant_class_count_softmax': flipped_softmax_count,
        'flipped_dominant_class_percentage_softmax': (flipped_softmax_count / len(mono_scores)) * 100,
        # Common statistics
        'failed_train_features': failed_train,
        'failed_test_features': failed_test
    }

    return summary


def compare_train_test_locked(train_analysis, test_analysis, locked_features,
                            feature_matrix_train, labels_train, feature_matrix_test, labels_test,
                            output_file="locked_feature_comparison.json",
                            cache_root: str = "cache",
                            eps: float = 1e-6,
                            recall_percentile: float = 95.0,
                            selectivity_threshold: float = 0.9,
                            selectivity_gap: float = 0.2,
                            nearly_mono_threshold: float = 0.65,
                            already_locked: bool = False):
    """Compare locked features between train and test splits with monosemanticity, coverage & heuristic labels.

    Args:
        already_locked: If True, feature_matrix_train/test are already filtered to locked columns
    """
    comparison = {
        'locked_feature_count': len(locked_features),
        'feature_comparisons': [],
        'summary_stats': {},
        'flippers': []
    }

    if recall_percentile < 0.0 or recall_percentile > 100.0:
        raise ValueError("recall_percentile must be within [0, 100]")
    if selectivity_threshold < 0.0 or selectivity_threshold > 1.0:
        raise ValueError("selectivity_threshold must be within [0, 1]")
    if selectivity_gap < 0.0 or selectivity_gap > 1.0:
        raise ValueError("selectivity_gap must be within [0, 1]")
    if nearly_mono_threshold < 0.0 or nearly_mono_threshold > 1.0:
        raise ValueError("nearly_mono_threshold must be within [0, 1]")

    label2desc = load_label_map(cache_root)

    rprint("Calculating monosemanticity scores...")
    mono_scores = calculate_monosemanticity_scores(
        feature_matrix_train, labels_train,
        feature_matrix_test, labels_test,
        locked_features,
        cache_root=cache_root,
        already_locked=already_locked
    )

    # --- recall thresholds and per-class recall (train/test) ---
    if locked_features:
        # Ensure tensors for labels to support device operations.
        labels_train_tensor = labels_train if isinstance(labels_train, torch.Tensor) else torch.tensor(labels_train, dtype=torch.long)
        labels_test_tensor = labels_test if isinstance(labels_test, torch.Tensor) else torch.tensor(labels_test, dtype=torch.long)

        max_label_train = int(labels_train_tensor.max().item()) if labels_train_tensor.numel() > 0 else -1
        max_label_test = int(labels_test_tensor.max().item()) if labels_test_tensor.numel() > 0 else -1
        num_classes = max(max_label_train, max_label_test) + 1 if max(max_label_train, max_label_test) >= 0 else 0

        # Select locked feature subsets for threshold + recall computation.
        if already_locked:
            # Matrices are already filtered to locked columns
            train_locked_matrix = feature_matrix_train
            test_locked_matrix = feature_matrix_test
        else:
            # Need to slice out locked columns
            train_locked_matrix = feature_matrix_train[:, locked_features]
            test_locked_matrix = feature_matrix_test[:, locked_features]

        thresholds = compute_activation_thresholds(train_locked_matrix, percentile=recall_percentile)
        train_recall_matrix = compute_recall_by_class(train_locked_matrix, labels_train_tensor, thresholds, num_classes)
        test_recall_matrix = compute_recall_by_class(test_locked_matrix, labels_test_tensor, thresholds, num_classes)
        train_precision_matrix = compute_precision_at_threshold(train_locked_matrix, labels_train_tensor, thresholds, num_classes)
        test_precision_matrix = compute_precision_at_threshold(test_locked_matrix, labels_test_tensor, thresholds, num_classes)

        train_auprc_matrix = compute_auprc_by_class(train_locked_matrix, labels_train_tensor, num_classes)
        test_auprc_matrix = compute_auprc_by_class(test_locked_matrix, labels_test_tensor, num_classes)

        thresholds_cpu = thresholds.detach().cpu()
        train_recall_cpu = train_recall_matrix.detach().cpu()
        test_recall_cpu = test_recall_matrix.detach().cpu()
        train_precision_cpu = train_precision_matrix.detach().cpu()
        test_precision_cpu = test_precision_matrix.detach().cpu()
        train_auprc_cpu = train_auprc_matrix.detach().cpu()
        test_auprc_cpu = test_auprc_matrix.detach().cpu()

        feature_pos_lookup = {feat_idx: pos for pos, feat_idx in enumerate(locked_features)}

        train_class_counts = torch.bincount(labels_train_tensor.cpu(), minlength=num_classes).to(dtype=torch.float32)
        test_class_counts = torch.bincount(labels_test_tensor.cpu(), minlength=num_classes).to(dtype=torch.float32)
        total_train = float(labels_train_tensor.shape[0]) if labels_train_tensor.numel() > 0 else 0.0
        total_test = float(labels_test_tensor.shape[0]) if labels_test_tensor.numel() > 0 else 0.0
        if total_train > 0.0:
            train_class_fraction = train_class_counts / total_train
        else:
            train_class_fraction = torch.zeros(num_classes, dtype=torch.float32)
        if total_test > 0.0:
            test_class_fraction = test_class_counts / total_test
        else:
            test_class_fraction = torch.zeros(num_classes, dtype=torch.float32)
        train_class_fraction_list = train_class_fraction.tolist()
        test_class_fraction_list = test_class_fraction.tolist()
    else:
        thresholds_cpu = torch.tensor([])
        train_recall_cpu = torch.zeros((0, 0))
        test_recall_cpu = torch.zeros((0, 0))
        train_precision_cpu = torch.zeros((0, 0))
        test_precision_cpu = torch.zeros((0, 0))
        train_auprc_cpu = torch.zeros((0, 0))
        test_auprc_cpu = torch.zeros((0, 0))
        feature_pos_lookup = {}
        num_classes = 0
        train_class_fraction_list = []
        test_class_fraction_list = []

    # --- coverage per feature (full matrices, then index) ---
    if already_locked:
        # Matrices are already filtered, so coverage is directly computed
        coverage_train_full = (feature_matrix_train > eps).float().mean(dim=0)  # [num_locked]
        coverage_test_full  = (feature_matrix_test  > eps).float().mean(dim=0)
    else:
        # Compute coverage on full matrices
        coverage_train_full = (feature_matrix_train > eps).float().mean(dim=0)  # [F_total]
        coverage_test_full  = (feature_matrix_test  > eps).float().mean(dim=0)

    purity_diffs = []
    same_majority_count = 0
    classification_counts = {'monosemantic': 0, 'nearly-mono': 0, 'dead': 0, 'polysemantic': 0}
    classification_counts_auprc = {'monosemantic': 0, 'nearly-mono': 0, 'polysemantic': 0}
    coverage_floor = 0.10
    min_coverage_floor = 0.01
    coverage_floor_relax = 0.9  # relax coverage when percentile gating is very strict
    percentile_positive_fraction = max(1e-6, (100.0 - recall_percentile) / 100.0)
    percentile_floor = max(min_coverage_floor, percentile_positive_fraction * coverage_floor_relax)

    auprc_selectivity_flags = []

    for local_idx, feature_idx in enumerate(locked_features):
        train_info = train_analysis[feature_idx]
        test_info  = test_analysis[feature_idx]
        mono_info  = mono_scores[feature_idx]

        train_purity = train_info['purity']
        test_purity  = test_info['purity']
        min_purity   = min(train_purity, test_purity)

        margin = mono_info['M']  # unified margin used for heuristic
        # Use local_idx for coverage lookup in already_locked mode
        cov_idx = local_idx if already_locked else feature_idx
        train_cov = float(coverage_train_full[cov_idx].item())
        test_cov  = float(coverage_test_full[cov_idx].item())
        combined_cov = train_cov + test_cov  # zero only if both zero

        # Heuristic classification
        if (margin >= 0.30) or (margin >= 0.18 and min_purity >= 0.95):
            cls_label = "monosemantic"
        elif margin >= 0.10 and min_purity >= 0.60:
            cls_label = "nearly-mono"
        elif combined_cov == 0.0:  # dead if coverage zero on both splits
            cls_label = "dead"
        else:
            cls_label = "polysemantic"
        classification_counts[cls_label] += 1

        same_majority = train_info['majority_label'] == test_info['majority_label']
        purity_diff = abs(train_purity - test_purity)
        if same_majority:
            same_majority_count += 1
        purity_diffs.append(purity_diff)

        feature_comparison = {
            'feature_index': feature_idx,
            'train_majority': train_info['majority_label_name'],
            'test_majority': test_info['majority_label_name'],
            'train_purity': train_purity,
            'test_purity': test_purity,
            'purity_difference': purity_diff,
            'same_majority_label': same_majority,
            'train_max_activation': train_info['max_activation'],
            'test_max_activation': test_info['max_activation'],
            'train_mean_activation': train_info['mean_activation'],
            'test_mean_activation': test_info['mean_activation'],
            # Coverage
            'train_coverage': train_cov,
            'test_coverage': test_cov,
            'coverage_difference': abs(train_cov - test_cov),
            # Monosemanticity scores (normalized)
            'monosemanticity_score': mono_info['M'],
            'train_margin': mono_info['m_train'],
            'test_margin': mono_info['m_test'],
            'dominant_class_train': mono_info['c_star_train_name'],
            'dominant_class_test': mono_info['c_star_test_name'],
            'dominant_class_flipped': mono_info['flipped'],
            # Monosemanticity scores (softmax)
            'monosemanticity_score_softmax': mono_info['M_softmax'],
            'train_margin_softmax': mono_info['m_train_softmax'],
            'test_margin_softmax': mono_info['m_test_softmax'],
            'dominant_class_train_softmax': mono_info['c_star_train_name_softmax'],
            'dominant_class_test_softmax': mono_info['c_star_test_name_softmax'],
            'dominant_class_flipped_softmax': mono_info['flipped_softmax'],
            # Heuristic classification
            'min_train_test_purity': min_purity,
            'classification_label': cls_label
        }

        if locked_features:
            pos = feature_pos_lookup[feature_idx]
            feature_comparison.update({
                'recall_threshold': float(thresholds_cpu[pos].item()),
                'recall_percentile': recall_percentile,
                'train_recall_by_class': train_recall_cpu[:, pos].tolist(),
                'test_recall_by_class': test_recall_cpu[:, pos].tolist(),
                'train_precision_at_threshold_by_class': train_precision_cpu[:, pos].tolist(),
                'test_precision_at_threshold_by_class': test_precision_cpu[:, pos].tolist(),
                'train_auprc_by_class': train_auprc_cpu[:, pos].tolist(),
                'test_auprc_by_class': test_auprc_cpu[:, pos].tolist()
            })

            if num_classes > 0:
                train_auprc_vec = train_auprc_cpu[:, pos].numpy()
                test_auprc_vec = test_auprc_cpu[:, pos].numpy()
                combined = np.minimum(train_auprc_vec, test_auprc_vec)
                combined_list = combined.tolist()
                best_idx = int(np.argmax(combined)) if any(combined) else 0
                best_score = float(combined_list[best_idx]) if combined_list else 0.0
                sorted_scores = sorted(combined_list, reverse=True)
                second_best = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                selectivity_gap_val = best_score - second_best
                best_score_rounded = round(best_score, 3)
                is_selective = (best_score_rounded >= selectivity_threshold) and (selectivity_gap_val >= selectivity_gap)
                train_auprc_vec = train_auprc_cpu[:, pos].numpy()
                test_auprc_vec = test_auprc_cpu[:, pos].numpy()
                combined = np.minimum(train_auprc_vec, test_auprc_vec)
                combined_list = combined.tolist()
                best_idx = int(np.argmax(combined)) if any(combined) else 0
                best_score = float(combined_list[best_idx]) if combined_list else 0.0
                sorted_scores = sorted(combined_list, reverse=True)
                second_best = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                selectivity_gap_val = best_score - second_best
                best_score_rounded = round(best_score, 3)
                train_class_cov = float(train_recall_cpu[best_idx, pos].item()) if best_idx < train_recall_cpu.shape[0] else 0.0
                test_class_cov = float(test_recall_cpu[best_idx, pos].item()) if best_idx < test_recall_cpu.shape[0] else 0.0

                train_floor_candidates = [coverage_floor, percentile_floor]
                if best_idx < len(train_class_fraction_list):
                    train_class_frac = train_class_fraction_list[best_idx]
                    if train_class_frac > 0.0:
                        train_floor_candidates.append(train_class_frac)
                train_floor = max(min_coverage_floor, min(train_floor_candidates))

                test_floor_candidates = [coverage_floor, percentile_floor]
                if best_idx < len(test_class_fraction_list):
                    test_class_frac = test_class_fraction_list[best_idx]
                    if test_class_frac > 0.0:
                        test_floor_candidates.append(test_class_frac)
                test_floor = max(min_coverage_floor, min(test_floor_candidates))
                class_cov_ok = (train_class_cov >= train_floor) and (test_class_cov >= test_floor)
                is_selective = (best_score_rounded >= selectivity_threshold) and (selectivity_gap_val >= selectivity_gap) and class_cov_ok
                auprc_selectivity_flags.append(is_selective)
                if not class_cov_ok:
                    is_selective = False
                    auprc_label = 'polysemantic'
                elif selectivity_gap_val < selectivity_gap:
                    auprc_label = 'polysemantic'
                elif best_score_rounded >= selectivity_threshold:
                    auprc_label = 'monosemantic'
                elif best_score_rounded >= nearly_mono_threshold:
                    auprc_label = 'nearly-mono'
                else:
                    auprc_label = 'polysemantic'
                auprc_selectivity_flags.append(is_selective)
                if auprc_label in classification_counts_auprc:
                    classification_counts_auprc[auprc_label] += 1
                feature_comparison.update({
                    'auprc_selectivity_score': best_score,
                    'auprc_selectivity_gap': selectivity_gap_val,
                    'auprc_selectivity_class_index': best_idx,
                    'auprc_selectivity_class_name': label2desc.get(best_idx, f"Label_{best_idx}"),
                    'auprc_selectivity_is_monosemantic': bool(is_selective),
                    'auprc_selectivity_threshold': selectivity_threshold,
                    'auprc_selectivity_gap_threshold': selectivity_gap,
                    'auprc_selectivity_class_train_coverage': train_class_cov,
                    'auprc_selectivity_class_test_coverage': test_class_cov,
                    'auprc_selectivity_train_recall_floor': train_floor,
                    'auprc_selectivity_test_recall_floor': test_floor,
                    'auprc_selectivity_score_rounded': best_score_rounded,
                    'auprc_selectivity_nearly_mono_threshold': nearly_mono_threshold,
                    'classification_label_auprc': auprc_label
                })
            else:
                auprc_selectivity_flags.append(False)

        comparison['feature_comparisons'].append(feature_comparison)
        if not same_majority:
            comparison['flippers'].append(feature_comparison)

    mono_summary = calculate_monosemanticity_summary_stats(mono_scores)

    # Coverage summary stats (locked only)
    train_coverages_locked = [fc['train_coverage'] for fc in comparison['feature_comparisons']]
    test_coverages_locked  = [fc['test_coverage']  for fc in comparison['feature_comparisons']]
    coverage_diffs = [fc['coverage_difference'] for fc in comparison['feature_comparisons']]
    if len(train_coverages_locked) > 1:
        coverage_corr = float(np.corrcoef(train_coverages_locked, test_coverages_locked)[0, 1])
    else:
        coverage_corr = 0.0

    comparison['summary_stats'] = {
        'same_majority_label_count': same_majority_count,
        'same_majority_label_percentage': same_majority_count / len(locked_features) * 100 if locked_features else 0.0,
        'average_purity_difference': sum(purity_diffs) / len(purity_diffs) if purity_diffs else 0.0,
        'max_purity_difference': max(purity_diffs) if purity_diffs else 0.0,
        'min_purity_difference': min(purity_diffs) if purity_diffs else 0.0,
        'flipper_count': len(comparison['flippers']),
        'classification_counts': classification_counts,
        'classification_percentages': {
            k: (v / len(locked_features) * 100) if locked_features else 0.0
            for k, v in classification_counts.items()
        },
        'classification_counts_auprc': classification_counts_auprc,
        'classification_percentages_auprc': {
            k: (v / len(locked_features) * 100) if locked_features else 0.0
            for k, v in classification_counts_auprc.items()
        },
        'coverage': {
            'mean_train_coverage': float(np.mean(train_coverages_locked)) if train_coverages_locked else 0.0,
            'mean_test_coverage': float(np.mean(test_coverages_locked)) if test_coverages_locked else 0.0,
            'mean_coverage_difference': float(np.mean(coverage_diffs)) if coverage_diffs else 0.0,
            'max_coverage_difference': float(np.max(coverage_diffs)) if coverage_diffs else 0.0,
            'min_coverage_difference': float(np.min(coverage_diffs)) if coverage_diffs else 0.0,
            'coverage_correlation': coverage_corr
        },
        'monosemanticity': mono_summary
    }

    if locked_features:
        mono_count = int(sum(auprc_selectivity_flags))
        comparison['summary_stats']['auprc_selectivity'] = {
            'is_monosemantic_count': mono_count,
            'percentage': (mono_count / len(locked_features) * 100.0) if locked_features else 0.0,
            'threshold': selectivity_threshold,
            'gap_threshold': selectivity_gap,
            'nearly_mono_threshold': nearly_mono_threshold
        }

    if locked_features:
        comparison['recall_metadata'] = {
            'percentile': recall_percentile,
            'num_classes': num_classes
        }

    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    rprint("Monosemanticity + coverage + classification analysis complete:")
    rprint(f"  Mean monosemanticity score: {mono_summary['mean_monosemanticity']:.3f}")
    rprint(f"  High monosemanticity features (>0.8): {mono_summary['high_monosemanticity_count']}/{mono_summary['total_features']} ({mono_summary['high_monosemanticity_percentage']:.1f}%)")
    rprint(f"  Dominant class flipped: {mono_summary['flipped_dominant_class_count']}/{mono_summary['total_features']} ({mono_summary['flipped_dominant_class_percentage']:.1f}%)")
    rprint(f"  Classification counts: {classification_counts}")
    if mono_summary['failed_train_features'] > 0 or mono_summary['failed_test_features'] > 0:
        rprint(f"  Failed cases - Train: {mono_summary['failed_train_features']}, Test: {mono_summary['failed_test_features']}")

    return comparison


def print_locked_comparison_summary(comparison):
    """Print a summary of the locked feature comparison"""
    stats = comparison['summary_stats']

    print(f"\n=== LOCKED FEATURE CONSISTENCY ANALYSIS ===")
    print(f"Total locked features: {comparison['locked_feature_count']}")
    print(f"Features with same majority label: {stats['same_majority_label_count']}/{comparison['locked_feature_count']} ({stats['same_majority_label_percentage']:.1f}%)")
    print(f"Flipper features (different majority): {stats['flipper_count']}", file=sys.stderr)
    print(f"Average purity difference: {stats['average_purity_difference']:.3f}")
    print(f"Max purity difference: {stats['max_purity_difference']:.3f}")
    print(f"Min purity difference: {stats['min_purity_difference']:.3f}")

    # Sort by consistency (same majority + low purity diff)
    consistent_features = [f for f in comparison['feature_comparisons']
                          if f['same_majority_label']]
    consistent_features.sort(key=lambda x: x['purity_difference'])

    print(f"\nTop 5 most consistent features (same majority, lowest purity diff):")
    for i, feat in enumerate(consistent_features[:5]):
        print(f"  {i+1}. Feature {feat['feature_index']}: {feat['train_majority']} "
              f"(train purity: {feat['train_purity']:.3f}, test purity: {feat['test_purity']:.3f}, "
              f"diff: {feat['purity_difference']:.3f})")

    # Show flippers
    if comparison['flippers']:
        print(f"\nFlipper features (different majority labels):")
        for i, feat in enumerate(comparison['flippers'][:5]):
            print(f"  {i+1}. Feature {feat['feature_index']}: {feat['train_majority']} → {feat['test_majority']} "
                  f"(purity diff: {feat['purity_difference']:.3f})")

def rprint(*args, **kwargs):
        print(*args, **kwargs, file=report_file)
