"""Data access utilities for the HistoSCOPE app."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

MODEL_INDEX: Dict[str, Dict] = {}
DISPLAY_NAME_MAP = {"baseline": "baseline UNI activations"}


def register_model_index(model_index: Dict[str, Dict]) -> None:
    """Register the discovered model index for downstream helpers."""
    global MODEL_INDEX
    MODEL_INDEX = model_index
    load_cache.cache_clear()
    load_cluster_cache.cache_clear()
    load_metadata.cache_clear()


def split_model_key(model_key: str | None) -> Tuple[str, str]:
    """Return (dataset, model_name) from a MODEL_INDEX key."""
    if not model_key:
        return "default", ""
    parts = model_key.split("/")
    if len(parts) >= 3:
        return parts[-2], "/".join(parts[:-2] + parts[-1:])
    if len(parts) == 2:
        return parts[0], parts[1]
    return "default", parts[0]


def discover_models(base_dir: str) -> Dict[str, Dict]:
    """Discover models that have an interactive cache anywhere under base_dir."""
    models: Dict[str, Dict] = {}
    if not os.path.isdir(base_dir):
        return models

    for root, dirs, _ in os.walk(base_dir):
        if "analysis" not in dirs:
            continue

        ic_base = os.path.join(root, "analysis", "interactive-cache")
        if not os.path.isdir(ic_base):
            continue

        splits: List[str] = []
        for split in sorted(os.listdir(ic_base)):
            split_dir = os.path.join(ic_base, split)
            cache_path = os.path.join(split_dir, "cache.npz")
            image_paths_path = os.path.join(split_dir, "image_paths.json")
            if os.path.exists(cache_path) and os.path.exists(image_paths_path):
                splits.append(split)

        cluster_splits: List[str] = []
        cluster_dir = os.path.join(root, "analysis", "cluster-cache")
        if os.path.isdir(cluster_dir):
            for fname in sorted(os.listdir(cluster_dir)):
                if not fname.endswith("_clusters.npz"):
                    continue
                split_name = fname[: -len("_clusters.npz")]
                json_path = os.path.join(cluster_dir, f"{split_name}_clusters.json")
                if os.path.exists(json_path):
                    cluster_splits.append(split_name)

        if splits:
            rel_name = os.path.relpath(root, base_dir)
            if rel_name == ".":
                rel_name = os.path.basename(root)
            models[rel_name] = {
                "dir": root,
                "splits": splits,
                "cluster_splits": cluster_splits,
                "modified": os.path.getmtime(root),
            }

        if "analysis" in dirs:
            dirs.remove("analysis")

    sorted_items = sorted(models.items(), key=lambda kv: kv[1].get("modified", 0.0), reverse=True)
    return dict(sorted_items)


def get_model_dir(model_name: str) -> str:
    if model_name not in MODEL_INDEX:
        raise KeyError(f"Unknown model name: {model_name}")
    return MODEL_INDEX[model_name]["dir"]


def model_display_label(model_key: str) -> str:
    """Return a human-friendly label for the given model key."""
    _, model_name = split_model_key(model_key)
    short_name = model_name.split("/")[-1]
    return DISPLAY_NAME_MAP.get(short_name, short_name)


def is_baseline_model(model_key: str | None) -> bool:
    """Return True if the provided key refers to a baseline model."""
    if not model_key:
        return False
    _, model_name = split_model_key(model_key)
    return model_name == "baseline"


def get_cluster_splits(model_name: str) -> List[str]:
    if model_name not in MODEL_INDEX:
        return []
    splits = MODEL_INDEX[model_name].get("cluster_splits", [])
    return list(splits)


@lru_cache(maxsize=16)
def load_cache(model_name: str, split: str):
    """Load cached arrays for a given model and split; memoized."""
    if model_name not in MODEL_INDEX:
        raise KeyError(f"Unknown model: {model_name}")

    mdir = MODEL_INDEX[model_name]["dir"]
    cdir = os.path.join(mdir, "analysis", "interactive-cache", split)
    arr = np.load(os.path.join(cdir, "cache.npz"), allow_pickle=True)
    with open(os.path.join(cdir, "image_paths.json")) as f:
        image_paths = json.load(f)

    feat_ids = arr["top_feature_indices"].tolist()
    feat_labels = [str(int(i)) for i in feat_ids]
    class_names = arr["class_names"].tolist()
    if "class_counts" in arr.files:
        counts: Iterable[int] = arr["class_counts"].tolist()
        class_labels = [f"{n} (n={c})" for n, c in zip(class_names, counts)]
    else:
        class_labels = [f"{n}" for n in class_names]

    return {
        "H": arr["heatmap"],
        "feat_labels": feat_labels,
        "class_labels": class_labels,
        "TOPK_IDX": arr["topk_indices"],
        "TOPK_VAL": arr["topk_values"],
        "image_paths": image_paths,
    }


@lru_cache(maxsize=16)
def load_cluster_cache(model_name: str, split: str = "train") -> Optional[Dict[str, object]]:
    """Load cluster artifacts for a given model/split if available."""
    if model_name not in MODEL_INDEX:
        raise KeyError(f"Unknown model: {model_name}")

    mdir = MODEL_INDEX[model_name]["dir"]
    cluster_dir = os.path.join(mdir, "analysis", "cluster-cache")
    npz_path = os.path.join(cluster_dir, f"{split}_clusters.npz")
    meta_path = os.path.join(cluster_dir, f"{split}_clusters.json")
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        return None

    arr = np.load(npz_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    if "representative_patches" in metadata and isinstance(metadata["representative_patches"], dict):
        try:
            rep_map = {int(k): v for k, v in metadata["representative_patches"].items()}
        except ValueError:
            rep_map = metadata["representative_patches"]
        metadata["representative_patches_int"] = rep_map

    result: Dict[str, object] = {
        "feature_indices": arr["feature_indices"],
        "class_ids": arr["class_ids"],
        "class_means": arr["class_means"],
        "cluster_ids": arr["cluster_ids"],
        "pca_coords": arr["pca_coords"],
        "feature_strength": arr["feature_strength"],
        "metadata": metadata,
    }
    if "class_means_normalized" in arr.files:
        result["class_means_normalized"] = arr["class_means_normalized"]
    if "pca_components" in arr.files:
        result["pca_components"] = arr["pca_components"]
    if "pca_explained" in arr.files:
        result["pca_explained"] = arr["pca_explained"]

    return result


@lru_cache(maxsize=16)
def load_metadata(model_name: str) -> Dict:
    mdir = get_model_dir(model_name)
    meta_path = os.path.join(mdir, "metadata.json")
    with open(meta_path, "r") as f:
        return json.load(f)


def load_full_report(model_name: str) -> str:
    mdir = get_model_dir(model_name)
    report_path = os.path.join(mdir, "analysis", "report.txt")
    if not os.path.exists(report_path):
        return "No report.txt found"
    try:
        with open(report_path, "r") as f:
            return f.read()
    except Exception as exc:
        return f"Error reading report.txt: {exc}"


def load_locked_feature_comparison(model_name: str) -> Dict:
    """Load locked or baseline feature comparison data if available."""
    if model_name not in MODEL_INDEX:
        return {}
    mdir = get_model_dir(model_name)
    candidates: List[str] = []
    if is_baseline_model(model_name):
        candidates.append(os.path.join(mdir, "analysis", "baseline_feature_comparison.json"))
    candidates.append(os.path.join(mdir, "analysis", "locked_feature_comparison.json"))
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"Error loading feature comparison {path}: {exc}")
    return {}


def slider_bounds(H: np.ndarray) -> Tuple[int, int, int]:
    max_feats = min(200, H.shape[1])
    default_feats = min(50, H.shape[1])
    min_feats = min(5, max_feats)
    return min_feats, default_feats, max_feats


def _feature_notes_path(model_name: str) -> str:
    """Return the on-disk path that stores per-neuron notes for a model."""
    mdir = get_model_dir(model_name)
    return os.path.join(mdir, "analysis", "feature_notes.json")


def load_feature_notes(model_name: str) -> Dict[str, str]:
    """Load persisted neuron notes for the given model."""
    if model_name not in MODEL_INDEX:
        return {}

    path = _feature_notes_path(model_name)
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"Error loading feature notes from {path}: {exc}")
        return {}

    if not isinstance(payload, dict):
        return {}

    notes: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            notes[str(key)] = value
    return notes


def save_feature_notes(model_name: str, notes: Dict[str, str]) -> None:
    """Persist neuron notes for the given model to disk."""
    if model_name not in MODEL_INDEX:
        raise KeyError(f"Unknown model: {model_name}")

    path = _feature_notes_path(model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    safe_notes = {str(key): value for key, value in notes.items() if isinstance(value, str)}

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_notes, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Error saving feature notes to {path}: {exc}")
