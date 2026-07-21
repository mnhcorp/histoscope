"""Application context and bootstrap logic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from . import data


@dataclass(frozen=True)
class AppContext:
    root: Path
    sae_models_dir: Path
    model_index: Dict[str, Dict]
    default_dataset: str
    default_model: str
    default_split: str
    dataset_models: Dict[str, List[str]]
    dataset_options: List[Dict[str, str]]
    default_model_by_dataset: Dict[str, str]
    initial_model_options: List[Dict[str, str]]
    initial_cache: Dict[str, object]
    slider_bounds: Tuple[int, int, int]
    cluster_splits: Dict[str, List[str]]
    page_size: int = 50


def _inject_baseline(model_index: Dict[str, Dict], sae_models_dir: Path) -> None:
    baseline_dir = sae_models_dir / "baseline"
    try:
        baseline_ic = baseline_dir / "analysis" / "interactive-cache"
        baseline_splits: List[str] = []
        for split in ("train", "test"):
            split_dir = baseline_ic / split
            cache_path = split_dir / "cache.npz"
            image_paths = split_dir / "image_paths.json"
            if cache_path.exists() and image_paths.exists():
                baseline_splits.append(split)
        if baseline_splits:
            try:
                modified = baseline_dir.stat().st_mtime
            except FileNotFoundError:
                modified = 0.0
            model_index["baseline"] = {
                "dir": str(baseline_dir),
                "splits": baseline_splits,
                "cluster_splits": [],
                "modified": modified,
            }
    except Exception as exc:
        print(f"Warning: could not register baseline model: {exc}")


def _build_dataset_maps(model_index: Dict[str, Dict]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    dataset_models: Dict[str, List[str]] = {}
    default_model_by_dataset: Dict[str, str] = {}

    for model_key in model_index.keys():
        dataset_name, _ = data.split_model_key(model_key)
        dataset_models.setdefault(dataset_name, []).append(model_key)

    for dataset_name, models in dataset_models.items():
        models.sort(key=lambda key: model_index[key].get("modified", 0.0), reverse=True)
        if models:
            default_model_by_dataset[dataset_name] = models[0]

    return dataset_models, default_model_by_dataset


def create_context(models_dir: str | Path | None = None) -> AppContext:
    root = Path(__file__).resolve().parent.parent
    sae_models_dir = Path(models_dir).expanduser().resolve() if models_dir else root / "sae-models"
    model_index = data.discover_models(str(sae_models_dir))
    if not model_index:
        raise RuntimeError(
            "No models with interactive cache found under: "
            f"{sae_models_dir}\nExpected: sae-models/<model>/analysis/interactive-cache/<split>/"
            "cache.npz + image_paths.json"
        )

    _inject_baseline(model_index, sae_models_dir)
    data.register_model_index(model_index)

    default_model = next(iter(model_index.keys()))
    default_dataset, _ = data.split_model_key(default_model)
    splits = model_index[default_model]["splits"]
    default_split = "train" if "train" in splits else splits[0]

    dataset_models, default_model_by_dataset = _build_dataset_maps(model_index)

    dataset_options = [
        {"label": dataset_name, "value": dataset_name}
        for dataset_name in sorted(dataset_models.keys())
    ]

    initial_model_options = [
        {"label": data.model_display_label(model_key), "value": model_key}
        for model_key in dataset_models.get(default_dataset, [])
    ]

    initial_cache = data.load_cache(default_model, default_split)
    slider_bounds = data.slider_bounds(initial_cache["H"])
    cluster_splits = {model_key: info.get("cluster_splits", []) for model_key, info in model_index.items()}

    return AppContext(
        root=root,
        sae_models_dir=sae_models_dir,
        model_index=model_index,
        default_dataset=default_dataset,
        default_model=default_model,
        default_split=default_split,
        dataset_models=dataset_models,
        dataset_options=dataset_options,
        default_model_by_dataset=default_model_by_dataset,
        initial_model_options=initial_model_options,
        initial_cache=initial_cache,
        slider_bounds=slider_bounds,
        cluster_splits=cluster_splits,
    )
