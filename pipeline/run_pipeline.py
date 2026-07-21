"""Train the paper SAE and build the artifacts consumed by Histoscope."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_NAME = "model-exp49152-l2-zscore-tied-prebias-acttopk250-bs32-lr0.0001"


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Histoscope SAE and generate its dashboard artifacts."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Embedding cache containing train/, test/, and label_map.json",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("sae-models"),
        help="Output root for the checkpoint and analysis bundle",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Existing model directory. When supplied, skip training and build analysis artifacts only.",
    )
    parser.add_argument("--num-features", type=int, default=100)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--all-viz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_dir = Path(__file__).resolve().parent
    repo_root = pipeline_dir.parent
    cache_root = args.cache_root.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()

    required = [cache_root / "train", cache_root / "test", cache_root / "label_map.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required cache inputs:\n  " + "\n  ".join(missing))

    if args.checkpoint:
        model_dir = args.checkpoint.expanduser().resolve()
    else:
        train_command = [
            sys.executable,
            str(pipeline_dir / "simple_sae_spider.py"),
            "--dataset", "spider",
            "--cache-root", str(cache_root),
            "--out-dir", str(models_dir),
            "--input-dim", "1024",
            "--hidden-dim", "49152",
            "--batch-size", "32",
            "--learning-rate", "0.0001",
            "--epochs", "2",
            "--lambda-values", "0.4",
            "--l2",
            "--zscore",
            "--tie-weights",
            "--use-pre-bias",
            "--activation", "topk",
            "--topk-k", "250",
            "--seed", "42",
        ]
        run(train_command, cwd=repo_root)
        model_dir = models_dir / "spider" / MODEL_NAME

    if not (model_dir / "model.pt").exists() or not (model_dir / "metadata.json").exists():
        raise SystemExit(f"Checkpoint bundle is incomplete: {model_dir}")

    analysis_command = [
        sys.executable,
        str(pipeline_dir / "sae_feature_analysis_v2.py"),
        str(model_dir),
        "--num-features", str(args.num_features),
        "--topk-samples", "25",
        "--recall-percentile", "95",
        "--auprc-nearly-mono-threshold", "0.65",
    ]
    if args.cluster:
        analysis_command.append("--cluster")
    if args.all_viz:
        analysis_command.append("--all-viz")
    run(analysis_command, cwd=repo_root)

    print("\nHistoscope artifacts created under:")
    print(model_dir / "analysis")
    print("\nLaunch the dashboard with:")
    print(f"{sys.executable} histoscope.py --models-dir {models_dir}")


if __name__ == "__main__":
    main()
