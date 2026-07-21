"""Entry point for the HistoSCOPE Dash application."""
import argparse

from histoscope.app import create_app, main

__all__ = ["create_app", "main"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the HistoSCOPE dashboard")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing model analysis bundles (default: ./sae-models)",
    )
    parser.add_argument(
        "--medgemma",
        action="store_true",
        help="Enable MedGemma image captioning model (loads large transformers weights)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(enable_medgemma=args.medgemma, models_dir=args.models_dir)
