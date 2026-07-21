"""Dash application assembly for HistoSCOPE."""
from __future__ import annotations

from pathlib import Path

import dash

from .patch_browser import register_patch_browser_callbacks

from . import layout
from .context import AppContext, create_context
from .descriptions import ensure_models_loaded, set_medgemma_enabled
from .tabs import cluster_analysis, feature_analysis, flipper, heatmap, report


def create_app(
    enable_medgemma: bool = False,
    models_dir: str | Path | None = None,
) -> tuple[dash.Dash, AppContext]:
    context = create_context(models_dir=models_dir)
    set_medgemma_enabled(enable_medgemma)
    ensure_models_loaded()

    app = dash.Dash(__name__, title="HistoSCOPE")
    app.layout = layout.build_layout(context)

    heatmap.register_callbacks(app, context)
    cluster_analysis.register_callbacks(app, context)
    feature_analysis.register_callbacks(app)
    flipper.register_callbacks(app)
    report.register_callbacks(app)
    register_patch_browser_callbacks(app, context.model_index)

    return app, context


def main(enable_medgemma: bool = False, models_dir: str | Path | None = None) -> None:
    app, _ = create_app(enable_medgemma=enable_medgemma, models_dir=models_dir)
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
