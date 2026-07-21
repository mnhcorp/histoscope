"""Flipper analysis tab layout and callbacks."""
from __future__ import annotations

import os
from typing import List

import dash
from dash import Input, Output, dcc, html

from .. import data
from ..context import AppContext
from ..utils import encode_original_size_image


def discover_flipper_images(model_name: str) -> List[dict]:
    if not model_name or model_name not in data.MODEL_INDEX:
        return []
    mdir = data.get_model_dir(model_name)
    flippers_dir = os.path.join(mdir, "analysis", "top-k-features", "flippers")

    flipper_features: List[dict] = []
    if not os.path.exists(flippers_dir):
        return flipper_features

    for img_file in os.listdir(flippers_dir):
        if img_file.startswith("feature_") and img_file.endswith("_combined.png"):
            feature_id = img_file.replace("feature_", "").replace("_combined.png", "")
            img_path = os.path.join(flippers_dir, img_file)
            flipper_features.append(
                {"feature_id": feature_id, "image_path": img_path, "filename": img_file}
            )

    return sorted(flipper_features, key=lambda item: int(item["feature_id"]))


def layout(context: AppContext) -> dcc.Tab:
    return dcc.Tab(
        label="Flipper Analysis",
        value="flipper_tab",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Model"),
                            dcc.Dropdown(
                                id="model_select_tab4",
                                options=context.initial_model_options,
                                value=context.default_model,
                                clearable=False,
                                style={"width": "300px"},
                            ),
                        ],
                        style={"padding": "1rem"},
                    ),
                    html.Div(
                        [
                            html.P(
                                "Features that flipped their class association from train to test split:",
                                style={"fontSize": "1.0rem", "color": "#666", "marginBottom": "1rem"},
                            ),
                            html.Div(id="flipper_gallery", style={"padding": "0"}),
                        ],
                        style={"padding": "1rem"},
                    ),
                ]
            )
        ],
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(Output("flipper_gallery", "children"), Input("model_select_tab4", "value"))
    def update_flipper_gallery(model_name):
        flipper_features = discover_flipper_images(model_name)
        if not flipper_features:
            return html.Div(
                "No flipper analysis images found for this model.",
                style={"color": "orange", "fontSize": "1.1rem", "textAlign": "center", "padding": "2rem"},
            )

        dropdown_options = [
            {"label": f"Feature {feature['feature_id']}", "value": feature["image_path"]}
            for feature in flipper_features
        ]

        return html.Div(
            [
                html.H3(
                    f"Found {len(flipper_features)} flipper features",
                    style={"marginBottom": "1rem"},
                ),
                html.Div(
                    [
                        html.Label("Select feature:", style={"marginBottom": "0.5rem"}),
                        dcc.Dropdown(
                            id="flipper_feature_select",
                            options=dropdown_options,
                            placeholder="Select a flipper feature...",
                            clearable=False,
                            style={"width": "300px"},
                        ),
                    ],
                    style={"marginBottom": "2rem"},
                ),
                html.Div(id="flipper_image_display"),
            ]
        )

    @app.callback(Output("flipper_image_display", "children"), Input("flipper_feature_select", "value"))
    def display_flipper_image(image_path):
        if not image_path or not os.path.exists(image_path):
            return html.Div(
                "Select a feature to view its combined analysis.",
                style={"color": "#666", "textAlign": "center", "padding": "2rem"},
            )
        try:
            img_src = encode_original_size_image(image_path)
            return html.Div(
                [html.Img(src=img_src, style={"maxWidth": "100%", "height": "auto"})],
                style={"textAlign": "center"},
            )
        except Exception as exc:
            return html.Div(
                f"Error loading image: {exc}",
                style={"color": "red", "textAlign": "center", "padding": "2rem"},
            )
