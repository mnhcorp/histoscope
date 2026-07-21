"""Feature analysis tab layout and callbacks."""
from __future__ import annotations

import os
from typing import Dict, List

import dash
from dash import Input, Output, State, dcc, html

from .. import data
from ..context import AppContext
from ..utils import encode_thumb


def discover_feature_images(model_name: str, split: str = "train") -> Dict[str, List[Dict[str, str]]]:
    if not model_name or model_name not in data.MODEL_INDEX:
        return {}
    mdir = data.get_model_dir(model_name)
    features_dir = os.path.join(mdir, "analysis", "top-k-features", split)

    class_features: Dict[str, List[Dict[str, str]]] = {}
    if not os.path.exists(features_dir):
        return class_features

    for class_name in os.listdir(features_dir):
        class_path = os.path.join(features_dir, class_name)
        if os.path.isdir(class_path):
            features: List[Dict[str, str]] = []
            for img_file in os.listdir(class_path):
                if img_file.startswith("feature_") and img_file.endswith(".png"):
                    feature_id = img_file.replace("feature_", "").replace(".png", "")
                    features.append(
                        {
                            "feature_id": feature_id,
                            "image_path": os.path.join(class_path, img_file),
                        }
                    )
            if features:
                class_features[class_name] = sorted(features, key=lambda item: int(item["feature_id"]))
    return class_features


def layout(context: AppContext) -> dcc.Tab:
    return dcc.Tab(
        label="Feature Analysis",
        value="feature_tab",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Model"),
                            dcc.Dropdown(
                                id="model_select_tab3",
                                options=context.initial_model_options,
                                value=context.default_model,
                                clearable=False,
                                style={"width": "300px"},
                            ),
                            html.Label("Split", style={"marginLeft": "1rem"}),
                            dcc.Dropdown(
                                id="split_select_tab3",
                                options=[{"label": "train", "value": "train"}, {"label": "test", "value": "test"}],
                                value="train",
                                clearable=False,
                                style={"width": "120px", "marginLeft": "0.5rem"},
                            ),
                            html.Label("Filter by class", style={"marginLeft": "1rem"}),
                            dcc.Dropdown(
                                id="class_filter",
                                options=[{"label": "All classes", "value": "all"}],
                                value="all",
                                clearable=False,
                                style={"width": "200px", "marginLeft": "0.5rem"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "alignItems": "center",
                            "padding": "1rem",
                        },
                    ),
                    html.Div(id="feature_gallery", style={"padding": "1rem"}),
                ]
            )
        ],
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        [Output("class_filter", "options"), Output("class_filter", "value")],
        [Input("model_select_tab3", "value"), Input("split_select_tab3", "value")],
        State("class_filter", "value"),
    )
    def update_class_filter_options(model_name, split, current_class):
        class_features = discover_feature_images(model_name, split)
        options = [{"label": "All classes", "value": "all"}]
        if class_features:
            for class_name in sorted(class_features.keys()):
                feature_count = len(class_features[class_name])
                options.append({"label": f"{class_name} ({feature_count})", "value": class_name})

        available_classes = ["all"] + list(class_features.keys())
        if current_class in available_classes:
            return options, current_class
        return options, "all"

    @app.callback(
        Output("feature_gallery", "children"),
        [Input("model_select_tab3", "value"), Input("split_select_tab3", "value"), Input("class_filter", "value")],
    )
    def update_feature_gallery(model_name, split, class_filter):
        class_features = discover_feature_images(model_name, split)

        if not class_features:
            return html.Div(
                "No feature analysis images found for this model/split combination.",
                style={"color": "orange", "fontSize": "1.1rem", "textAlign": "center", "padding": "2rem"},
            )

        locked_comparison = data.load_locked_feature_comparison(model_name)
        feature_purity_map = {}
        if locked_comparison and "feature_comparisons" in locked_comparison:
            for feature_data in locked_comparison["feature_comparisons"]:
                feature_id = str(feature_data["feature_index"])
                if data.is_baseline_model(model_name):
                    if split == "train":
                        purity = feature_data.get("purity_train_at_100", 0)
                    else:
                        purity = feature_data.get("purity_test_at_100", 0)
                else:
                    if split == "train":
                        purity = feature_data.get("train_purity", 0)
                    else:
                        purity = feature_data.get("test_purity", 0)
                feature_purity_map[feature_id] = purity

        if class_filter != "all":
            filtered_features = {class_filter: class_features.get(class_filter, [])}
        else:
            filtered_features = class_features

        if not any(filtered_features.values()):
            return html.Div(
                f"No features found for class: {class_filter}",
                style={"color": "orange", "fontSize": "1.1rem", "textAlign": "center", "padding": "2rem"},
            )

        gallery_sections: List = []
        for class_name, features in filtered_features.items():
            if not features:
                continue
            gallery_sections.append(
                html.H3(
                    f"{class_name} ({len(features)} features)",
                    style={"borderBottom": "2px solid #ddd", "paddingBottom": "0.5rem", "marginTop": "2rem"},
                )
            )

            feature_grid = []
            for feature in features:
                try:
                    img_src = encode_thumb(feature["image_path"], max_side=300)
                    feature_id = feature["feature_id"]
                    purity_text = ""
                    purity_color = "green"
                    if feature_id in feature_purity_map:
                        purity = feature_purity_map[feature_id]
                        purity_text = f" (purity: {purity:.2f})"
                        purity_color = "green" if purity >= 0.50 else "red"

                    feature_grid.append(
                        html.Div(
                            [
                                html.Button(
                                    [
                                        html.Img(
                                            src=img_src,
                                            style={"width": "100%", "height": "200px", "objectFit": "cover"},
                                        )
                                    ],
                                    id={"type": "feature_image", "index": feature["image_path"]},
                                    style={
                                        "border": "1px solid #ddd",
                                        "background": "none",
                                        "padding": "0",
                                        "cursor": "pointer",
                                        "width": "100%",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            f"Feature {feature['feature_id']}",
                                            style={"fontWeight": "bold"},
                                        ),
                                        html.Span(
                                            purity_text,
                                            style={"color": purity_color, "fontSize": "0.8rem"},
                                        )
                                        if purity_text
                                        else html.Span(),
                                    ],
                                    style={"textAlign": "center", "fontSize": "0.9rem", "padding": "0.5rem"},
                                ),
                            ],
                            style={"width": "200px", "margin": "0.5rem"},
                        )
                    )
                except Exception as exc:
                    print(f"Error loading feature image {feature['image_path']}: {exc}")
                    continue

            if feature_grid:
                gallery_sections.append(
                    html.Div(
                        feature_grid,
                        style={"display": "flex", "flexWrap": "wrap", "gap": "1rem", "padding": "1rem 0"},
                    )
                )

        return html.Div(gallery_sections)
