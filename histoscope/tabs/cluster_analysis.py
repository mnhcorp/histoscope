"""Cluster analysis tab layout and callbacks."""
from __future__ import annotations

from typing import Dict, List, Optional

import dash
import numpy as np
from dash import Input, Output, State, dcc, html
from plotly import colors as plotly_colors
from plotly import graph_objects as go

from .. import data
from ..context import AppContext
from ..utils import encode_thumb


def _empty_figure(message: str = "No cluster data available.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=30),
        annotations=[
            dict(
                text=message,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=16, color="#666"),
            )
        ],
    )
    return fig


def layout(context: AppContext) -> dcc.Tab:
    default_cluster_splits = context.cluster_splits.get(context.default_model, [])
    default_split = None
    if "train" in default_cluster_splits:
        default_split = "train"
    elif default_cluster_splits:
        default_split = default_cluster_splits[0]

    split_options = [{"label": s, "value": s} for s in default_cluster_splits]

    return dcc.Tab(
        label="Cluster Analysis",
        value="cluster_tab",
        children=[
            dcc.Store(id="cluster_data_store", data=None),
            html.Div(
                [
                    html.Label("Model"),
                    dcc.Dropdown(
                        id="model_select_tab5",
                        options=context.initial_model_options,
                        value=context.default_model,
                        clearable=False,
                        style={"width": "320px"},
                    ),
                    html.Label("Split", style={"marginLeft": "1rem"}),
                    dcc.Dropdown(
                        id="cluster_split_select",
                        options=split_options,
                        value=default_split,
                        clearable=False,
                        placeholder="Select split",
                        style={"width": "160px", "marginLeft": "0.5rem"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                    "padding": "0.75rem 1rem 0.25rem",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(id="cluster_status", style={"padding": "0.25rem 1rem", "color": "#444"}),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(id="cluster_scatter", figure=_empty_figure(), style={"height": "600px"}),
                        type="circle",
                    ),
                ],
                style={"padding": "0 1rem"},
            ),
            html.Div(
                [
                    html.H4("Representative patch preview", style={"margin": "0.5rem 0"}),
                    html.Div(
                        id="cluster_hover_preview",
                        style={
                            "minHeight": "180px",
                            "display": "flex",
                            "alignItems": "flex-start",
                            "gap": "1rem",
                            "padding": "0.2rem 0",
                        },
                    ),
                ],
                style={"padding": "0 1rem"},
            ),
            html.Div(
                id="cluster_summary",
                style={"padding": "0.5rem 1rem 1.5rem"},
            ),
        ],
    )


def register_callbacks(app: dash.Dash, context: AppContext) -> None:
    @app.callback(
        Output("cluster_split_select", "options"),
        Output("cluster_split_select", "value"),
        Input("model_select_tab5", "value"),
        State("cluster_split_select", "value"),
    )
    def update_cluster_split_options(model_name, current_value):
        if not model_name:
            return [], None
        splits = data.get_cluster_splits(model_name)
        options = [{"label": s, "value": s} for s in splits]
        if not splits:
            return options, None
        if current_value in splits:
            return options, current_value
        default = "train" if "train" in splits else splits[0]
        return options, default

    @app.callback(
        Output("cluster_data_store", "data"),
        Output("cluster_scatter", "figure"),
        Output("cluster_status", "children"),
        Output("cluster_summary", "children"),
        Input("model_select_tab5", "value"),
        Input("cluster_split_select", "value"),
    )
    def load_cluster_data(model_name: str, split: Optional[str]):
        if not model_name:
            message = html.Div("Select a model to view cluster analysis.", style={"color": "#666"})
            return None, _empty_figure("Select a model to begin."), message, html.Div()
        if not split:
            message = html.Div("No cluster artifacts detected for this model.", style={"color": "#a33"})
            hint = "Run sae_feature_analysis_v2.py --cluster to generate clustering artifacts."
            return None, _empty_figure(hint), message, html.Div()

        payload = data.load_cluster_cache(model_name, split)
        if not payload:
            message = html.Div(
                [
                    html.Span("Cluster artifacts not found for "),
                    html.B(f"{model_name} • {split}"),
                    html.Span(". Run analysis with "),
                    html.Code("--cluster"),
                    html.Span(" to generate them."),
                ],
                style={"color": "#a33"},
            )
            return None, _empty_figure("No clustering artifacts found."), message, html.Div()

        metadata = payload.get("metadata", {})
        representative = metadata.get("representative_patches_int") or {}
        store_payload: Dict[str, object] = {
            "model": model_name,
            "split": split,
            "feature_indices": payload["feature_indices"].astype(int).tolist(),
            "cluster_ids": payload["cluster_ids"].astype(int).tolist(),
            "coords": payload["pca_coords"].tolist(),
            "class_means": payload["class_means"].tolist(),
            "class_names": metadata.get("class_names", []),
            "feature_strength": payload["feature_strength"].tolist(),
            "algorithm": metadata.get("algorithm", "unknown"),
            "algorithm_params": metadata.get("algorithm_params", {}),
            "cluster_summary": metadata.get("cluster_summary", []),
            "representative_patches": {str(k): v for k, v in representative.items()},
            "num_features": int(len(payload["feature_indices"])),
            "num_clusters": int(len({int(c) for c in payload["cluster_ids"].tolist()})),
        }

        scatter = _build_cluster_figure(store_payload)
        status = _format_status(store_payload)
        summary = _build_summary(store_payload)

        return store_payload, scatter, status, summary

    @app.callback(
        Output("cluster_hover_preview", "children"),
        Input("cluster_scatter", "hoverData"),
        State("cluster_data_store", "data"),
    )
    def update_hover_preview(hover_data, store):
        placeholder = html.Div(
            "Hover a point to see its representative patch.",
            style={"color": "#555", "fontStyle": "italic"},
        )
        if not store:
            return html.Div(
                "Cluster preview available after running analysis with --cluster.",
                style={"color": "#777"},
            )
        if not hover_data or not hover_data.get("points"):
            return placeholder

        point = hover_data["points"][0].get("customdata") or []
        if len(point) < 7:
            return placeholder

        neuron_id = point[0]
        cluster_label = point[1]
        top_class = point[2]
        top_score = point[3]
        strength = point[4]
        patch_label = point[5]
        patch_path = point[6]

        thumb = None
        if patch_path:
            try:
                thumb = encode_thumb(patch_path, max_side=240)
            except Exception:
                thumb = None

        details = html.Div(
            [
                html.Div(
                    [
                        html.Strong(f"Feature {neuron_id}"),
                        html.Span(f" • {cluster_label}", style={"marginLeft": "0.35rem"}),
                    ],
                    style={"fontSize": "1rem"},
                ),
                html.Div(f"Dominant class: {top_class} ({top_score})"),
                html.Div(f"Strength: {strength}"),
                html.Div(f"Patch label: {patch_label or 'unavailable'}"),
                html.Div(
                    patch_path,
                    style={"color": "#999", "fontSize": "0.75rem", "marginTop": "0.4rem"},
                ),
            ],
            style={"maxWidth": "420px"},
        )

        if thumb:
            image = html.Img(src=thumb, style={"maxHeight": "180px", "border": "1px solid #ccc"})
            return html.Div([image, details])
        return html.Div([details])


def _format_status(data_dict: Dict[str, object]) -> html.Div:
    algo = str(data_dict.get("algorithm", "unknown")).upper()
    params = data_dict.get("algorithm_params", {})
    split = data_dict.get("split", "train")
    pieces = [
        f"Algorithm: {algo}",
        f"Split: {split}",
        f"Features: {data_dict.get('num_features', '–')}",
    ]
    if params:
        readable = ", ".join(f"{k}={v}" for k, v in params.items() if k not in {"algorithm", "use_absolute"})
        if readable:
            pieces.append(readable)
    if params.get("use_absolute"):
        pieces.append("abs activations")
    return html.Div(" • ".join(pieces), style={"fontSize": "0.95rem", "color": "#333"})


def _build_cluster_figure(data_dict: Dict[str, object]) -> go.Figure:
    coords = np.asarray(data_dict.get("coords", []), dtype=float)
    if coords.size == 0:
        return _empty_figure("Cluster coordinates unavailable.")

    feature_indices = data_dict.get("feature_indices", [])
    cluster_ids = np.asarray(data_dict.get("cluster_ids", []), dtype=int)
    class_means = np.asarray(data_dict.get("class_means", []), dtype=float)
    class_names = data_dict.get("class_names", [])
    feature_strength = np.asarray(data_dict.get("feature_strength", []), dtype=float)
    representative = data_dict.get("representative_patches", {})
    cluster_summary = {int(entry["cluster_id"]): entry for entry in data_dict.get("cluster_summary", []) if isinstance(entry, dict)}

    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, max(0, 2 - coords.shape[1]))), mode="constant")

    if coords.shape[0] != len(feature_indices):
        return _empty_figure("Mismatch between coordinate data and feature count.")

    unique_clusters = sorted(set(int(c) for c in cluster_ids))
    palette = plotly_colors.qualitative.Plotly or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    custom_rows: List[List[object]] = []

    for idx, fid in enumerate(feature_indices):
        cluster_id = int(cluster_ids[idx])
        vector = class_means[idx] if idx < class_means.shape[0] else np.zeros(len(class_names))
        if vector.ndim == 0:
            vector = np.array([vector])
        top_idx = int(np.argmax(vector)) if vector.size else 0
        top_name = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"
        top_score = float(vector[top_idx]) if vector.size else 0.0
        strength = float(feature_strength[idx]) if idx < feature_strength.size else 0.0

        cluster_label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        summary_entry = cluster_summary.get(cluster_id)
        if summary_entry:
            cluster_label = f"{cluster_label} • {summary_entry.get('top_class_name', top_name)}"

        patches = representative.get(str(fid)) or []
        patch_label = "N/A"
        patch_activation = "—"
        patch_path = ""
        if patches:
            top_patch = patches[0]
            patch_label = top_patch.get("label_name", "Unknown")
            patch_activation = f"{top_patch.get('activation', 0.0):.3f}"
            patch_path = top_patch.get("image_path", "")

        custom_rows.append(
            [
                fid,
                cluster_label,
                top_name,
                f"{top_score:.3f}",
                f"{strength:.3f}",
                f"{patch_label} ({patch_activation})",
                patch_path,
            ]
        )

    fig = go.Figure()
    for idx, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        if not np.any(mask):
            continue
        color = palette[idx % len(palette)]
        subset = np.where(mask)[0]
        trace_custom = [custom_rows[i] for i in subset]
        name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        summary_entry = cluster_summary.get(cluster_id)
        if summary_entry:
            top_cls = summary_entry.get("top_class_name")
            name = f"{name} • {top_cls}"
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=name,
                customdata=trace_custom,
                hovertemplate=(
                    "<b>Feature %{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Dominant class: %{customdata[2]} (%{customdata[3]})<br>"
                    "Strength: %{customdata[4]}<br>"
                    "Patch: %{customdata[5]}<extra></extra>"
                ),
                marker=dict(size=12, color=color, line=dict(color="#1f1f1f", width=0.5)),
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def _build_summary(data_dict: Dict[str, object]) -> html.Div:
    summary = data_dict.get("cluster_summary", [])
    if not summary:
        return html.Div("Cluster summary unavailable for this model/split.", style={"color": "#666"})

    rows = []
    header = html.Tr(
        [
            html.Th("Cluster"),
            html.Th("Size"),
            html.Th("Top class"),
            html.Th("Score"),
        ]
    )
    for entry in sorted(summary, key=lambda e: e.get("cluster_id", 0)):
        cluster_id = entry.get("cluster_id", "-")
        label = "Noise" if cluster_id == -1 else f"{cluster_id}"
        rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(entry.get("size", "-")),
                    html.Td(entry.get("top_class_name", "—")),
                    html.Td(f"{entry.get('top_class_score', 0.0):.3f}"),
                ]
            )
        )

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "maxWidth": "680px",
        },
    )

    return html.Div(
        [
            html.H4("Cluster summary", style={"marginBottom": "0.5rem"}),
            table,
        ]
    )
