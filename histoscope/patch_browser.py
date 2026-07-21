"""Patch browser tab and callbacks for HistoSCOPE."""
from __future__ import annotations

import base64
import json
import os
import random
from functools import lru_cache
from io import BytesIO
from typing import Dict, List, Optional

import dash
import numpy as np
from dash import ALL, Input, Output, State, dcc, html
from PIL import Image
import plotly.graph_objects as go


# --------- Data loading / caching helpers ---------
def encode_thumb_local(path: str, max_side: int = 160) -> str:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4/5+hHgAGgwJ/lTgL1gAAAABJRU5ErkJggg=="


def _patch_activations_dir(model_dir: str) -> str:
    return os.path.join(model_dir, "analysis", "patch-activations")


@lru_cache(maxsize=16)
def load_feature_order(model_dir: str) -> List[str]:
    path = os.path.join(_patch_activations_dir(model_dir), "locked_feature_order.json")
    with open(path, "r") as f:
        data = json.load(f)
    feat_ids = data.get("locked_feature_indices", [])
    return [str(fid) for fid in feat_ids]


@lru_cache(maxsize=32)
def load_patch_activations(model_dir: str, split: str) -> np.ndarray:
    act_path = os.path.join(_patch_activations_dir(model_dir), f"{split}_locked_patch_activations.npy")
    return np.load(act_path)


@lru_cache(maxsize=32)
def load_patch_metadata(model_dir: str, split: str) -> List[Dict]:
    meta_path = os.path.join(_patch_activations_dir(model_dir), f"{split}_patch_metadata.jsonl")
    rows: List[Dict] = []
    if not os.path.exists(meta_path):
        return rows
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# --------- Layout (Tab) ---------
def patch_browser_tab() -> dcc.Tab:
    """Return the Patch Browser tab layout."""
    return dcc.Tab(
        label="Patch Browser",
        value="patch_browser_tab",
        children=[
            html.Div(
                [
                    dcc.Store(id="patch_selected_row", data=None),
                    dcc.Store(id="patch_sample_rows", data=None),
                    html.Div(
                        [
                            html.Label("Class"),
                            dcc.Dropdown(
                                id="patch_class_filter",
                                options=[{"label": "All classes", "value": "all"}],
                                value="all",
                                clearable=False,
                                style={"width": "240px"},
                            ),
                            html.Label("Max patches", style={"marginLeft": "1.25rem"}),
                            dcc.Dropdown(
                                id="patch_limit",
                                options=[{"label": str(v), "value": v} for v in [50, 100, 200, 400, 800]],
                                value=200,
                                clearable=False,
                                style={"width": "110px", "marginLeft": "0.4rem"},
                            ),
                            html.Label("Top-K features", style={"marginLeft": "1.25rem"}),
                            dcc.Dropdown(
                                id="patch_topk",
                                options=[{"label": str(v), "value": v} for v in [10, 20, 30, 50, 75, 100]]
                                + [{"label": "All", "value": -1}],
                                value=30,
                                clearable=False,
                                style={"width": "120px", "marginLeft": "0.4rem"},
                            ),
                            html.Div(
                                id="patch_selected_info",
                                style={"marginLeft": "1.5rem", "fontSize": "0.85rem", "color": "#555"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "alignItems": "center",
                            "gap": "6px",
                            "padding": "0.6rem 0.9rem",
                            "borderBottom": "1px solid #ddd",
                            "background": "#fafafa",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="patch_activation_hist",
                                figure=go.Figure(
                                    layout={
                                        "xaxis": {"title": "Feature"},
                                        "yaxis": {"title": "Activation"},
                                    }
                                ),
                                style={"height": "550px"},
                            )
                        ],
                        style={"padding": "0.75rem 1rem 0.25rem"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="patch_strip",
                                style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "gap": "8px",
                                    "padding": "0.55rem 0.75rem",
                                    "overflowX": "auto",
                                    "whiteSpace": "nowrap",
                                    "borderTop": "1px solid #ddd",
                                    "background": "#fff",
                                },
                            )
                        ],
                        style={"width": "100%", "boxSizing": "border-box"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "width": "100%"},
            )
        ],
    )


# --------- Callbacks ---------
def register_patch_browser_callbacks(app: dash.Dash, model_index: Dict[str, Dict]) -> None:
    def _get_model_dir(model_name: Optional[str]) -> Optional[str]:
        if not model_name or model_name not in model_index:
            return None
        return model_index[model_name]["dir"]

    @app.callback(
        Output("patch_sample_rows", "data"),
        Input("model_select", "value"),
        Input("split_select", "value"),
        Input("patch_class_filter", "value"),
        Input("patch_limit", "value"),
        prevent_initial_call=True,
    )
    def compute_sample(model_name, split, class_filter, limit):
        try:
            mdir = _get_model_dir(model_name)
            if not mdir:
                return []
            meta = load_patch_metadata(mdir, split)
            if not meta:
                return []
            rows = [r for r in meta if class_filter == "all" or r["class_label_name"] == class_filter]
            if not rows:
                return []
            random.shuffle(rows)
            rows = rows[: int(limit)]
            return [r["row_index"] for r in rows]
        except Exception:
            return []

    @app.callback(
        Output("patch_class_filter", "options"),
        Output("patch_class_filter", "value"),
        Input("model_select", "value"),
        Input("split_select", "value"),
        State("patch_class_filter", "value"),
        prevent_initial_call=True,
    )
    def update_patch_class_options(model_name, split, current):
        mdir = _get_model_dir(model_name)
        if not mdir:
            default_opts = [{"label": "All classes (0)", "value": "all"}]
            return default_opts, "all"
        meta = load_patch_metadata(mdir, split)
        counts: Dict[str, int] = {}
        for row in meta:
            cls = row.get("class_label_name", "Unknown")
            counts[cls] = counts.get(cls, 0) + 1
        opts = [
            {"label": f"All classes ({sum(counts.values())})", "value": "all"}
        ] + [
            {"label": f"{name} ({count})", "value": name}
            for name, count in sorted(counts.items(), key=lambda item: item[0])
        ]
        if current and (current == "all" or current in counts):
            return opts, current
        return opts, "all"

    @app.callback(
        Output("patch_strip", "children"),
        Input("patch_selected_row", "data"),
        Input("patch_sample_rows", "data"),
        State("model_select", "value"),
        State("split_select", "value"),
        prevent_initial_call=True,
    )
    def render_patch_strip(selected_row, sample_rows, model_name, split):
        mdir = _get_model_dir(model_name)
        if not mdir:
            return []
        try:
            meta = load_patch_metadata(mdir, split)
            sample_rows = sample_rows or []
            thumbs = []
            selected_set = {selected_row} if selected_row is not None else set()
            for rid in sample_rows:
                rec = next((r for r in meta if r["row_index"] == rid), None)
                if not rec:
                    continue
                img_path = rec.get("patch_path") or rec.get("image_path")
                if img_path and not os.path.isabs(img_path):
                    candidate = os.path.join(os.getcwd(), img_path)
                    if os.path.exists(candidate):
                        img_path = candidate
                thumb = encode_thumb_local(img_path or "")
                cap = f"{rec.get('class_label_name','?')}\n#{rec.get('patch_id','')}"
                border_style = "2px solid #ff9800" if rid in selected_set else "1px solid #ddd"
                shadow = "0 0 6px rgba(255,152,0,0.6)" if rid in selected_set else "0"
                thumbs.append(
                    html.Div(
                        [
                            html.Button(
                                html.Img(
                                    src=thumb,
                                    style={
                                        "width": "98px",
                                        "height": "98px",
                                        "objectFit": "cover",
                                        "borderRadius": "4px",
                                    },
                                ),
                                id={"type": "patch_thumb", "row": rid},
                                style={
                                    "border": border_style,
                                    "padding": "0",
                                    "cursor": "pointer",
                                    "background": "#fff",
                                    "borderRadius": "4px",
                                    "boxShadow": shadow,
                                },
                            ),
                            html.Div(
                                cap,
                                style={
                                    "fontSize": "0.62rem",
                                    "maxWidth": "96px",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "whiteSpace": "nowrap",
                                    "textAlign": "center",
                                    "marginTop": "2px",
                                    "color": "#444",
                                },
                            ),
                        ],
                        style={
                            "display": "inline-flex",
                            "flexDirection": "column",
                            "alignItems": "center",
                            "width": "102px",
                        },
                    )
                )
            return thumbs
        except Exception as exc:
            return html.Div(f"Error loading patches: {exc}", style={"color": "red", "padding": "0.5rem"})

    @app.callback(
        Output("patch_selected_row", "data"),
        Output("patch_selected_info", "children"),
        Input({"type": "patch_thumb", "row": ALL}, "n_clicks"),
        State("model_select", "value"),
        State("split_select", "value"),
        prevent_initial_call=True,
    )
    def set_selected_patch(all_clicks, model_name, split):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered[0]["prop_id"]
        try:
            trig_id = json.loads(trigger.split(".")[0])
            row_index = int(trig_id["row"])
        except Exception:
            raise dash.exceptions.PreventUpdate
        mdir = _get_model_dir(model_name)
        if not mdir:
            raise dash.exceptions.PreventUpdate
        meta = load_patch_metadata(mdir, split)
        rec = next((r for r in meta if r["row_index"] == row_index), None)
        if not rec:
            return dash.no_update, dash.no_update
        info = (
            f"Selected: {rec.get('patch_id','?')} | Class: {rec.get('class_label_name','?')} | Row: {row_index}"
        )
        return row_index, info

    @app.callback(
        Output("patch_activation_hist", "figure"),
        Input("patch_selected_row", "data"),
        Input("patch_topk", "value"),
        State("model_select", "value"),
        State("split_select", "value"),
        prevent_initial_call=True,
    )
    def show_patch_hist(row_index, topk, model_name, split):
        if row_index is None:
            raise dash.exceptions.PreventUpdate
        mdir = _get_model_dir(model_name)
        if not mdir:
            raise dash.exceptions.PreventUpdate
        try:
            acts = load_patch_activations(mdir, split)
            feat_ids = load_feature_order(mdir)
            if row_index < 0 or row_index >= acts.shape[0]:
                raise ValueError("Row out of range")
            vec = acts[row_index]
            feat_axis_full = [str(i) for i in range(len(vec))]
            if len(feat_ids) == len(vec):
                feat_axis_full = feat_ids
            meta = load_patch_metadata(mdir, split)
            rec = next((r for r in meta if r["row_index"] == row_index), None)
            patch_id = rec.get("patch_id", f"row {row_index}") if rec else f"row {row_index}"
            order = np.argsort(-vec)
            if isinstance(topk, int) and topk > 0 and topk != -1:
                order = order[: min(topk, len(order))]
            sorted_vals = vec[order]
            sorted_feats = [feat_axis_full[i] for i in order]
            fig = go.Figure(
                data=go.Bar(x=sorted_feats, y=sorted_vals, marker_color="#4a90e2"),
                layout=go.Layout(
                    xaxis=dict(title="Feature (sorted by activation)", tickangle=45),
                    yaxis=dict(title="Activation"),
                    margin=dict(l=55, r=10, t=48, b=120),
                    height=540,
                ),
            )
            fig.update_layout(title=f"{patch_id} activations")
            return fig
        except Exception as exc:
            return go.Figure(layout=go.Layout(title=f"Error: {exc}", height=540))

    app.callback_map = app.callback_map  # appease linters; dash registers via decorators


__all__ = ["patch_browser_tab", "register_patch_browser_callbacks"]
