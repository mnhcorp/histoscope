"""Heatmap tab layout and callbacks."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import dash
import numpy as np
from dash import ALL, Input, Output, State, callback_context, dcc, html
from plotly import graph_objects as go

from .. import data
from ..context import AppContext
from ..descriptions import describe_patch, is_description_cached, set_description_context
from ..utils import encode_original_size_image, encode_thumb, get_image_dimensions


def load_and_sort_features(model_name: str, split: str, sort_method: str = "default") -> List[str]:
    """Load features and optionally sort by monosemanticity or purity."""
    cache = data.load_cache(model_name, split)
    feat_labels = cache["feat_labels"]

    if sort_method == "default":
        return feat_labels

    locked_comparison = data.load_locked_feature_comparison(model_name)
    if (not locked_comparison) or ("feature_comparisons" not in locked_comparison):
        return feat_labels

    feature_data: Dict[str, Dict[str, float]] = {}
    for feature_info in locked_comparison.get("feature_comparisons", []):
        fid = str(feature_info.get("feature_index"))
        if data.is_baseline_model(model_name):
            if split == "train":
                purity_val = feature_info.get("purity_train_at_100", 0)
            else:
                purity_val = feature_info.get("purity_test_at_100", 0)
            auprc_val = float(feature_info.get("auprc_selectivity_score", 0.0))
        else:
            purity_val = feature_info.get(f"{split}_purity", 0)
            auprc_key = f"{split}_auprc_by_class"
            auprc_list = feature_info.get(auprc_key, [])
            if isinstance(auprc_list, list) and auprc_list:
                try:
                    auprc_val = float(max(float(x) for x in auprc_list))
                except (TypeError, ValueError):
                    auprc_val = 0.0
            else:
                auprc_val = 0.0
        feature_data[fid] = {
            "mono_score": feature_info.get("monosemanticity_score", 0),
            "mono_score_softmax": feature_info.get("monosemanticity_score_softmax", 0),
            "purity": purity_val,
            "ms_score": feature_info.get("MS_score", 0),
            "auprc": auprc_val,
        }

    locked_features: List[Tuple[str, Dict[str, float]]] = []
    unlocked_features: List[str] = []

    for feat_id in feat_labels:
        if feat_id in feature_data:
            locked_features.append((feat_id, feature_data[feat_id]))
        else:
            unlocked_features.append(feat_id)

    if sort_method == "mono_desc":
        locked_features.sort(key=lambda x: x[1]["mono_score"], reverse=True)
    elif sort_method == "mono_asc":
        locked_features.sort(key=lambda x: x[1]["mono_score"])
    elif sort_method == "mono_softmax_desc":
        locked_features.sort(key=lambda x: x[1]["mono_score_softmax"], reverse=True)
    elif sort_method == "mono_softmax_asc":
        locked_features.sort(key=lambda x: x[1]["mono_score_softmax"])
    elif sort_method == "purity_desc":
        locked_features.sort(key=lambda x: x[1]["purity"], reverse=True)
    elif sort_method == "purity_asc":
        locked_features.sort(key=lambda x: x[1]["purity"])
    elif sort_method == "ms_desc":
        locked_features.sort(key=lambda x: x[1]["ms_score"], reverse=True)
    elif sort_method == "ms_asc":
        locked_features.sort(key=lambda x: x[1]["ms_score"])
    elif sort_method == "auprc_desc":
        locked_features.sort(key=lambda x: x[1]["auprc"], reverse=True)
    elif sort_method == "auprc_asc":
        locked_features.sort(key=lambda x: x[1]["auprc"])

    sorted_feat_labels = [feat_id for feat_id, _ in locked_features] + unlocked_features
    return sorted_feat_labels


def make_heatmap_fig(
    H: np.ndarray,
    feat_labels: List[str],
    class_labels: List[str],
    colorscale: str = "Viridis",
    selected: Optional[Tuple[str, str]] = None,
):
    x_indices = list(range(len(feat_labels)))
    y_indices = list(range(len(class_labels)))

    customdata: List[List[List[str]]] = []
    for ci, c_name in enumerate(class_labels):
        row: List[List[str]] = []
        for fi, f_name in enumerate(feat_labels):
            row.append([c_name, f_name])
        customdata.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=H,
            x=x_indices,
            y=y_indices,
            customdata=customdata,
            colorscale=colorscale,
            colorbar=dict(title="Mean activation"),
            zmid=float(np.median(H)),
            zmin=float(np.percentile(H, 5)),
            zmax=float(np.percentile(H, 95)),
            hovertemplate="Class: %{customdata[0]}<br>Feature: %{customdata[1]}<br>Activation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=80, r=10, t=10, b=100),
        xaxis=dict(
            title="Feature",
            tickmode="array",
            tickvals=x_indices,
            ticktext=feat_labels,
            tickangle=-45,
        ),
        yaxis=dict(
            title="Class",
            tickmode="array",
            tickvals=y_indices,
            ticktext=class_labels,
            autorange="reversed",
        ),
        clickmode="event+select",
    )

    if selected:
        sel_class, sel_feat = selected
        if sel_feat in feat_labels and sel_class in class_labels:
            fi = feat_labels.index(sel_feat)
            ci = class_labels.index(sel_class)
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=fi - 0.5,
                x1=fi + 0.5,
                y0=ci - 0.5,
                y1=ci + 0.5,
                line=dict(color="cyan", width=3),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

    return fig


def _render_model_metadata(model_name: Optional[str]) -> object:
    if not model_name or model_name not in data.MODEL_INDEX:
        return "Select a model to see metadata."
    if data.is_baseline_model(model_name):
        base_dir = data.get_model_dir(model_name)
        cmp_path = os.path.join(base_dir, "analysis", "baseline_feature_comparison.json")
        classification_counts = None
        classification_counts_auprc = None
        purity_stats = None
        if os.path.exists(cmp_path):
            try:
                with open(cmp_path, "r") as f:
                    cmp_data = json.load(f)
                classification_counts = cmp_data.get("summary_stats", {}).get("classification_counts", {})
                classification_counts_auprc = cmp_data.get("summary_stats", {}).get("classification_counts_auprc", {})
                purity_stats = cmp_data.get("summary_stats", {}).get("purity_at_100", {})
            except Exception as exc:
                print(f"Failed loading baseline comparison: {exc}")
        lines = ["Baseline UNI activations (raw 1024-dim embedding)"]
        if classification_counts:
            total = sum(classification_counts.values()) or 1
            lines.append("\nHeuristic classification counts:")
            for key in ["monosemantic", "nearly-mono", "polysemantic", "dead"]:
                if key in classification_counts:
                    value = classification_counts[key]
                    pct = 100.0 * value / total
                    lines.append(f"  {key:13s}: {value:4d} ({pct:5.1f}%)")
        if classification_counts_auprc:
            total = sum(classification_counts_auprc.values()) or 1
            lines.append("\nAUPRC classification counts:")
            for key in ["monosemantic", "nearly-mono", "polysemantic"]:
                if key in classification_counts_auprc:
                    value = classification_counts_auprc[key]
                    pct = 100.0 * value / total
                    lines.append(f"  {key:13s}: {value:4d} ({pct:5.1f}%)")
        if purity_stats:
            lines.append("\nPurity@100 summary (train/test):")
            lines.append(
                f"  train mean/median: {purity_stats.get('train_mean',0):.3f} / {purity_stats.get('train_median',0):.3f}"
            )
            lines.append(
                f"  test  mean/median: {purity_stats.get('test_mean',0):.3f} / {purity_stats.get('test_median',0):.3f}"
            )
        return html.Pre(
            "\n".join(lines),
            style={"fontFamily": "monospace", "fontSize": "0.9rem", "whiteSpace": "pre-wrap"},
        )

    meta = data.load_metadata(model_name)
    hp = meta.get("hyperparameters", {})
    ds = meta.get("dataset_args", {})
    fm = meta.get("final_metrics", {})

    mdir = data.get_model_dir(model_name)
    dead_analysis_path = os.path.join(mdir, "analysis", "dead_feature_analysis.json")
    dead_info = None
    if os.path.exists(dead_analysis_path):
        try:
            with open(dead_analysis_path, "r") as f:
                dead_analysis = json.load(f)
            dead_info = {
                "dead_percentage": dead_analysis.get("dead_features", {}).get("percentage", 0),
                "near_dead_percentage": dead_analysis.get("near_dead_features", {}).get("percentage", 0),
                "active_percentage": dead_analysis.get("active_features", {}).get("percentage", 0),
                "coverage_correlation": dead_analysis.get("robustness_metrics", {}).get("coverage_correlation", 0),
                "split_consistency": dead_analysis.get("robustness_metrics", {}).get("split_consistency_good", False),
                "tau_threshold": dead_analysis.get("tau_threshold", 1e-5),
            }
        except Exception as exc:
            print(f"Error loading dead feature analysis: {exc}")

    classification_counts = None
    classification_counts_auprc = None
    locked_cmp_path = os.path.join(mdir, "analysis", "locked_feature_comparison.json")
    if os.path.exists(locked_cmp_path):
        try:
            with open(locked_cmp_path, "r") as f:
                cmp_data = json.load(f)
            classification_counts = cmp_data.get("summary_stats", {}).get("classification_counts", None)
            classification_counts_auprc = cmp_data.get("summary_stats", {}).get("classification_counts_auprc", None)
        except Exception as exc:
            print(f"Error loading locked feature comparison: {exc}")

    def fmt(val):
        if isinstance(val, float):
            return f"{val:.6f}"
        return str(val)

    lines = [
        "Hyperparameters:",
        f"  input_dim: {hp.get('input_dim')}",
        f"  hidden_dim: {hp.get('hidden_dim')}",
        f"  lambda_l1: {hp.get('lambda_l1')}",
        f"  learning_rate: {hp.get('learning_rate')}",
        f"  num_epochs: {hp.get('num_epochs')}",
        f"  batch_size: {hp.get('batch_size')}",
        "",
        "Dataset args:",
        f"  emb_dir: {ds.get('emb_dir')}",
        f"  l2_normalize: {ds.get('l2_normalize')}",
        f"  compute_stats: {ds.get('compute_stats')}",
        f"  stats_path: {ds.get('stats_path')}",
        "",
        "Final metrics:",
        f"  final_loss: {fmt(fm.get('final_loss'))}",
        f"  recon_loss: {fmt(fm.get('final_reconstruction_loss'))}",
        f"  sparsity_loss: {fmt(fm.get('final_sparsity_loss'))}",
        f"  avg_active_features: {fm.get('avg_active_features')}",
    ]

    if dead_info:
        lines.extend(
            [
                "",
                "Feature activity analysis:",
                f"  Dead features: {dead_info['dead_percentage']:.1f}%",
                f"  Near-dead features: {dead_info['near_dead_percentage']:.1f}%",
                f"  Threshold (τ): {dead_info['tau_threshold']:.0e}",
            ]
        )

    if classification_counts_auprc:
        total = sum(classification_counts_auprc.values()) or 1
        lines.extend(
            [
                "",
                "AUPRC classification:",
                f"  monosemantic: {classification_counts_auprc.get('monosemantic', 0)} ({100.0 * classification_counts_auprc.get('monosemantic', 0) / total:5.1f}%)",
                f"  nearly-mono: {classification_counts_auprc.get('nearly-mono', 0)} ({100.0 * classification_counts_auprc.get('nearly-mono', 0) / total:5.1f}%)",
                f"  polysemantic: {classification_counts_auprc.get('polysemantic', 0)} ({100.0 * classification_counts_auprc.get('polysemantic', 0) / total:5.1f}%)",
            ]
        )

    return html.Pre("\n".join([str(x) for x in lines]))


def _clean_class_labels(class_labels: List[str]) -> List[str]:
    cleaned: List[str] = []
    for label in class_labels:
        if isinstance(label, str) and " (n=" in label:
            cleaned.append(label.split(" (n=")[0])
        else:
            cleaned.append(label)
    return cleaned


def make_distribution_fig(
    values: np.ndarray,
    class_labels: List[str],
    feat_id: str,
    top_k: int = 30,
) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=10, t=20, b=80),
        hovermode="x",
        showlegend=False,
    )

    if values.size == 0:
        fig.add_annotation(
            text="No activations for this feature.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12),
        )
        fig.update_xaxes(title="Class", tickangle=-35, automargin=True)
        fig.update_yaxes(title="Mean activation", rangemode="tozero")
        return fig

    clean_labels = _clean_class_labels(class_labels)
    activations = np.asarray(values, dtype=float)
    keep_mask = np.isfinite(activations)
    clean_labels = [lbl for lbl, keep in zip(clean_labels, keep_mask) if keep]
    activations = activations[keep_mask]

    if activations.size == 0:
        fig.add_annotation(
            text="No valid activations.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12),
        )
        fig.update_xaxes(title="Class", tickangle=-35, automargin=True)
        fig.update_yaxes(title="Mean activation", rangemode="tozero")
        return fig

    paired = list(zip(clean_labels, activations.tolist()))
    paired.sort(key=lambda item: item[1], reverse=True)
    if top_k and len(paired) > top_k:
        paired = paired[:top_k]

    labels_sorted = [p[0] for p in paired]
    values_sorted = np.array([p[1] for p in paired], dtype=float)

    display_labels = [f"C{i}" for i in range(1, len(labels_sorted) + 1)]

    fig.add_trace(
        go.Bar(
            x=display_labels,
            y=values_sorted,
            marker=dict(color="#2b6cb0"),
            opacity=0.85,
            name="Mean activation",
            customdata=np.array(labels_sorted, dtype=object),
            hovertemplate="Class: %{customdata}<br>Activation: %{y:.3f}<extra></extra>",
        )
    )

    if values_sorted.size > 1:
        window = max(3, min(9, values_sorted.size if values_sorted.size % 2 == 1 else values_sorted.size - 1))
        window = max(3, window)
        if window % 2 == 0:
            window += 1
        half = window // 2
        kernel_idx = np.arange(-half, half + 1)
        sigma = max(1.0, window / 3.0)
        weights = np.exp(-0.5 * (kernel_idx / sigma) ** 2)
        weights /= weights.sum()
        padded = np.pad(values_sorted, (half, half), mode="edge")
        smoothed = np.convolve(padded, weights, mode="same")[half:-half]
        fig.add_trace(
            go.Scatter(
                x=display_labels,
                y=smoothed,
                mode="lines",
                line=dict(color="#c53030", width=2),
                name="KDE",
                hovertemplate="Smoothed activation: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_xaxes(title="Class", tickangle=-35, automargin=True)
    fig.update_yaxes(title="Mean activation", rangemode="tozero")
    return fig


def _baseline_dropdown_state(model_name: Optional[str], context: AppContext) -> Tuple[list[dict], int, dict, dict]:
    hidden_style = {"width": "160px", "display": "none", "marginLeft": "0.5rem"}
    hidden_label = {"marginLeft": "2rem", "display": "none"}
    if not model_name or not data.is_baseline_model(model_name):
        return [], 1, hidden_style, hidden_label
    page_size = context.page_size
    try:
        split0 = context.model_index[model_name]["splits"][0]
        cache0 = data.load_cache(model_name, split0)
        total_feats = cache0["H"].shape[1]
    except Exception:
        total_feats = 1024
    pages = int(np.ceil(total_feats / page_size))
    options = [
        {
            "label": f"Features {i * page_size + 1}-{min((i + 1) * page_size, total_feats)}",
            "value": i + 1,
        }
        for i in range(pages)
    ]
    visible_style = {"width": "260px", "display": "inline-block", "marginLeft": "0.5rem"}
    label_style = {"marginLeft": "2rem", "display": "inline-block"}
    return options, 1, visible_style, label_style


def _sae_pagination_state(model_name: Optional[str], split: str, sort_method: str, neurons_per_page: int, context: AppContext) -> Tuple[list[dict], int, dict, dict]:
    """Determine if SAE model needs pagination and return appropriate state."""
    hidden_style = {"width": "200px", "display": "none", "marginLeft": "0.5rem"}
    hidden_label = {"marginLeft": "2rem", "display": "none"}

    # Only show for non-baseline models
    if not model_name or data.is_baseline_model(model_name):
        return [], 1, hidden_style, hidden_label

    try:
        # Get sorted features to determine total count
        sorted_feat_labels = load_and_sort_features(model_name, split, sort_method)
        total_feats = len(sorted_feat_labels)

        # Only show pagination if total features exceed neurons_per_page
        if total_feats <= neurons_per_page:
            return [], 1, hidden_style, hidden_label

        # Calculate number of pages
        pages = int(np.ceil(total_feats / neurons_per_page))
        options = [
            {
                "label": f"Features {i * neurons_per_page + 1}-{min((i + 1) * neurons_per_page, total_feats)}",
                "value": i + 1,
            }
            for i in range(pages)
        ]

        visible_style = {"width": "200px", "display": "inline-block", "marginLeft": "0.5rem"}
        label_style = {"marginLeft": "2rem", "display": "inline-block"}
        return options, 1, visible_style, label_style

    except Exception as exc:
        print(f"Error in _sae_pagination_state: {exc}")
        return [], 1, hidden_style, hidden_label


def layout(context: AppContext) -> dcc.Tab:
    H0 = context.initial_cache["H"]
    feat_labels0 = context.initial_cache["feat_labels"]
    class_labels0 = context.initial_cache["class_labels"]
    min_feats, default_feats, _ = context.slider_bounds

    initial_fig = make_heatmap_fig(
        H0[:, :default_feats],
        feat_labels0[:default_feats],
        class_labels0,
    )

    num_feat_options = [opt for opt in [10, 20, 30, 40, 50] if opt <= H0.shape[1]]
    num_feat_dropdown = [{"label": str(v), "value": v} for v in num_feat_options]

    baseline_options, baseline_value, baseline_style, baseline_label_style = _baseline_dropdown_state(
        context.default_model, context
    )

    # Initialize SAE pagination state
    sae_options, sae_value, sae_style, sae_label_style = _sae_pagination_state(
        context.default_model, context.default_split, "default", default_feats, context
    )

    return dcc.Tab(
        label="Feature Heatmap",
        value="heatmap_tab",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Model"),
                                    dcc.Dropdown(
                                        id="model_select",
                                        options=context.initial_model_options,
                                        value=context.default_model,
                                        clearable=False,
                                        style={"minWidth": "400px"},
                                    ),
                                    html.Label("Split", style={"marginLeft": "1rem"}),
                                    dcc.Dropdown(
                                        id="split_select",
                                        options=[{"label": s, "value": s} for s in context.model_index[context.default_model]["splits"]],
                                        value=context.default_split,
                                        clearable=False,
                                        style={"width": "140px", "display": "inline-block", "marginLeft": "0.5rem"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "12px",
                                    "alignItems": "center",
                                    "padding": "0.5rem 1rem",
                                },
                            ),
                            html.Div(
                                [
                                    html.Label("Number of features"),
                                    dcc.Dropdown(
                                        id="num_feats",
                                        options=num_feat_dropdown,
                                        value=min(num_feat_options[-1], default_feats) if num_feat_options else default_feats,
                                        clearable=False,
                                        style={"width": "120px"},
                                    ),
                                    html.Label("Colormap", style={"marginLeft": "2rem"}),
                                    dcc.Dropdown(
                                        id="colormap_select",
                                        options=[
                                            {"label": "Viridis (Default)", "value": "Viridis"},
                                            {"label": "Plasma", "value": "Plasma"},
                                            {"label": "Inferno", "value": "Inferno"},
                                            {"label": "Hot", "value": "Hot"},
                                            {"label": "Cividis", "value": "Cividis"},
                                        ],
                                        value="Viridis",
                                        clearable=False,
                                        style={"width": "200px", "display": "inline-block", "marginLeft": "0.5rem"},
                                    ),
                                    html.Label("Sort by", style={"marginLeft": "2rem"}),
                                    dcc.Dropdown(
                                        id="feature_sort",
                                        options=[
                                            {"label": "Default (Feature ID)", "value": "default"},
                                            {"label": "Monosemanticity (High to Low)", "value": "mono_desc"},
                                            {"label": "Monosemanticity (Low to High)", "value": "mono_asc"},
                                            {"label": "Monosemanticity Softmax (High to Low)", "value": "mono_softmax_desc"},
                                            {"label": "Monosemanticity Softmax (Low to High)", "value": "mono_softmax_asc"},
                                            {"label": "Purity (High to Low)", "value": "purity_desc"},
                                            {"label": "Purity (Low to High)", "value": "purity_asc"},
                                            {"label": "MS Score (High to Low)", "value": "ms_desc"},
                                            {"label": "MS Score (Low to High)", "value": "ms_asc"},
                                            {"label": "AUPRC (High to Low)", "value": "auprc_desc"},
                                            {"label": "AUPRC (Low to High)", "value": "auprc_asc"},
                                        ],
                                        value="default",
                                        clearable=False,
                                        style={"width": "280px", "display": "inline-block", "marginLeft": "0.5rem"},
                                    ),
                                    html.Label(
                                        "Baseline page",
                                        id="baseline_page_label",
                                        style=baseline_label_style,
                                    ),
                                    dcc.Dropdown(
                                        id="baseline_page",
                                        options=baseline_options,
                                        value=baseline_value,
                                        clearable=False,
                                        style=baseline_style,
                                    ),
                                    html.Label(
                                        "SAE page",
                                        id="sae_page_label",
                                        style=sae_label_style,
                                    ),
                                    dcc.Dropdown(
                                        id="sae_page",
                                        options=sae_options,
                                        value=sae_value,
                                        clearable=False,
                                        style=sae_style,
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "12px",
                                    "alignItems": "center",
                                    "padding": "0.5rem 1rem",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="heatmap",
                                        figure=initial_fig,
                                        clear_on_unhover=True,
                                    )
                                ]
                            ),
                            html.Label("Montage size", style={"marginLeft": "2rem"}),
                            dcc.Dropdown(
                                id="montage_k",
                                options=[{"label": str(v), "value": v} for v in [2, 5, 10]],
                                value=5,
                                clearable=False,
                                style={"width": "120px", "marginLeft": "0.5rem"},
                    ),
                    html.Div(
                        id="montage",
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "padding": "0.5rem 1rem"},
                    ),
                ],
                style={"flex": "1 1 80%"},
            ),
            html.Div(
                [
                    dcc.Tabs(
                        id="sidebar_tabs",
                        value="model",
                        colors={"border": "#1f2933", "primary": "#1a365d", "background": "#e0e6ed"},
                        children=[
                            dcc.Tab(
                                label="Model",
                                value="model",
                                children=[
                                    html.Div(
                                        id="model_meta",
                                        children=_render_model_metadata(context.default_model),
                                        style={"fontSize": "0.9rem", "whiteSpace": "pre-wrap", "padding": "0.75rem"},
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Feature",
                                value="feature",
                                children=[
                                    dcc.Store(
                                        id="feature_notes_store",
                                        data={
                                            "model": context.default_model,
                                            "notes": {},
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                id="feature_meta",
                                                style={
                                                    "fontSize": "0.9rem",
                                                    "whiteSpace": "pre-wrap",
                                                    "padding": "0.75rem",
                                                    "border": "1px solid #e2e8f0",
                                                    "borderRadius": "6px",
                                                    "backgroundColor": "#fff",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Activation distribution",
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "0.25rem",
                                                        },
                                                    ),
                                                    dcc.Graph(
                                                        id="feature_distribution",
                                                        figure=go.Figure(),
                                                        config={"displayModeBar": False},
                                                        style={
                                                            "height": "220px",
                                                            "padding": "0.25rem",
                                                            "border": "1px solid #e2e8f0",
                                                            "borderRadius": "6px",
                                                            "backgroundColor": "#fff",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                    "padding": "0.75rem",
                                                    "backgroundColor": "#f7fafc",
                                                    "borderRadius": "6px",
                                                    "border": "1px solid #e2e8f0",
                                                    "gap": "0.25rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Feature notes",
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "0.25rem",
                                                        },
                                                    ),
                                                    dcc.Textarea(
                                                        id="feature_notes",
                                                        value="",
                                                        rows=10,
                                                        style={
                                                            "width": "100%",
                                                            "resize": "vertical",
                                                            "fontSize": "0.9rem",
                                                            "lineHeight": "1.35",
                                                            "padding": "0.5rem",
                                                            "border": "1px solid #bbb",
                                                            "borderRadius": "4px",
                                                            "minHeight": "10em",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "Save",
                                                                id="feature_notes_save",
                                                                n_clicks=0,
                                                                style={
                                                                    "padding": "0.35rem 0.9rem",
                                                                    "borderRadius": "4px",
                                                                    "border": "1px solid #2b6cb0",
                                                                    "backgroundColor": "#2b6cb0",
                                                                    "color": "white",
                                                                    "cursor": "pointer",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "Clear",
                                                                id="feature_notes_clear",
                                                                n_clicks=0,
                                                                style={
                                                                    "padding": "0.35rem 0.9rem",
                                                                    "borderRadius": "4px",
                                                                    "border": "1px solid #c53030",
                                                                    "backgroundColor": "#fff",
                                                                    "color": "#c53030",
                                                                    "cursor": "pointer",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "gap": "0.5rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="feature_notes_status",
                                                        style={
                                                            "fontSize": "0.8rem",
                                                            "color": "#4a5568",
                                                            "marginTop": "0.5rem",
                                                            "minHeight": "1.1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flexDirection": "column",
                                                    "padding": "0.75rem",
                                                    "backgroundColor": "#f7fafc",
                                                    "borderRadius": "6px",
                                                    "border": "1px solid #e2e8f0",
                                                    "gap": "0.25rem",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "0.75rem",
                                            "padding": "0.75rem",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
                style={"flex": "0 0 22%", "borderLeft": "1px solid #ddd", "maxWidth": "420px", "maxHeight": "75vh", "overflowY": "auto"},
            ),
        ],
        style={"display": "flex", "gap": "0"},
    )
        ],
    )


def process_line_formatting(text: str) -> List:
    if not text:
        return []
    out: List = []
    import re

    pattern = r"\*\*(.*?)\*\*"
    last = 0
    for match in re.finditer(pattern, text):
        if match.start() > last:
            out.append(text[last : match.start()])
        out.append(html.Strong(match.group(1)))
        last = match.end()
    if last < len(text):
        out.append(text[last:])
    return out


def format_description(description: str):
    if not isinstance(description, str):
        return html.Div("No description.")

    lines = description.split("\n")
    blocks: List = []

    for line in lines:
        if not line.strip():
            blocks.append(html.Br())
            continue
        if line.startswith("Gist:"):
            gist_children = [html.Strong("Gist: ")] + process_line_formatting(line[5:].strip())
            blocks.append(html.Div(gist_children, style={"marginBottom": "0.5rem"}))
            continue
        if line.startswith("Elements:"):
            blocks.append(
                html.Div([html.Strong("Elements:")], style={"marginTop": "0.5rem", "marginBottom": "0.25rem"})
            )
            continue
        if line.startswith("- "):
            bullet = process_line_formatting(line[2:].strip())
            bullet_children = [html.Span("• ", style={"marginRight": "0.25rem"})] + bullet
            blocks.append(
                html.Div(
                    bullet_children,
                    style={"marginLeft": "1rem", "marginBottom": "0.25rem"},
                )
            )
            continue
        regular = process_line_formatting(line)
        blocks.append(html.Div(regular))

    return html.Div(blocks)


def register_callbacks(app: dash.Dash, context: AppContext) -> None:
    page_size = context.page_size

    @app.callback(
        Output("split_select", "options"),
        Output("split_select", "value"),
        Input("model_select", "value"),
        State("split_select", "value"),
    )
    def update_split_options(model_name, current_split):
        if not model_name or model_name not in context.model_index:
            raise dash.exceptions.PreventUpdate
        splits = context.model_index[model_name]["splits"]
        opts = [{"label": s, "value": s} for s in splits]
        if current_split in splits:
            value = current_split
        else:
            value = "train" if "train" in splits else splits[0]
        return opts, value

    @app.callback(
        Output("num_feats", "options"),
        Output("num_feats", "value"),
        Input("model_select", "value"),
        Input("split_select", "value"),
        State("num_feats", "value"),
    )
    def update_num_feats_bounds(model_name, split, current):
        if not model_name or model_name not in context.model_index:
            raise dash.exceptions.PreventUpdate

        cache = data.load_cache(model_name, split)
        H = cache["H"]
        min_f, def_f, max_f = data.slider_bounds(H)

        all_options = [10, 20, 30, 40, 50]
        valid_options = [opt for opt in all_options if opt <= max_f]
        dropdown_options = [{"label": str(opt), "value": opt} for opt in valid_options]

        val = int(max(min_f, min((current or def_f), max_f)))
        if val not in valid_options and valid_options:
            val = valid_options[-1]
        return dropdown_options, val

    @app.callback(
        Output("baseline_page", "options"),
        Output("baseline_page", "value"),
        Output("baseline_page", "style"),
        Output("baseline_page_label", "style"),
        Input("model_select", "value"),
    )
    def configure_baseline_pages(model_name):
        return _baseline_dropdown_state(model_name, context)

    @app.callback(
        Output("sae_page", "options"),
        Output("sae_page", "value"),
        Output("sae_page", "style"),
        Output("sae_page_label", "style"),
        Input("model_select", "value"),
        Input("split_select", "value"),
        Input("feature_sort", "value"),
        Input("num_feats", "value"),
    )
    def configure_sae_pages(model_name, split, sort_method, neurons_per_page):
        if not model_name or not split:
            hidden_style = {"width": "200px", "display": "none", "marginLeft": "0.5rem"}
            hidden_label = {"marginLeft": "2rem", "display": "none"}
            return [], 1, hidden_style, hidden_label
        return _sae_pagination_state(model_name, split, sort_method, neurons_per_page or 50, context)

    @app.callback(
        Output("selected_model_store", "data"),
        [
            Input("model_select", "value"),
            Input("model_select_tab2", "value"),
            Input("model_select_tab3", "value"),
            Input("model_select_tab4", "value"),
            Input("dataset_select", "value"),
        ],
        State("selected_model_store", "data"),
        prevent_initial_call=True,
    )
    def update_selected_model_store(tab1_value, tab2_value, tab3_value, tab4_value, dataset_value, current):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        trigger = ctx.triggered[0]
        prop_id = trigger.get("prop_id", "")
        if prop_id == "dataset_select.value":
            models = context.dataset_models.get(dataset_value, [])
            if not models:
                return dash.no_update
            default_model = context.default_model_by_dataset.get(dataset_value, models[0])
            if current == default_model:
                return dash.no_update
            return default_model

        triggered_value = trigger.get("value")
        if triggered_value is None:
            return dash.no_update

        valid_models = context.dataset_models.get(dataset_value, [])
        if triggered_value not in valid_models or triggered_value == current:
            return dash.no_update
        return triggered_value

    @app.callback(
        Output("model_select", "options"),
        Output("model_select_tab2", "options"),
        Output("model_select_tab3", "options"),
        Output("model_select_tab4", "options"),
        Input("dataset_select", "value"),
    )
    def update_model_dropdown_options(dataset_value):
        models = context.dataset_models.get(dataset_value, [])
        options = [{"label": data.model_display_label(m), "value": m} for m in models]
        return options, options, options, options

    @app.callback(
        Output("model_select", "value"),
        Output("model_select_tab2", "value"),
        Output("model_select_tab3", "value"),
        Output("model_select_tab4", "value"),
        Input("selected_model_store", "data"),
        State("model_select", "value"),
        State("model_select_tab2", "value"),
        State("model_select_tab3", "value"),
        State("model_select_tab4", "value"),
    )
    def sync_model_dropdowns(selected_model, tab1_value, tab2_value, tab3_value, tab4_value):
        if not selected_model or selected_model not in context.model_index:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        tab1_update = selected_model if tab1_value != selected_model else dash.no_update
        tab2_update = selected_model if tab2_value != selected_model else dash.no_update
        tab3_update = selected_model if tab3_value != selected_model else dash.no_update
        tab4_update = selected_model if tab4_value != selected_model else dash.no_update
        return tab1_update, tab2_update, tab3_update, tab4_update

    @app.callback(
        Output("model_meta", "children"),
        Input("model_select", "value"),
    )
    def update_model_meta(model_name):
        return _render_model_metadata(model_name)

    @app.callback(
        Output("feature_meta", "children"),
        Output("feature_distribution", "figure"),
        Input("selected_cell", "data"),
        Input("model_select", "value"),
        Input("split_select", "value"),
    )
    def update_feature_meta(selected_cell, model_name, split):
        def blank_distribution(message: str) -> go.Figure:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=40, r=10, t=30, b=40),
                bargap=0.05,
                showlegend=False,
            )
            fig.update_xaxes(title="Mean activation")
            fig.update_yaxes(title="Class count", rangemode="tozero")
            fig.add_annotation(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12),
            )
            return fig

        if not model_name or model_name not in context.model_index:
            return (
                html.Div("Select a model to see feature details."),
                blank_distribution("Select a model to view distribution."),
            )
        if not selected_cell or not isinstance(selected_cell, dict):
            return (
                html.Div("Click a heatmap cell to view feature details."),
                blank_distribution("Click a heatmap cell to view distribution."),
            )

        feat_id = selected_cell.get("feat")
        class_label = selected_cell.get("class")
        if feat_id is None or class_label is None:
            return (
                html.Div("Click a heatmap cell to view feature details."),
                blank_distribution("Click a heatmap cell to view distribution."),
            )

        try:
            locked_cmp = data.load_locked_feature_comparison(model_name)
        except Exception as exc:
            return (
                html.Div(f"Failed to load feature comparison data: {exc}"),
                blank_distribution("Unable to load distribution."),
            )

        feature_info = None
        if locked_cmp and "feature_comparisons" in locked_cmp:
            for fc in locked_cmp["feature_comparisons"]:
                if str(fc.get("feature_index")) == str(feat_id):
                    feature_info = fc
                    break
        if not feature_info:
            try:
                cache = data.load_cache(model_name, split)
                feat_labels = cache.get("feat_labels", [])
                class_labels = cache.get("class_labels", [])
                H = cache.get("H")
                if isinstance(H, np.ndarray) and str(feat_id) in feat_labels:
                    feat_idx = feat_labels.index(str(feat_id))
                    dist_fig = make_distribution_fig(H[:, feat_idx], class_labels, str(feat_id))
                else:
                    dist_fig = blank_distribution("Distribution unavailable.")
            except Exception:
                dist_fig = blank_distribution("Distribution unavailable.")
            return (
                html.Div("No detailed metrics available for this feature."),
                dist_fig,
            )

        cache = data.load_cache(model_name, split)
        class_labels = cache.get("class_labels", [])
        feat_labels = cache.get("feat_labels", [])
        H = cache.get("H")
        if isinstance(H, np.ndarray) and str(feat_id) in feat_labels:
            feat_idx = feat_labels.index(str(feat_id))
            distribution_fig = make_distribution_fig(H[:, feat_idx], class_labels, str(feat_id))
        else:
            distribution_fig = blank_distribution("Distribution unavailable.")
        clean_class = class_label.split(" (n=")[0] if isinstance(class_label, str) else class_label
        try:
            class_index = class_labels.index(clean_class)
        except ValueError:
            class_index = None

        def fmt(value, digits=4):
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            return str(value)

        lines = [f"Feature #{feat_id}"]
        lines.append(f"Selected class: {clean_class}")

        lines.append("")
        lines.append("Heuristic classification:")
        lines.append(f"  margin label: {feature_info.get('classification_label', 'n/a')}")
        lines.append(
            f"  AUPRC label: {feature_info.get('classification_label_auprc', 'n/a')}"
        )

        lines.append("")
        lines.append("Margins & purity:")
        lines.append(
            f"  margin (train/test): {fmt(feature_info.get('train_margin'))} / {fmt(feature_info.get('test_margin'))}"
        )
        lines.append(
            f"  purity (train/test): {fmt(feature_info.get('train_purity'))} / {fmt(feature_info.get('test_purity'))}"
        )
        lines.append(
            f"  coverage (train/test): {fmt(feature_info.get('train_coverage'))} / {fmt(feature_info.get('test_coverage'))}"
        )

        lines.append("")
        lines.append("AUPRC selectivity:")
        lines.append(
            f"  best score: {fmt(feature_info.get('auprc_selectivity_score'))}"
        )
        lines.append(
            f"  gap vs second best: {fmt(feature_info.get('auprc_selectivity_gap'))}"
        )
        lines.append(
            f"  best class (train/test min): {feature_info.get('auprc_selectivity_class_name', 'n/a')}"
        )

        if class_index is not None:
            train_recall = feature_info.get('train_recall_by_class') or []
            test_recall = feature_info.get('test_recall_by_class') or []
            train_precision = feature_info.get('train_precision_at_threshold_by_class') or []
            test_precision = feature_info.get('test_precision_at_threshold_by_class') or []
            train_auprc = feature_info.get('train_auprc_by_class') or []
            test_auprc = feature_info.get('test_auprc_by_class') or []
            threshold = feature_info.get('recall_threshold')

            lines.append("")
            lines.append(f"Class metrics for '{clean_class}':")
            if threshold is not None:
                lines.append(f"  threshold: {fmt(threshold)}")
            if class_index < len(train_recall) and class_index < len(test_recall):
                lines.append(
                    f"  recall (train/test): {fmt(train_recall[class_index])} / {fmt(test_recall[class_index])}"
                )
            if class_index < len(train_precision) and class_index < len(test_precision):
                lines.append(
                    f"  precision@τ (train/test): {fmt(train_precision[class_index])} / {fmt(test_precision[class_index])}"
                )
            if class_index < len(train_auprc) and class_index < len(test_auprc):
                lines.append(
                    f"  AUPRC (train/test): {fmt(train_auprc[class_index])} / {fmt(test_auprc[class_index])}"
                )

        return html.Pre("\n".join(lines)), distribution_fig

    @app.callback(
        Output("feature_notes", "value"),
        Input("selected_cell", "data"),
        Input("feature_notes_store", "data"),
    )
    def populate_feature_notes(selected_cell, store_data):
        if not selected_cell or not isinstance(selected_cell, dict):
            return ""
        feat_id = selected_cell.get("feat")
        if feat_id is None:
            return ""
        notes_store = store_data or {}
        notes = notes_store.get("notes") if isinstance(notes_store, dict) else {}
        if not isinstance(notes, dict):
            return ""
        return notes.get(str(feat_id), "")

    @app.callback(
        Output("feature_notes_store", "data"),
        Output("feature_notes_status", "children"),
        Input("model_select", "value"),
        Input("feature_notes_save", "n_clicks"),
        Input("feature_notes_clear", "n_clicks"),
        State("selected_cell", "data"),
        State("feature_notes", "value"),
        State("feature_notes_store", "data"),
    )
    def sync_feature_notes(model_name, save_clicks, clear_clicks, selected_cell, note_value, store_data):
        ctx = callback_context
        if not ctx.triggered:
            # Initial load: prime the store from disk for the selected model.
            if model_name and model_name in context.model_index:
                initial_notes = data.load_feature_notes(model_name)
                return {"model": model_name, "notes": initial_notes}, ""
            return {"model": model_name or "", "notes": {}}, ""

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        store = store_data if isinstance(store_data, dict) else {}
        notes_map = dict(store.get("notes") or {})

        if trigger == "model_select":
            if not model_name or model_name not in context.model_index:
                return {"model": model_name or "", "notes": {}}, ""
            refreshed_notes = data.load_feature_notes(model_name)
            return {"model": model_name, "notes": refreshed_notes}, ""

        if not model_name or model_name not in context.model_index:
            return dash.no_update, "Select a model before editing notes."

        if not selected_cell or not isinstance(selected_cell, dict) or selected_cell.get("feat") is None:
            return dash.no_update, "Select a feature before editing notes."

        feat_id = str(selected_cell.get("feat"))

        if trigger == "feature_notes_save":
            note_text = note_value or ""
            if not note_text.strip():
                if feat_id in notes_map:
                    notes_map.pop(feat_id, None)
                    data.save_feature_notes(model_name, notes_map)
                    return {"model": model_name, "notes": notes_map}, f"Removed empty note for feature #{feat_id}."
                return dash.no_update, f"No note content to save for feature #{feat_id}."

            current = notes_map.get(feat_id)
            if current == note_text:
                return dash.no_update, f"No changes to note for feature #{feat_id}."

            notes_map[feat_id] = note_text
            data.save_feature_notes(model_name, notes_map)
            return {"model": model_name, "notes": notes_map}, f"Saved note for feature #{feat_id}."

        if trigger == "feature_notes_clear":
            if feat_id in notes_map:
                notes_map.pop(feat_id, None)
                data.save_feature_notes(model_name, notes_map)
                return {"model": model_name, "notes": notes_map}, f"Cleared note for feature #{feat_id}."
            return dash.no_update, f"No saved note to clear for feature #{feat_id}."

        return dash.no_update, ""

    @app.callback(
        Output("feature_notes_save", "disabled"),
        Output("feature_notes_clear", "disabled"),
        Input("selected_cell", "data"),
        Input("feature_notes_store", "data"),
    )
    def toggle_note_buttons(selected_cell, store_data):
        valid_selection = bool(selected_cell and isinstance(selected_cell, dict) and selected_cell.get("feat") is not None)
        if not valid_selection:
            return True, True
        feat_id = str(selected_cell.get("feat"))
        notes_store = store_data or {}
        notes = notes_store.get("notes") if isinstance(notes_store, dict) else {}
        note_value = notes.get(feat_id) if isinstance(notes, dict) else None
        has_note = isinstance(note_value, str) and bool(note_value.strip())
        return False, not has_note

    @app.callback(
        Output("heatmap", "figure"),
        Input("model_select", "value"),
        Input("split_select", "value"),
        Input("num_feats", "value"),
        Input("colormap_select", "value"),
        Input("feature_sort", "value"),
        Input("baseline_page", "value"),
        Input("sae_page", "value"),
        Input("selected_cell", "data"),
    )
    def update_heatmap(model_name, split, n, colorscale, sort_method, baseline_page, sae_page, selected_data):
        if not model_name:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Select a model to view heatmap",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return empty_fig
        if model_name not in context.model_index:
            raise dash.exceptions.PreventUpdate

        cache = data.load_cache(model_name, split)
        H = cache["H"]
        original_feat_labels = cache["feat_labels"]
        class_labels = cache["class_labels"]
        min_f, def_f, max_f = data.slider_bounds(H)
        if data.is_baseline_model(model_name):
            n = page_size
        else:
            n = int(max(min_f, min(n or def_f, max_f)))

        sorted_feat_labels = load_and_sort_features(model_name, split, sort_method)
        if sort_method != "default":
            original_to_index = {feat_id: idx for idx, feat_id in enumerate(original_feat_labels)}
            sorted_indices = [original_to_index[feat_id] for feat_id in sorted_feat_labels if feat_id in original_to_index]
            H_sorted = H[:, sorted_indices]
            feat_labels = sorted_feat_labels[: len(sorted_indices)]
        else:
            H_sorted = H
            feat_labels = original_feat_labels

        # Handle baseline model pagination
        if data.is_baseline_model(model_name):
            page = baseline_page or 1
            start = (page - 1) * page_size
            end = start + page_size
            H_page = H_sorted[:, start:end]
            feat_page = feat_labels[start:end]

            selected = None
            if selected_data:
                sel_class = selected_data.get("class")
                sel_feat = selected_data.get("feat")
                if sel_class in class_labels and sel_feat in feat_page:
                    selected = (sel_class, sel_feat)

            return make_heatmap_fig(H_page, feat_page, class_labels, colorscale, selected=selected)

        # Handle SAE model pagination
        total_feats = len(feat_labels)
        if total_feats > n and sae_page:
            # Pagination is active for SAE models
            page = sae_page or 1
            start = (page - 1) * n
            end = start + n
            H_page = H_sorted[:, start:end]
            feat_page = feat_labels[start:end]

            selected = None
            if selected_data:
                sel_class = selected_data.get("class")
                sel_feat = selected_data.get("feat")
                if sel_class in class_labels and sel_feat in feat_page:
                    selected = (sel_class, sel_feat)

            return make_heatmap_fig(H_page, feat_page, class_labels, colorscale, selected=selected)
        else:
            # No pagination needed - show first n features
            selected = None
            if selected_data:
                sel_class = selected_data.get("class")
                sel_feat = selected_data.get("feat")
                if sel_class in class_labels and sel_feat in feat_labels[:n]:
                    selected = (sel_class, sel_feat)

            return make_heatmap_fig(H_sorted[:, :n], feat_labels[:n], class_labels, colorscale, selected=selected)

    @app.callback(
        Output("selected_cell", "data"),
        Input("heatmap", "clickData"),
        Input("model_select", "value"),
        Input("split_select", "value"),
        Input("feature_sort", "value"),
        Input("baseline_page", "value"),
        Input("sae_page", "value"),
        State("num_feats", "value"),
        prevent_initial_call=True,
    )
    def store_selection(clickData, model_name, split, sort_method, baseline_page, sae_page, n_feats):
        if not model_name or model_name not in context.model_index:
            raise dash.exceptions.PreventUpdate
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        trigger = ctx.triggered[0]["prop_id"]
        if trigger.startswith("model_select.") or trigger.startswith("split_select."):
            return None
        if clickData and "points" in clickData and clickData["points"]:
            point = clickData["points"][0]
            x_idx = point.get("x")
            y_idx = point.get("y")
            if isinstance(x_idx, (int, float)) and isinstance(y_idx, (int, float)):
                cache = data.load_cache(model_name, split)
                feat_labels_sorted = load_and_sort_features(model_name, split, sort_method)
                feat_labels = feat_labels_sorted
                class_labels = cache["class_labels"]

                # Calculate the actual feature index accounting for pagination
                actual_x_idx = int(x_idx)

                if data.is_baseline_model(model_name):
                    # Baseline model pagination
                    page = baseline_page or 1
                    offset = (page - 1) * page_size
                    actual_x_idx = offset + int(x_idx)
                else:
                    # SAE model pagination
                    n = n_feats or 50
                    total_feats = len(feat_labels)
                    if total_feats > n and sae_page:
                        page = sae_page or 1
                        offset = (page - 1) * n
                        actual_x_idx = offset + int(x_idx)

                if 0 <= actual_x_idx < len(feat_labels) and 0 <= int(y_idx) < len(class_labels):
                    return {"class": class_labels[int(y_idx)], "feat": feat_labels[actual_x_idx]}
        return dash.no_update

    @app.callback(
        Output("sidebar_tabs", "value"),
        Input("heatmap", "clickData"),
        State("sidebar_tabs", "value"),
        prevent_initial_call=True,
    )
    def focus_neuron_tab(clickData, current_tab):
        if not clickData or not clickData.get("points"):
            raise dash.exceptions.PreventUpdate
        if current_tab == "feature":
            raise dash.exceptions.PreventUpdate
        return "feature"

    @app.callback(
        Output("image_modal", "style"),
        Output("modal_image", "src"),
        Output("modal_image", "style"),
        Output("modal_caption", "children"),
        [
            Input({"type": "montage_image", "index": ALL}, "n_clicks"),
            Input({"type": "feature_image", "index": ALL}, "n_clicks"),
        ],
        Input("close_modal", "n_clicks"),
        State("model_select", "value"),
        State("split_select", "value"),
        State("selected_cell", "data"),
        prevent_initial_call=True,
    )
    def handle_modal(montage_clicks, feature_clicks, close_clicks, model_name, split, selected_cell):
        ctx = callback_context
        if not ctx.triggered:
            return {"display": "none"}, "", {}, ""

        trigger_id = ctx.triggered[0]["prop_id"]
        if "close_modal" in trigger_id:
            return {"display": "none"}, "", {}, ""

        class_label = None
        if selected_cell and isinstance(selected_cell, dict):
            class_label = selected_cell.get("class")
        class_label_clean = None
        if isinstance(class_label, str):
            if " (n=" in class_label:
                class_label_clean = class_label.split(" (n=")[0]
            else:
                class_label_clean = class_label

        if "montage_image" in trigger_id and any(montage_clicks or []):
            import json as _json

            trigger_dict = _json.loads(trigger_id.rsplit(".", 1)[0])
            image_index = trigger_dict.get("index", 0)
            try:
                cache = data.load_cache(model_name, split)
                image_paths = cache["image_paths"]
                if 0 <= image_index < len(image_paths):
                    path = image_paths[image_index]
                    img_src = encode_original_size_image(path)
                    orig_width, orig_height = get_image_dimensions(path)
                    modal_width = orig_width * 2
                    modal_height = orig_height * 2
                    max_width = min(modal_width, int(0.9 * 1920))
                    max_height = min(modal_height, int(0.9 * 1080))
                    img_style = {
                        "width": f"{max_width}px",
                        "height": f"{max_height}px",
                        "objectFit": "contain",
                        "display": "block",
                    }
                    modal_style = {
                        "position": "fixed",
                        "zIndex": "1000",
                        "left": "0",
                        "top": "0",
                        "width": "100%",
                        "height": "100%",
                        "overflow": "auto",
                        "backgroundColor": "transparent",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                    }
                    set_description_context(model_name, class_label_clean)
                    description = describe_patch(path)
                    formatted_description = format_description(description)
                    return modal_style, img_src, img_style, formatted_description
            except Exception:
                pass
            return {"display": "none"}, "", {}, ""

        if "feature_image" in trigger_id and any(feature_clicks or []):
            import json as _json

            try:
                trigger_dict = _json.loads(trigger_id.rsplit(".", 1)[0])
                image_path = trigger_dict.get("index", "")
            except (_json.JSONDecodeError, IndexError):
                image_path = ""
            if not image_path or not os.path.exists(image_path):
                return {"display": "none"}, "", {}, ""
            try:
                img_src = encode_original_size_image(image_path)
                orig_width, orig_height = get_image_dimensions(image_path)
                display_width = int(orig_width * 0.75)
                display_height = int(orig_height * 0.75)
                img_style = {
                    "width": f"{display_width}px",
                    "height": f"{display_height}px",
                    "objectFit": "contain",
                    "display": "block",
                }
                modal_style = {
                    "position": "fixed",
                    "zIndex": "1000",
                    "left": "0",
                    "top": "0",
                    "width": "100%",
                    "height": "100%",
                    "overflow": "auto",
                    "backgroundColor": "transparent",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                }
                set_description_context(model_name, class_label_clean)
                return modal_style, img_src, img_style, ""
            except Exception as exc:
                print(f"Error loading feature image for modal: {exc}")
        return {"display": "none"}, "", {}, ""

    @app.callback(
        Output("montage", "children"),
        Input("heatmap", "clickData"),
        Input("montage_k", "value"),
        Input("split_select", "value"),
        Input("feature_sort", "value"),
        Input("baseline_page", "value"),
        Input("sae_page", "value"),
        State("model_select", "value"),
        State("num_feats", "value"),
    )
    def on_click(data_point, k, split, sort_method, baseline_page, sae_page, model_name, n_feats):
        if not data_point or not model_name or model_name not in context.model_index:
            return []
        if not split:
            return []
        cache = data.load_cache(model_name, split)
        H = cache["H"]
        original_feat_labels = cache["feat_labels"]
        class_labels = cache["class_labels"]
        TOPK_IDX = cache["TOPK_IDX"]
        TOPK_VAL = cache["TOPK_VAL"]
        image_paths = cache["image_paths"]
        min_f, def_f, max_f = data.slider_bounds(H)
        if data.is_baseline_model(model_name):
            n_feats = page_size
        else:
            n_feats = int(max(min_f, min(n_feats or def_f, max_f)))

        sorted_feat_labels = load_and_sort_features(model_name, split, sort_method)
        if sort_method != "default":
            original_to_index = {feat_id: idx for idx, feat_id in enumerate(original_feat_labels)}
            sorted_indices = [original_to_index[feat_id] for feat_id in sorted_feat_labels if feat_id in original_to_index]
            feat_labels = sorted_feat_labels[: len(sorted_indices)]
            display_to_original = {i: sorted_indices[i] for i in range(len(sorted_indices))}
        else:
            feat_labels = original_feat_labels
            display_to_original = {i: i for i in range(len(feat_labels))}

        if data.is_baseline_model(model_name):
            page = baseline_page or 1
            start = (page - 1) * page_size
            end = start + page_size
            visible_feat_labels = feat_labels[start:end]
            if sort_method != "default":
                display_to_original_page = {}
                for local_i, global_pos in enumerate(range(start, min(end, len(feat_labels)))):
                    if global_pos in display_to_original:
                        display_to_original_page[local_i] = display_to_original[global_pos]
                display_to_original = display_to_original_page
            else:
                display_to_original = {i: (start + i) for i in range(len(visible_feat_labels))}
        else:
            # SAE model - check if pagination is needed
            total_feats = len(feat_labels)
            if total_feats > n_feats and sae_page:
                # SAE pagination is active
                page = sae_page or 1
                start = (page - 1) * n_feats
                end = start + n_feats
                visible_feat_labels = feat_labels[start:end]

                if sort_method != "default":
                    display_to_original_page = {}
                    for local_i, global_pos in enumerate(range(start, min(end, len(feat_labels)))):
                        if global_pos in display_to_original:
                            display_to_original_page[local_i] = display_to_original[global_pos]
                    display_to_original = display_to_original_page
                else:
                    display_to_original = {i: (start + i) for i in range(len(visible_feat_labels))}
            else:
                # No pagination - show first n_feats
                visible_feat_labels = feat_labels[:n_feats]
                # display_to_original already set correctly above

        feat_to_col_visible = {lbl: idx for idx, lbl in enumerate(visible_feat_labels)}

        point = data_point["points"][0]
        raw_x = point.get("x")
        raw_y = point.get("y")
        try:
            raw_x = int(raw_x)
            raw_y = int(raw_y)
        except (TypeError, ValueError):
            return [html.Div("Invalid selection indices.", style={"color": "red"})]

        if raw_y < 0 or raw_y >= len(class_labels) or raw_x < 0 or raw_x >= len(visible_feat_labels):
            return [html.Div("Selection out of bounds.", style={"color": "red"})]

        row_label = class_labels[raw_y]
        col_label = visible_feat_labels[raw_x]

        if col_label not in feat_to_col_visible:
            return [html.Div(f"Feature not visible: {col_label}", style={"color": "red"})]

        display_j = feat_to_col_visible[col_label]
        j = display_to_original.get(display_j, display_j)
        clean_row_label = row_label.split(" (n=")[0] if " (n=" in row_label else row_label

        idxs = TOPK_IDX[raw_y, j, :]
        vals = TOPK_VAL[raw_y, j, :]

        classification_suffix = ""
        try:
            locked_comp = data.load_locked_feature_comparison(model_name)
            if locked_comp and "feature_comparisons" in locked_comp:
                cls_map = {
                    str(fc["feature_index"]): fc.get("classification_label_auprc", fc.get("classification_label", ""))
                    for fc in locked_comp["feature_comparisons"]
                }
                cls_label = cls_map.get(col_label, "")
                if cls_label:
                    classification_suffix = f" ({cls_label})"
        except Exception:
            pass

        header = html.H3(
            f"Showing top activating images from class '{clean_row_label}' for feature #{col_label}{classification_suffix}",
            style={"fontSize": "20px", "fontWeight": "bold", "marginBottom": "0.5rem", "color": "#333"},
        )
        set_description_context(model_name, clean_row_label)

        separator = html.Div(
            style={
                "height": "1px",
                "backgroundColor": "#ddd",
                "width": "100%",
                "margin": "0.25rem 0 0.75rem 0",
            }
        )

        out: List = []
        shown = 0
        for s_idx, v in zip(idxs, vals):
            s_idx = int(s_idx)
            if s_idx < 0:
                continue
            path = image_paths[s_idx]
            try:
                src = encode_thumb(path, max_side=256)
                cached = is_description_cached(model_name, path, clean_row_label)
                cache_indicator = None
                if cached:
                    cache_indicator = html.Div(
                        "✓",
                        style={
                            "position": "absolute",
                            "top": "4px",
                            "right": "4px",
                            "backgroundColor": "rgba(34, 139, 34, 0.9)",
                            "color": "white",
                            "borderRadius": "50%",
                            "width": "20px",
                            "height": "20px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "12px",
                            "fontWeight": "bold",
                            "zIndex": "10",
                        },
                    )

                image_container_children = [
                    html.Img(
                        src=src,
                        style={"height": "160px", "width": "100%", "objectFit": "cover"},
                    )
                ]
                if cache_indicator:
                    image_container_children.append(cache_indicator)

                image_container = html.Div(
                    image_container_children,
                    style={"position": "relative", "width": "100%"},
                )

                out.append(
                    html.Div(
                        [
                            html.Button(
                                image_container,
                                id={"type": "montage_image", "index": s_idx},
                                style={
                                    "border": "none",
                                    "background": "none",
                                    "padding": "0",
                                    "cursor": "pointer",
                                    "width": "100%",
                                },
                            ),
                            html.Div(f"act={float(v):.2f}", style={"textAlign": "center", "fontSize": "12px"}),
                        ],
                        style={"display": "flex", "flexDirection": "column", "alignItems": "center"},
                    )
                )
                shown += 1
                if shown >= int(k):
                    break
            except Exception:
                continue

        if not out:
            return [header, separator, html.Div("No activations found for this cell.", style={"color": "orange"})]

        return [header, separator, html.Div(out, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"})]
