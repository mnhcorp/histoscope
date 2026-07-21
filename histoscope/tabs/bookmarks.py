"""Bookmarks tab layout and callbacks."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import dash
from dash import ALL, Input, Output, State, callback_context, dcc, html

from .. import data
from ..context import AppContext

BOOKMARKS_VERSION = 1
MAX_BOOKMARKS = 200
BASELINE_PAGE_SIZE = 50


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean_class_label(label: Optional[str]) -> str:
    if not label or not isinstance(label, str):
        return ""
    if " (n=" in label:
        return label.split(" (n=")[0]
    return label


def bookmark_key(model: str, split: str, feat: str) -> str:
    return f"{model}::{split}::{feat}"


def _safe_int(value: Optional[object]) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def ensure_payload(raw: Optional[Dict]) -> Dict[str, List[Dict[str, str]]]:
    if not isinstance(raw, dict) or raw.get("version") != BOOKMARKS_VERSION:
        return {"version": BOOKMARKS_VERSION, "items": []}

    items: Iterable = raw.get("items", [])
    cleaned: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        dataset = item.get("dataset")
        split = item.get("split")
        feat = item.get("feat")
        if not all([model, dataset, split, feat]):
            continue
        key = str(item.get("key") or bookmark_key(model, split, feat))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "key": key,
                "model": str(model),
                "dataset": str(dataset),
                "split": str(split),
                "feat": str(feat),
                "class_label": str(item.get("class_label", "")),
                "class_label_clean": str(
                    item.get("class_label_clean", _clean_class_label(item.get("class_label")))
                ),
                "created": str(item.get("created", _utc_now_iso())),
                "sort_method": str(item.get("sort_method", "default")),
                "num_feats": _safe_int(item.get("num_feats")),
                "baseline_page": _safe_int(item.get("baseline_page")),
            }
        )
    return {"version": BOOKMARKS_VERSION, "items": cleaned[:MAX_BOOKMARKS]}


def _format_timestamp(ts: Optional[str]) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return ts


def _render_bookmark_card(item: Dict[str, str]) -> html.Div:
    key = item["key"]
    model_label = data.model_display_label(item["model"])
    class_label = item.get("class_label_clean") or "Any class"
    timestamp = _format_timestamp(item.get("created"))

    sort_method = item.get("sort_method", "default")
    open_button = html.Button(
        [
            html.Span(f"Feature #{item['feat']}", style={"display": "block", "fontWeight": "600"}),
            html.Span(model_label, style={"display": "block", "color": "#1a365d"}),
            html.Span(
                f"Dataset: {item['dataset']} · Split: {item['split']}",
                style={"display": "block", "fontSize": "0.85rem", "color": "#475569"},
            ),
            html.Span(
                f"Sort: {sort_method}",
                style={"display": "block", "fontSize": "0.8rem", "color": "#475569"},
            ),
            html.Span(
                f"Class focus: {class_label}",
                style={"display": "block", "fontSize": "0.85rem", "color": "#475569"},
            ),
            html.Span(
                f"Added: {timestamp}",
                style={"display": "block", "fontSize": "0.75rem", "color": "#64748b", "marginTop": "0.25rem"},
            ),
        ],
        id={"type": "bookmark_open", "index": key},
        n_clicks=0,
        style={
            "flex": "1 1 auto",
            "textAlign": "left",
            "padding": "0.75rem 0.9rem",
            "border": "1px solid #cbd5e1",
            "borderRadius": "6px",
            "backgroundColor": "#f8fafc",
            "cursor": "pointer",
        },
    )

    remove_button = html.Button(
        "Remove",
        id={"type": "bookmark_remove", "index": key},
        n_clicks=0,
        style={
            "marginLeft": "0.75rem",
            "padding": "0.45rem 0.85rem",
            "backgroundColor": "#dc2626",
            "color": "white",
            "border": "none",
            "borderRadius": "4px",
            "cursor": "pointer",
        },
    )

    return html.Div(
        [open_button, remove_button],
        style={
            "display": "flex",
            "alignItems": "stretch",
            "marginBottom": "0.75rem",
        },
    )


def layout(context: AppContext) -> dcc.Tab:
    del context  # unused in layout construction for now
    return dcc.Tab(
        label="Bookmarks",
        value="bookmarks_tab",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Bookmarks",
                                style={"margin": "0", "fontSize": "1.4rem", "fontWeight": "600"},
                            ),
                            html.Div(
                                "Bookmark features from the heatmap and revisit them here.",
                                style={"fontSize": "0.95rem", "color": "#475569", "marginTop": "0.35rem"},
                            ),
                            html.Button(
                                "Clear all",
                                id="bookmark_clear_button",
                                n_clicks=0,
                                disabled=True,
                                style={
                                    "marginTop": "0.75rem",
                                    "padding": "0.4rem 0.9rem",
                                    "backgroundColor": "#e11d48",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={"marginBottom": "1.25rem"},
                    ),
                    html.Div(id="bookmarks_feedback", style={"minHeight": "1.2rem", "color": "#1a365d"}),
                    html.Div(id="bookmarks_summary", style={"marginBottom": "0.75rem", "color": "#1f2937"}),
                    html.Div(id="bookmarks_list"),
                ],
                style={"padding": "1.25rem", "maxWidth": "840px"},
            )
        ],
    )


def _create_bookmark(
    model: str,
    dataset: str,
    split: str,
    feat: str,
    class_label: Optional[str],
    sort_method: Optional[str],
    num_feats: Optional[int],
    baseline_page: Optional[int],
) -> Dict[str, str]:
    feat_str = str(feat)
    key = bookmark_key(model, split, feat_str)
    return {
        "key": key,
        "model": model,
        "dataset": dataset,
        "split": split,
        "feat": feat_str,
        "class_label": class_label or "",
        "class_label_clean": _clean_class_label(class_label),
        "created": _utc_now_iso(),
        "sort_method": sort_method or "default",
        "num_feats": num_feats,
        "baseline_page": baseline_page,
    }


def register_callbacks(app, context: AppContext) -> None:  # pragma: no cover - Dash wiring
    dataset_lookup: Dict[str, str] = {}
    for dataset_name, models in context.dataset_models.items():
        for model_key in models:
            dataset_lookup[model_key] = dataset_name

    @app.callback(  # type: ignore[misc]
        Output("bookmarks_summary", "children"),
        Output("bookmarks_list", "children"),
        Input("bookmarks_store", "data"),
    )
    def render_bookmarks(store_data):
        payload = ensure_payload(store_data)
        items = payload["items"]
        if not items:
            summary = ""
            empty = html.Div(
                "No bookmarks yet. Use the heatmap's Add Bookmark button to save features.",
                style={"color": "#64748b", "fontStyle": "italic"},
            )
            return summary, [empty]

        count = len(items)
        summary = f"You have {count} bookmark{'s' if count != 1 else ''}."
        cards = [_render_bookmark_card(item) for item in items]
        return summary, cards

    @app.callback(  # type: ignore[misc]
        Output("bookmark_clear_button", "disabled"),
        Input("bookmarks_store", "data"),
    )
    def disable_clear_button(store_data):
        payload = ensure_payload(store_data)
        return not payload["items"]

    @app.callback(  # type: ignore[misc]
        Output("bookmarks_store", "data"),
        Output("bookmark_add_feedback", "children"),
        Output("bookmarks_feedback", "children"),
        Input("bookmark_add_button", "n_clicks"),
        Input({"type": "bookmark_remove", "index": ALL}, "n_clicks"),
        Input("bookmark_clear_button", "n_clicks"),
        State("selected_cell", "data"),
        State("model_select", "value"),
        State("split_select", "value"),
        State("dataset_select", "value"),
        State("num_feats", "value"),
        State("baseline_page", "value"),
        State("feature_sort", "value"),
        State("bookmarks_store", "data"),
        prevent_initial_call=True,
    )
    def manage_bookmarks(
        add_clicks,
        remove_clicks,
        clear_clicks,
        selected_cell,
        model_value,
        split_value,
        dataset_value,
        num_feats_value,
        baseline_page_value,
        sort_value,
        store_data,
    ):
        trigger = callback_context.triggered_id
        payload = ensure_payload(store_data)
        items = payload["items"]
        heatmap_msg = dash.no_update
        tab_msg = dash.no_update

        if trigger == "bookmark_add_button":
            if not selected_cell or not model_value or not split_value:
                heatmap_msg = "Select a feature before bookmarking."
                return dash.no_update, heatmap_msg, tab_msg

            feat = selected_cell.get("feat")
            if feat is None:
                heatmap_msg = "Unable to bookmark: missing feature id."
                return dash.no_update, heatmap_msg, tab_msg

            dataset_for_model = dataset_lookup.get(model_value)
            if not dataset_for_model:
                try:
                    dataset_for_model, _ = data.split_model_key(model_value)
                except Exception:
                    dataset_for_model = None
            if not dataset_for_model:
                dataset_for_model = dataset_value
            if not dataset_for_model:
                heatmap_msg = "Unable to determine dataset for this model."
                return dash.no_update, heatmap_msg, tab_msg

            if data.is_baseline_model(model_value):
                baseline_page_for_bookmark = _safe_int(baseline_page_value)
            else:
                baseline_page_for_bookmark = None

            bookmark = _create_bookmark(
                model=model_value,
                dataset=dataset_for_model,
                split=split_value,
                feat=str(feat),
                class_label=selected_cell.get("class"),
                sort_method=sort_value,
                num_feats=_safe_int(num_feats_value),
                baseline_page=baseline_page_for_bookmark,
            )
            key = bookmark["key"]
            filtered = [item for item in items if item["key"] != key]
            filtered.insert(0, bookmark)
            payload = {"version": BOOKMARKS_VERSION, "items": filtered[:MAX_BOOKMARKS]}
            heatmap_msg = f"Bookmarked feature #{bookmark['feat']} ({bookmark['split']})."
            tab_msg = f"Saved bookmark for feature #{bookmark['feat']}."
            return payload, heatmap_msg, tab_msg

        if trigger == "bookmark_clear_button":
            if not items:
                raise dash.exceptions.PreventUpdate
            payload = {"version": BOOKMARKS_VERSION, "items": []}
            tab_msg = "Cleared all bookmarks."
            return payload, "", tab_msg

        if isinstance(trigger, dict) and trigger.get("type") == "bookmark_remove":
            key = trigger.get("index")
            filtered = [item for item in items if item["key"] != key]
            if len(filtered) == len(items):
                raise dash.exceptions.PreventUpdate
            payload = {"version": BOOKMARKS_VERSION, "items": filtered}
            tab_msg = f"Removed bookmark for feature {key.split('::')[-1]}"
            return payload, "", tab_msg

        raise dash.exceptions.PreventUpdate

    @app.callback(  # type: ignore[misc]
        Output("main_tabs", "value"),
        Output("dataset_select", "value"),
        Output("selected_model_store", "data"),
        Output("split_select", "value"),
        Output("feature_sort", "value"),
        Output("num_feats", "value"),
        Output("baseline_page", "value"),
        Output("selected_cell", "data"),
        Input({"type": "bookmark_open", "index": ALL}, "n_clicks"),
        State("bookmarks_store", "data"),
        State("dataset_select", "value"),
        State("selected_model_store", "data"),
        State("split_select", "value"),
        State("feature_sort", "value"),
        State("num_feats", "value"),
        State("baseline_page", "value"),
        prevent_initial_call=True,
    )
    def open_bookmark(
        _open_clicks,
        store_data,
        current_dataset,
        current_model_store,
        current_split,
        current_sort,
        current_num_feats,
        current_baseline_page,
    ):
        trigger = callback_context.triggered_id
        if not isinstance(trigger, dict) or trigger.get("type") != "bookmark_open":
            raise dash.exceptions.PreventUpdate

        key = trigger.get("index")
        payload = ensure_payload(store_data)
        lookup = {item["key"]: item for item in payload["items"]}
        if key not in lookup:
            raise dash.exceptions.PreventUpdate

        bookmark = lookup[key]
        dataset_value = bookmark["dataset"]
        model_value = bookmark["model"]
        split_value = bookmark["split"]
        selected_cell = {
            "feat": bookmark["feat"],
            "class": bookmark.get("class_label") or bookmark.get("class_label_clean") or "",
        }
        dataset_output = dataset_value if dataset_value else dash.no_update

        model_store_output = model_value

        split_output = split_value if split_value else dash.no_update

        sort_method = bookmark.get("sort_method", "default")
        sort_output = sort_method if sort_method else dash.no_update

        cache = None
        try:
            cache = data.load_cache(model_value, split_value)
        except Exception:
            cache = None

        if data.is_baseline_model(model_value):
            baseline_target = bookmark.get("baseline_page")
            if baseline_target is None and cache is not None:
                try:
                    from . import heatmap as _heatmap

                    sorted_feats = _heatmap.load_and_sort_features(model_value, split_value, sort_method)
                    if str(selected_cell["feat"]) in sorted_feats:
                        idx = sorted_feats.index(str(selected_cell["feat"]))
                        baseline_target = 1 + idx // BASELINE_PAGE_SIZE
                except Exception:
                    baseline_target = None
                if baseline_target is None:
                    baseline_target = 1
            baseline_output = (
                baseline_target
                if baseline_target is not None and baseline_target != current_baseline_page
                else dash.no_update
            )
            num_feats_output = dash.no_update
        else:
            desired_num_feats = bookmark.get("num_feats")
            try:
                desired_num_feats = int(desired_num_feats) if desired_num_feats is not None else None
            except (TypeError, ValueError):
                desired_num_feats = None

            feat_id = selected_cell["feat"]
            if desired_num_feats is None and cache is not None:
                try:
                    from . import heatmap as _heatmap

                    sorted_feats = _heatmap.load_and_sort_features(model_value, split_value, sort_method)
                    if str(feat_id) in sorted_feats:
                        desired_num_feats = sorted_feats.index(str(feat_id)) + 1
                except Exception:
                    desired_num_feats = None

            if desired_num_feats is not None and cache is not None:
                min_f, _def_f, max_f = data.slider_bounds(cache["H"])
                allowed = [opt for opt in [10, 20, 30, 40, 50] if opt <= max_f]
                if allowed:
                    desired_num_feats = max(min_f, desired_num_feats)
                    mapped = None
                    for opt in allowed:
                        if opt >= desired_num_feats:
                            mapped = opt
                            break
                    if mapped is None:
                        mapped = allowed[-1]
                    desired_num_feats = mapped

            if desired_num_feats is None:
                num_feats_output = dash.no_update
            else:
                num_feats_output = (
                    desired_num_feats if desired_num_feats != current_num_feats else dash.no_update
                )
            baseline_output = dash.no_update

        return (
            "heatmap_tab",
            dataset_output,
            model_store_output,
            split_output,
            sort_output,
            num_feats_output,
            baseline_output,
            selected_cell,
        )
