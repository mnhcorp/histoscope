"""Top-level Dash layout for HistoSCOPE."""
from __future__ import annotations

from dash import dcc, html

from .patch_browser import patch_browser_tab

from .context import AppContext
from .tabs import cluster_analysis, feature_analysis, flipper, heatmap, report


def build_layout(context: AppContext):
    modal = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "×",
                        id="close_modal",
                        style={
                            "position": "absolute",
                            "top": "10px",
                            "right": "15px",
                            "fontSize": "28px",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                            "color": "#aaa",
                            "zIndex": "1001",
                        },
                    ),
                    html.Img(id="modal_image"),
                    html.Div(
                        id="modal_caption",
                        style={
                            "marginTop": "0.75rem",
                            "maxWidth": "780px",
                            "fontFamily": "monospace",
                            "fontSize": "0.85rem",
                            "whiteSpace": "pre-wrap",
                            "lineHeight": "1.25",
                            "color": "#222",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "backgroundColor": "#fff",
                    "margin": "auto",
                    "padding": "14px",
                    "border": "2px solid #888",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "boxShadow": "0 4px 18px rgba(0,0,0,0.35)",
                    "borderRadius": "6px",
                },
            )
        ],
        id="image_modal",
        style={
            "position": "fixed",
            "zIndex": "1000",
            "left": "0",
            "top": "0",
            "width": "100%",
            "height": "100%",
            "overflow": "auto",
            "backgroundColor": "rgba(0,0,0,0.55)",
            "display": "none",
            "alignItems": "center",
            "justifyContent": "center",
            "backdropFilter": "blur(2px)",
        },
    )

    return html.Div(
        [
            html.Link(
                href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap",
                rel="stylesheet",
            ),
            dcc.Store(id="selected_cell"),
            dcc.Store(id="selected_model_store", data=context.default_model),
            html.H1(
                "HistoSCOPE v0.5",
                style={
                    "textAlign": "left",
                    "fontSize": "2.0rem",
                    "margin": "1.0rem 0 1.0rem 0",
                    "padding": "0 0 0 1.25rem",
                    "fontFamily": "'Orbitron', monospace",
                },
            ),
            html.H2(
                "Explore feature activations for histology foundation models.",
                style={
                    "textAlign": "left",
                    "fontSize": "1.2rem",
                    "margin": "0 0 1.5rem 0",
                    "padding": "0 0 0 1.25rem",
                    "fontFamily": "'Orbitron', monospace",
                },
            ),
            html.Div(),
            html.Div(
                [
                    html.Label("Dataset", style={"marginRight": "0.5rem"}),
                    dcc.Dropdown(
                        id="dataset_select",
                        options=context.dataset_options,
                        value=context.default_dataset,
                        clearable=False,
                        style={"width": "260px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "padding": "0 1.25rem 0.5rem"},
            ),
            modal,
            dcc.Store(id="modal_image_path"),
            dcc.Tabs(
                id="main_tabs",
                value="heatmap_tab",
                style={"fontWeight": "bold"},
                children=[
                    heatmap.layout(context),
                    cluster_analysis.layout(context),
                    patch_browser_tab(),
                    feature_analysis.layout(context),
                    flipper.layout(context),
                    report.layout(context),
                ],
            ),
        ]
    )
