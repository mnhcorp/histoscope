"""Analysis report tab layout and callbacks."""
from __future__ import annotations

import dash
from dash import Input, Output, dcc, html

from .. import data
from ..context import AppContext


def layout(context: AppContext) -> dcc.Tab:
    return dcc.Tab(
        label="Analysis Report",
        value="report_tab",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Model"),
                            dcc.Dropdown(
                                id="model_select_tab2",
                                options=context.initial_model_options,
                                value=context.default_model,
                                clearable=False,
                                style={"minWidth": "400px"},
                            ),
                        ],
                        style={"padding": "1rem"},
                    ),
                    html.Div(
                        id="report_content",
                        style={"padding": "1rem", "fontSize": "0.9rem"},
                    ),
                ]
            )
        ],
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(Output("report_content", "children"), Input("model_select_tab2", "value"))
    def update_report_content(model_name):
        full_report = data.load_full_report(model_name)
        return html.Pre(
            full_report,
            style={"whiteSpace": "pre-wrap", "fontSize": "0.85rem", "fontFamily": "monospace"},
        )
