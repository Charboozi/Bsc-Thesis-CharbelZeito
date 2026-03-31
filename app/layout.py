from dash import html
from app.components.sidebar import create_sidebar
from app.components.explanation_panel import create_explanation_panel
from app.components.results_panel import create_results_panel
from app.components.graphs import create_main_graph_section


def create_layout():
    return html.Div(
        className="app-container",
        children=[
            html.Div(
                className="sidebar",
                children=create_sidebar(),
            ),
            html.Div(
                className="main-content",
                children=[
                    html.H1("Linear Regression", className="page-title"),
                    html.Div(
                        className="top-row",
                        children=[
                            html.Div(
                                className="card graph-card",
                                children=create_main_graph_section(),
                            ),
                            html.Div(
                                className="card explanation-card",
                                children=create_explanation_panel(),
                            ),
                        ],
                    ),
                    html.Div(
                        className="bottom-row",
                        children=[
                            html.Div(
                                className="card results-card",
                                children=create_results_panel(),
                            )
                        ],
                    ),
                ],
            ),
        ],
    )