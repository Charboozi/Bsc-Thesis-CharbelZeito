from dash import html, dcc
from app.components.sidebar import create_sidebar
from app.pages.linear_regression_page import create_linear_regression_page


def create_layout():
    return html.Div(
        className="app-container",
        children=[
            dcc.Store(id="selected-model-store", data="linear_regression"),
            html.Div(
                id="sidebar-container",
                className="sidebar",
                children=create_sidebar(selected_model="linear_regression"),
            ),
            html.Div(
                id="page-content",
                className="main-content",
                children=create_linear_regression_page(),
            ),
        ],
    )