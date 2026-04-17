from dash import html


def create_logistic_regression_page():
    return html.Div(
        children=[
            html.H1("Logistic Regression", className="page-title"),
        ]
    )