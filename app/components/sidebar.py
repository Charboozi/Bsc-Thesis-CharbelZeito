from dash import html, dcc


def create_sidebar():
    return [
        html.H2("Models"),
        html.Label("Learning type"),
        dcc.Dropdown(
            id="learning-type",
            options=[
                {"label": "Supervised", "value": "supervised"},
                {"label": "Unsupervised", "value": "unsupervised"},
            ],
            value="supervised",
            clearable=False,
        ),
        html.Br(),
        html.Label("Model"),
        dcc.Dropdown(
            id="model-select",
            options=[
                {"label": "Linear Regression", "value": "linear_regression"},
                {"label": "Logistic Regression", "value": "logistic_regression"},
            ],
            value="linear_regression",
            clearable=False,
        ),
        html.Br(),
        html.Label("Learning rate"),
        dcc.Slider(
            id="learning-rate",
            min=0.001,
            max=1.0,
            step=0.001,
            value=0.01,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Br(),
        html.Label("Epochs"),
        dcc.Input(id="epochs", type="number", value=100, min=1, step=1),
        html.Br(),
        html.Br(),
        html.Button("Start", id="start-button", n_clicks=0),
        html.Button("Pause", id="pause-button", n_clicks=0, style={"marginLeft": "10px"}),
        html.Button("Reset", id="reset-button", n_clicks=0, style={"marginLeft": "10px"}),
    ]