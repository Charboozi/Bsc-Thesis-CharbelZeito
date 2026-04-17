from dash import html, dcc
import plotly.graph_objects as go


def empty_figure(title: str, height: int = 360):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def create_linear_regression_page():
    return html.Div(
        className="lr-page",
        children=[
            dcc.Store(
                id="lr-data-store",
                data={
                    "x": [1, 2, 3, 4, 5, 6],
                    "y": [1.2, 2.1, 2.9, 4.2, 5.1, 5.8],
                },
            ),
            dcc.Store(id="lr-history-store", data=[]),
            dcc.Store(id="lr-animation-frames-store", data=[]),
            dcc.Store(id="lr-current-frame-store", data=0),
            dcc.Store(id="lr-is-playing-store", data=False),
            dcc.Store(id="lr-static-vis-store", data={}),
            dcc.Store(id="lr-section-boundaries-store", data=[]),

            dcc.Interval(
                id="lr-playback-interval",
                interval=60,
                disabled=True,
                n_intervals=0,
            ),

            html.H1("Linear Regression", className="page-title"),

            html.Div(
                className="lr-controls-row card",
                children=[
                    html.Div(
                        className="lr-input-group",
                        children=[
                            html.Label("x value"),
                            dcc.Input(id="lr-x-input", type="number", value=7),
                        ],
                    ),
                    html.Div(
                        className="lr-input-group",
                        children=[
                            html.Label("y value"),
                            dcc.Input(id="lr-y-input", type="number", value=7),
                        ],
                    ),
                    html.Div(
                        className="lr-input-group",
                        children=[
                            html.Label("Learning rate (α)"),
                            dcc.Input(
                                id="lr-learning-rate-input",
                                type="number",
                                value=0.01,
                                step=0.001,
                            ),
                        ],
                    ),
                    html.Div(
                        className="lr-input-group",
                        children=[
                            html.Label("Epochs"),
                            dcc.Input(
                                id="lr-epochs-input",
                                type="number",
                                value=8,
                                min=1,
                                step=1,
                            ),
                        ],
                    ),

                    html.Button("Add Point", id="lr-add-point-btn", n_clicks=0, className="action-btn"),
                    html.Button("Clear Points", id="lr-clear-points-btn", n_clicks=0, className="action-btn"),
                    html.Button("Load Sample Data", id="lr-load-sample-btn", n_clicks=0, className="action-btn"),

                    html.Button("Start Training", id="lr-start-animation-btn", n_clicks=0, className="train-btn"),

                    html.Button("Play", id="lr-play-btn", n_clicks=0, className="playback-btn", disabled=True),
                    html.Button("Pause", id="lr-pause-btn", n_clicks=0, className="playback-btn", disabled=True),
                    html.Button("Prev", id="lr-prev-btn", n_clicks=0, className="playback-btn", disabled=True),
                    html.Button("Next", id="lr-next-btn", n_clicks=0, className="playback-btn", disabled=True),
                    html.Button("Reset", id="lr-reset-btn", n_clicks=0, className="playback-btn", disabled=True),
                ],
            ),

            html.Div(
                className="lr-grid",
                children=[
                    html.Div(
                        className="card lr-card",
                        children=[
                            dcc.Graph(
                                id="lr-main-graph",
                                figure=empty_figure("Current Line, Errors, and Squares"),
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                    html.Div(
                        className="card lr-card",
                        children=[
                            dcc.Graph(
                                id="lr-contour-graph",
                                figure=empty_figure("Loss Landscape J(w, b) and Gradient Descent Path"),
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                    html.Div(
                        className="card lr-card",
                        children=[
                            dcc.Graph(
                                id="lr-squared-error-graph",
                                figure=empty_figure("Squared Error for Each Point"),
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                    html.Div(
                        className="card lr-card",
                        children=[
                            dcc.Graph(
                                id="lr-loss-graph",
                                figure=empty_figure("Mean Squared Error (MSE)"),
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                    html.Div(
                        className="card lr-card lr-explanation-card",
                        children=[
                            html.H3("What is happening now?"),
                            html.Div(id="lr-explanation-text"),
                        ],
                    ),
                ],
            ),

            html.Div(
                className="card lr-results-card",
                children=[
                    html.H3("Math Breakdown"),
                    html.Div(id="lr-results-content"),
                ],
            ),
        ],
    )