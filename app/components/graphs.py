from dash import dcc
import plotly.graph_objects as go


def create_main_graph_section():
    figure = go.Figure()
    figure.update_layout(
        title="Main Visualization",
        template="plotly_dark",
        height=500,
    )

    return [
        dcc.Graph(id="main-graph", figure=figure)
    ]