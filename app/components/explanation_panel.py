from dash import html


def create_explanation_panel():
    return [
        html.H3("What is happening now?"),
        html.P(
            "The explanation panel will update during training and describe each step."
        ),
    ]