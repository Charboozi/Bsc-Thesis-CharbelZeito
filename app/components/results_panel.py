from dash import html


def create_results_panel():
    return [
        html.H3("Results"),
        html.P("Final equation, loss, convergence info, and summary will appear here.")
    ]