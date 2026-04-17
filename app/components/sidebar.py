from dash import html


def nav_item(label: str, item_id: str, selected: bool = False, disabled: bool = False):
    class_name = "sidebar-item"
    if selected:
        class_name += " selected"
    if disabled:
        class_name += " disabled"

    return html.Div(
        label,
        id=item_id,
        className=class_name,
        n_clicks=0 if not disabled else None,
    )


def create_sidebar(selected_model: str = "linear_regression"):
    return html.Div(
        className="sidebar-inner",
        children=[
            html.H2("ML Visualizer", className="sidebar-title"),

            html.Div(
                className="sidebar-section",
                children=[
                    html.Div("Supervised", className="sidebar-section-title"),
                    nav_item(
                        "Linear Regression",
                        "nav-linear-regression",
                        selected=selected_model == "linear_regression",
                    ),
                    nav_item(
                        "Logistic Regression",
                        "nav-logistic-regression",
                        selected=selected_model == "logistic_regression",
                    ),
                ],
            ),

            html.Div(
                className="sidebar-section",
                children=[
                    html.Div("Unsupervised", className="sidebar-section-title"),
                    nav_item("K-Means", "nav-kmeans", disabled=True),
                    nav_item("PCA", "nav-pca", disabled=True),
                ],
            ),
        ],
    )