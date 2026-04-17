import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, no_update, html

from app.components.sidebar import create_sidebar
from app.pages.linear_regression_page import create_linear_regression_page
from app.pages.logistic_regression_page import create_logistic_regression_page
from app.models.linear_regression import (
    predict,
    mse_loss,
    generate_gradient_descent_states,
    generate_animation_frames,
)


@callback(
    Output("selected-model-store", "data"),
    Input("nav-linear-regression", "n_clicks"),
    Input("nav-logistic-regression", "n_clicks"),
    prevent_initial_call=True,
)
def update_selected_model(nav_linear_clicks, nav_logistic_clicks):
    triggered_id = ctx.triggered_id

    if triggered_id == "nav-linear-regression":
        return "linear_regression"

    if triggered_id == "nav-logistic-regression":
        return "logistic_regression"

    return no_update


@callback(
    Output("sidebar-container", "children"),
    Output("page-content", "children"),
    Input("selected-model-store", "data"),
)
def render_page(selected_model):
    if selected_model == "logistic_regression":
        return (
            create_sidebar(selected_model="logistic_regression"),
            create_logistic_regression_page(),
        )

    return (
        create_sidebar(selected_model="linear_regression"),
        create_linear_regression_page(),
    )


@callback(
    Output("lr-data-store", "data"),
    Output("lr-history-store", "data", allow_duplicate=True),
    Output("lr-animation-frames-store", "data", allow_duplicate=True),
    Output("lr-current-frame-store", "data", allow_duplicate=True),
    Output("lr-is-playing-store", "data", allow_duplicate=True),
    Output("lr-static-vis-store", "data", allow_duplicate=True),
    Output("lr-section-boundaries-store", "data", allow_duplicate=True),
    Input("lr-add-point-btn", "n_clicks"),
    Input("lr-clear-points-btn", "n_clicks"),
    Input("lr-load-sample-btn", "n_clicks"),
    State("lr-x-input", "value"),
    State("lr-y-input", "value"),
    State("lr-data-store", "data"),
    prevent_initial_call=True,
)
def update_lr_data(add_clicks, clear_clicks, sample_clicks, x_value, y_value, data):
    triggered_id = ctx.triggered_id

    if triggered_id == "lr-clear-points-btn":
        return {"x": [], "y": []}, [], [], 0, False, {}, []

    if triggered_id == "lr-load-sample-btn":
        return {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1.2, 2.1, 2.9, 4.2, 5.1, 5.8],
        }, [], [], 0, False, {}, []

    if triggered_id == "lr-add-point-btn":
        if x_value is None or y_value is None:
            return data, [], [], 0, False, {}, []

        new_x = list(data["x"])
        new_y = list(data["y"])
        new_x.append(float(x_value))
        new_y.append(float(y_value))
        return {"x": new_x, "y": new_y}, [], [], 0, False, {}, []

    return data, [], [], 0, False, {}, []


def get_section_boundaries(frames):
    if not frames:
        return []

    boundaries = [0]
    for i in range(1, len(frames)):
        prev_key = (frames[i - 1]["epoch"], frames[i - 1]["phase"])
        curr_key = (frames[i]["epoch"], frames[i]["phase"])
        if curr_key != prev_key:
            boundaries.append(i)

    return boundaries


def build_static_visual_data(x, y, history):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    all_w = [float(s["w"]) for s in history] if history else [0.0]
    all_b = [float(s["b"]) for s in history] if history else [0.0]

    w_pad = max(1.0, (max(all_w) - min(all_w)) * 1.0)
    b_pad = max(1.0, (max(all_b) - min(all_b)) * 1.0)

    w_vals = np.linspace(min(all_w) - w_pad, max(all_w) + w_pad, 40)
    b_vals = np.linspace(min(all_b) - b_pad, max(all_b) + b_pad, 40)

    z = []
    for b_test in b_vals:
        row = []
        for w_test in w_vals:
            preds = w_test * x + b_test
            row.append(float(np.mean((preds - y) ** 2)))
        z.append(row)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_pad = max(1.0, (x_max - x_min) * 0.25)

    y_pred0 = predict(x, 0.0, 0.0)
    y_min = min(float(np.min(y)), float(np.min(y_pred0)))
    y_max = max(float(np.max(y)), float(np.max(y_pred0)))
    y_pad = max(1.0, (y_max - y_min) * 0.25)

    return {
        "w_vals": list(map(float, w_vals)),
        "b_vals": list(map(float, b_vals)),
        "z": z,
        "x_min": x_min,
        "x_max": x_max,
        "x_pad": x_pad,
        "y_min_initial": y_min,
        "y_max_initial": y_max,
        "y_pad_initial": y_pad,
    }


@callback(
    Output("lr-history-store", "data", allow_duplicate=True),
    Output("lr-animation-frames-store", "data", allow_duplicate=True),
    Output("lr-current-frame-store", "data", allow_duplicate=True),
    Output("lr-is-playing-store", "data", allow_duplicate=True),
    Output("lr-static-vis-store", "data", allow_duplicate=True),
    Output("lr-section-boundaries-store", "data", allow_duplicate=True),
    Input("lr-start-animation-btn", "n_clicks"),
    State("lr-data-store", "data"),
    State("lr-learning-rate-input", "value"),
    State("lr-epochs-input", "value"),
    prevent_initial_call=True,
)
def start_lr_animation(n_clicks, data, learning_rate, epochs):
    x = data["x"]
    y = data["y"]

    if len(x) == 0:
        return [], [], 0, False, {}, []

    if learning_rate is None:
        learning_rate = 0.01

    if epochs is None or epochs < 1:
        epochs = 8

    states = generate_gradient_descent_states(
        x=x,
        y=y,
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        initial_w=0.0,
        initial_b=0.0,
    )

    frames = generate_animation_frames(states, subframes_per_phase=14)
    boundaries = get_section_boundaries(frames)
    static_vis = build_static_visual_data(x, y, states)

    return states, frames, 0, True, static_vis, boundaries


@callback(
    Output("lr-play-btn", "disabled"),
    Output("lr-pause-btn", "disabled"),
    Output("lr-prev-btn", "disabled"),
    Output("lr-next-btn", "disabled"),
    Output("lr-reset-btn", "disabled"),
    Input("lr-animation-frames-store", "data"),
)
def control_playback_button_enabled_state(frames):
    has_frames = bool(frames)
    disabled = not has_frames
    return disabled, disabled, disabled, disabled, disabled


@callback(
    Output("lr-is-playing-store", "data", allow_duplicate=True),
    Input("lr-play-btn", "n_clicks"),
    Input("lr-pause-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_play_pause_state(play_clicks, pause_clicks):
    triggered_id = ctx.triggered_id

    if triggered_id == "lr-play-btn":
        return True

    if triggered_id == "lr-pause-btn":
        return False

    return no_update


@callback(
    Output("lr-playback-interval", "disabled"),
    Input("lr-is-playing-store", "data"),
)
def control_interval(is_playing):
    return not is_playing


@callback(
    Output("lr-current-frame-store", "data", allow_duplicate=True),
    Output("lr-is-playing-store", "data", allow_duplicate=True),
    Input("lr-prev-btn", "n_clicks"),
    Input("lr-next-btn", "n_clicks"),
    Input("lr-reset-btn", "n_clicks"),
    Input("lr-playback-interval", "n_intervals"),
    State("lr-current-frame-store", "data"),
    State("lr-animation-frames-store", "data"),
    State("lr-section-boundaries-store", "data"),
    State("lr-is-playing-store", "data"),
    prevent_initial_call=True,
)
def update_current_frame(
    prev_clicks,
    next_clicks,
    reset_clicks,
    n_intervals,
    current_frame,
    frames,
    boundaries,
    is_playing,
):
    if not frames:
        return 0, False

    triggered_id = ctx.triggered_id
    max_frame = len(frames) - 1

    if triggered_id == "lr-reset-btn":
        return 0, False

    if triggered_id in ["lr-prev-btn", "lr-next-btn"]:
        boundaries = boundaries or [0]

        if triggered_id == "lr-prev-btn":
            previous_boundaries = [b for b in boundaries if b < current_frame]
            if previous_boundaries:
                return previous_boundaries[-1], False
            return 0, False

        if triggered_id == "lr-next-btn":
            next_boundaries = [b for b in boundaries if b > current_frame]
            if next_boundaries:
                return next_boundaries[0], False
            return max_frame, False

    if triggered_id == "lr-playback-interval":
        if current_frame >= max_frame:
            return max_frame, False
        return current_frame + 1, True

    return current_frame, is_playing


def make_base_2d_figure(title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def add_square_error_shapes(fig, x_vals, y_true, y_pred, progress):
    if progress <= 0:
        return

    for xi, yi, ypi in zip(x_vals, y_true, y_pred):
        side = abs(yi - ypi) * progress

        x0 = xi
        x1 = xi + side
        y0 = min(yi, ypi)
        y1 = y0 + side

        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="cyan", width=2),
            fillcolor="rgba(0,255,255,0.22)",
        )


def normalized_arrow(dx, dy, scale=0.35):
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return 0.0, 0.0

    return (dx / length) * scale, (dy / length) * scale


@callback(
    Output("lr-main-graph", "figure"),
    Output("lr-contour-graph", "figure"),
    Output("lr-squared-error-graph", "figure"),
    Output("lr-loss-graph", "figure"),
    Output("lr-explanation-text", "children"),
    Output("lr-results-content", "children"),
    Input("lr-data-store", "data"),
    Input("lr-history-store", "data"),
    Input("lr-animation-frames-store", "data"),
    Input("lr-current-frame-store", "data"),
    Input("lr-static-vis-store", "data"),
    State("lr-learning-rate-input", "value"),
)
def update_lr_visuals(data, history, frames, current_frame, static_vis, learning_rate):
    x = np.array(data["x"], dtype=float)
    y = np.array(data["y"], dtype=float)

    main_fig = make_base_2d_figure("Current Line, Errors, and Squares")
    contour_fig = make_base_2d_figure("Loss Landscape J(w, b) and Gradient Descent Path")
    squared_error_fig = make_base_2d_figure("Squared Error for Each Point")
    loss_fig = make_base_2d_figure("Mean Squared Error (MSE)")

    if learning_rate is None:
        learning_rate = 0.01

    alpha = float(learning_rate)

    if len(x) == 0:
        explanation = html.Div(
            [
                html.P("No data points yet."),
                html.P("Add some points first so the model has something to learn from."),
            ]
        )
        results = [
            html.P("Equation: not available"),
            html.P("Points: 0"),
            html.P("Loss: not available"),
        ]
        return main_fig, contour_fig, squared_error_fig, loss_fig, explanation, results

    main_fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if not history or not frames:
        w = 0.0
        b = 0.0
        y_pred = predict(x, w, b)
        errors = y - y_pred
        squared_errors = errors ** 2
        loss = mse_loss(y, y_pred)

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_pad = max(1.0, (x_max - x_min) * 0.25)

        y_min = min(float(np.min(y)), float(np.min(y_pred)))
        y_max = max(float(np.max(y)), float(np.max(y_pred)))
        y_pad = max(1.0, (y_max - y_min) * 0.25)

        x_line = np.linspace(x_min - x_pad, x_max + x_pad, 100)
        y_line = w * x_line + b

        main_fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name="Data Points",
                marker=dict(size=8),
            )
        )

        main_fig.add_trace(
            go.Scattergl(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Current Line",
                line=dict(width=3),
            )
        )

        main_fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad])
        main_fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], scaleanchor="x", scaleratio=1)

        # initial contour
        if static_vis:
            w_vals = static_vis.get("w_vals", [])
            b_vals = static_vis.get("b_vals", [])
            z = static_vis.get("z", [])
        else:
            w_vals = np.linspace(-2, 3, 40)
            b_vals = np.linspace(-2, 4, 40)
            z = []
            for b_test in b_vals:
                row = []
                for w_test in w_vals:
                    preds = w_test * x + b_test
                    row.append(float(np.mean((preds - y) ** 2)))
                z.append(row)

        contour_fig.add_trace(
            go.Contour(
                x=w_vals,
                y=b_vals,
                z=z,
                colorscale="Viridis",
                contours=dict(showlabels=True),
                colorbar=dict(title="J(w, b)"),
            )
        )

        contour_fig.add_trace(
            go.Scatter(
                x=[w],
                y=[b],
                mode="markers",
                name="Current (w, b)",
                marker=dict(size=10),
            )
        )

        contour_fig.update_xaxes(title="w (slope)")
        contour_fig.update_yaxes(title="b (intercept)")

        squared_error_fig.add_trace(
            go.Bar(
                x=[f"Point {i+1}" for i in range(len(squared_errors))],
                y=squared_errors,
                name="Squared Error",
            )
        )
        squared_error_fig.update_yaxes(title="(y - ŷ)^2")

        loss_fig.add_trace(
            go.Scatter(
                x=[0],
                y=[loss],
                mode="lines+markers",
                name="MSE",
            )
        )
        loss_fig.update_xaxes(title="Iteration")
        loss_fig.update_yaxes(title="MSE")

        explanation = html.Div(
            [
                html.P("The model starts with the line y = 0."),
                html.P("That means it predicts 0 for every x-value."),
                html.P("The model has not learned anything yet."),
                html.P("When you press Start Training, the app will show these phases:"),
                html.P("1. Current line"),
                html.P("2. Errors"),
                html.P("3. Squared errors"),
                html.P("4. MSE"),
                html.P("5. Gradient descent"),
                html.P("6. Update the line"),
            ]
        )

        results = [
            html.P("Variable guide:"),
            html.P("x = input value"),
            html.P("y = true output value"),
            html.P("ŷ = predicted output"),
            html.P("w = slope of the line"),
            html.P("b = intercept of the line"),
            html.P("error = y - ŷ"),
            html.P("MSE = average of squared errors"),
            html.P("α = learning rate"),
            html.P("dw = ∂J/∂w,  db = ∂J/∂b"),
            html.P("Current equation: y = 0x + 0"),
        ]

        return main_fig, contour_fig, squared_error_fig, loss_fig, explanation, results

    current_frame = min(current_frame, len(frames) - 1)
    frame = frames[current_frame]

    epoch = frame["epoch"]
    phase = frame["phase"]
    progress = float(frame["progress"])

    current_state = frame["current"]
    next_state = frame["next"]

    w0 = float(current_state["w"])
    b0 = float(current_state["b"])
    y_pred0 = np.array(current_state["y_pred"], dtype=float)
    errors0 = np.array(current_state["errors"], dtype=float)
    squared_errors0 = np.array(current_state["squared_errors"], dtype=float)
    loss0 = float(current_state["loss"])
    dw0 = float(current_state["dw"])
    db0 = float(current_state["db"])

    w1 = float(next_state["w"])
    b1 = float(next_state["b"])
    loss1 = float(next_state["loss"])

    previous_losses = frame["previous_losses"]

    if phase == "line":
        display_w = w0
        display_b = b0
        display_y_pred = y_pred0
        residual_progress = 0.0
        square_progress = 0.0
        displayed_loss = loss0
        displayed_squared_errors = np.zeros_like(squared_errors0)

    elif phase == "residuals":
        display_w = w0
        display_b = b0
        display_y_pred = y_pred0
        residual_progress = progress
        square_progress = 0.0
        displayed_loss = loss0
        displayed_squared_errors = np.zeros_like(squared_errors0)

    elif phase == "squared_errors":
        display_w = w0
        display_b = b0
        display_y_pred = y_pred0
        residual_progress = 1.0
        square_progress = progress
        displayed_loss = loss0
        displayed_squared_errors = squared_errors0 * progress

    elif phase == "mse":
        display_w = w0
        display_b = b0
        display_y_pred = y_pred0
        residual_progress = 1.0
        square_progress = 1.0
        displayed_loss = loss0
        displayed_squared_errors = squared_errors0

    elif phase == "gradient":
        display_w = w0
        display_b = b0
        display_y_pred = y_pred0
        residual_progress = 1.0
        square_progress = 1.0
        displayed_loss = loss0
        displayed_squared_errors = squared_errors0

    elif phase == "update":
        display_w = (1 - progress) * w0 + progress * w1
        display_b = (1 - progress) * b0 + progress * b1
        display_y_pred = predict(x, display_w, display_b)
        residual_progress = 0.0
        square_progress = 0.0
        displayed_loss = (1 - progress) * loss0 + progress * loss1
        displayed_squared_errors = squared_errors0

    else:  # final
        display_w = w0
        display_b = b0
        display_y_pred = y_pred0
        residual_progress = 0.0
        square_progress = 0.0
        displayed_loss = loss0
        displayed_squared_errors = squared_errors0

    # ---------------- Main graph ----------------
    x_min = static_vis.get("x_min", float(np.min(x)))
    x_max = static_vis.get("x_max", float(np.max(x)))
    x_pad = static_vis.get("x_pad", max(1.0, (x_max - x_min) * 0.25))

    y_min = min(float(np.min(y)), float(np.min(display_y_pred)))
    y_max = max(float(np.max(y)), float(np.max(display_y_pred)))
    y_pad = max(1.0, (y_max - y_min) * 0.25)

    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 100)
    y_line = display_w * x_line + display_b

    main_fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            name="Data Points",
            marker=dict(size=8),
        )
    )

    main_fig.add_trace(
        go.Scattergl(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Current Line",
            line=dict(width=3),
        )
    )

    if phase in ["residuals", "squared_errors", "mse", "gradient"]:
        main_fig.add_trace(
            go.Scattergl(
                x=x,
                y=display_y_pred,
                mode="markers",
                name="Predicted Points",
                marker=dict(size=7, symbol="x"),
            )
        )

    if residual_progress > 0:
        for xi, yi, ypi in zip(x, y, display_y_pred):
            y_end = ypi + residual_progress * (yi - ypi)
            main_fig.add_shape(
                type="line",
                x0=xi,
                y0=ypi,
                x1=xi,
                y1=y_end,
                line=dict(color="orange", width=2),
            )

    add_square_error_shapes(main_fig, x, y, display_y_pred, square_progress)

    main_fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad])
    main_fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], scaleanchor="x", scaleratio=1)

    # ---------------- Contour plot (J(w, b)) ----------------
    w_vals = static_vis.get("w_vals", [])
    b_vals = static_vis.get("b_vals", [])
    z = static_vis.get("z", [])

    contour_fig.add_trace(
        go.Contour(
            x=w_vals,
            y=b_vals,
            z=z,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="J(w, b)"),
        )
    )

    contour_fig.update_xaxes(title="w (slope)")
    contour_fig.update_yaxes(title="b (intercept)")

    prev_ws = [float(s["w"]) for s in history[: epoch + 1]]
    prev_bs = [float(s["b"]) for s in history[: epoch + 1]]

    if phase == "update":
        path_ws = prev_ws + [display_w]
        path_bs = prev_bs + [display_b]
    elif phase == "final":
        path_ws = [float(s["w"]) for s in history]
        path_bs = [float(s["b"]) for s in history]
    else:
        path_ws = prev_ws
        path_bs = prev_bs

    if len(path_ws) > 1:
        contour_fig.add_trace(
            go.Scatter(
                x=path_ws,
                y=path_bs,
                mode="lines+markers",
                name="Gradient descent path",
                line=dict(width=2),
                marker=dict(size=7),
            )
        )

    contour_fig.add_trace(
        go.Scatter(
            x=[display_w],
            y=[display_b],
            mode="markers",
            name="Current (w, b)",
            marker=dict(size=10),
        )
    )

    # During gradient phase: show current point, next point, and arrows
    if phase == "gradient":
        contour_fig.add_trace(
            go.Scatter(
                x=[w1],
                y=[b1],
                mode="markers",
                name="Next (w, b)",
                marker=dict(size=10, symbol="diamond"),
            )
        )

        # gradient direction (red)
        grad_dx, grad_dy = normalized_arrow(dw0, db0, scale=0.45)

        contour_fig.add_annotation(
            x=w0 + grad_dx,
            y=b0 + grad_dy,
            ax=w0,
            ay=b0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=3,
            arrowwidth=2,
            arrowcolor="red",
            text="∇J = (dw, db)",
            font=dict(color="red"),
        )

        # update direction (green) = -alpha * gradient
        contour_fig.add_annotation(
            x=w1,
            y=b1,
            ax=w0,
            ay=b0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=3,
            arrowwidth=2,
            arrowcolor="lime",
            text="-α∇J",
            font=dict(color="lime"),
        )

    if phase == "update":
        contour_fig.add_annotation(
            x=display_w,
            y=display_b,
            ax=w0,
            ay=b0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=3,
            arrowwidth=2,
            arrowcolor="lime",
            text="update",
            font=dict(color="lime"),
        )

    # ---------------- Squared error graph ----------------
    squared_error_fig.add_trace(
        go.Bar(
            x=[f"Point {i+1}" for i in range(len(displayed_squared_errors))],
            y=displayed_squared_errors,
            name="Squared Error",
        )
    )
    squared_error_fig.update_yaxes(title="(y - ŷ)^2")

    # ---------------- MSE graph ----------------
    loss_fig.update_xaxes(title="Iteration")
    loss_fig.update_yaxes(title="MSE")

    mse_x = list(range(len(previous_losses)))
    mse_y = list(previous_losses)

    if phase == "mse":
        mse_x.append(epoch)
        mse_y.append(loss0 * progress)

    elif phase in ["gradient", "update"]:
        mse_x.append(epoch)
        mse_y.append(loss0)

    elif phase == "final":
        mse_x.append(epoch)
        mse_y.append(loss0)

    if not mse_x:
        mse_x = [0]
        mse_y = [loss0]

    loss_fig.add_trace(
        go.Scatter(
            x=mse_x,
            y=mse_y,
            mode="lines+markers",
            name="MSE",
        )
    )

    # ---------------- Explanation panel ----------------
    if phase == "line":
        explanation = html.Div(
            [
                html.P(f"Iteration {epoch}: start with the current line."),
                html.P("Linear regression tries to draw a straight line that fits the data."),
                html.P("The equation of the line is:  ŷ = wx + b"),
                html.P("ŷ (y-hat) means the model's predicted y-value."),
                html.P("w is the slope of the line."),
                html.P("b is the intercept: where the line crosses the y-axis."),
                html.P(f"At this moment, w = {w0:.3f} and b = {b0:.3f}."),
                html.P("Using those values, the model draws the current line."),
            ]
        )

    elif phase == "residuals":
        explanation = html.Div(
            [
                html.P(f"Iteration {epoch}: calculate the error for each point."),
                html.P("For each x-value, the model predicts a y-value using the line."),
                html.P("Then it compares that prediction to the true y-value."),
                html.P("The orange vertical distance is the error."),
                html.P("Math: error = y - ŷ"),
                html.P("If the predicted value is far from the true value, the error is large."),
                html.P("If the predicted value is close to the true value, the error is small."),
            ]
        )

    elif phase == "squared_errors":
        explanation = html.Div(
            [
                html.P(f"Iteration {epoch}: square each error."),
                html.P("To avoid positive and negative errors canceling out, we square them."),
                html.P("Math: squared error = (y - ŷ)^2"),
                html.P("That is why a square is drawn for each error."),
                html.P("The side length of the square is the size of the error."),
                html.P("The area of the square is the squared error."),
                html.P("Squaring also makes large errors count much more than small errors."),
            ]
        )

    elif phase == "mse":
        explanation = html.Div(
            [
                html.P(f"Iteration {epoch}: compute Mean Squared Error (MSE)."),
                html.P("Take all the squared errors and average them."),
                html.P("Math: MSE = (1/n) Σ (y - ŷ)^2"),
                html.P("Here, n is the number of data points."),
                html.P("MSE is also called the loss or cost."),
                html.P("It is one number that tells us how bad the current line is overall."),
                html.P("A smaller MSE means the line fits the data better."),
            ]
        )

    elif phase == "gradient":
        explanation = html.Div(
            [
                html.P(f"Iteration {epoch}: use gradient descent to decide how to move the line."),
                html.P("The contour plot shows the loss J(w, b) for many combinations of w and b."),
                html.P("Each point on that plot represents one possible line."),
                html.P("The current line is at the current (w, b)."),
                html.P("Gradient descent uses derivatives to see how the loss changes."),
                html.P("dw = ∂J/∂w tells us how the loss changes when we change the slope w."),
                html.P("db = ∂J/∂b tells us how the loss changes when we change the intercept b."),
                html.P("The red arrow is the gradient ∇J = (dw, db)."),
                html.P("The gradient points in the direction of increasing loss."),
                html.P("To reduce the loss, we move in the opposite direction: -∇J."),
                html.P("That opposite direction is shown by the green arrow."),
                html.P("This step is called gradient descent because we move downhill on the loss landscape."),
            ]
        )

    elif phase == "update":
        explanation = html.Div(
            [
                html.P(f"Iteration {epoch}: update w and b."),
                html.P("We use the learning rate α to control how large the step should be."),
                html.P("Update rule: w_new = w - α·dw"),
                html.P("Update rule: b_new = b - α·db"),
                html.P(f"Here α = {alpha:.3f}."),
                html.P("If dw is positive, w decreases."),
                html.P("If dw is negative, w increases."),
                html.P("If db is positive, b decreases."),
                html.P("If db is negative, b increases."),
                html.P("As the line moves, the point in the contour plot moves as well."),
                html.P("Repeating this process causes the path to move toward lower loss"),
            ]
        )

    else:  # final
        explanation = html.Div(
            [
                html.P("Training is complete."),
                html.P("The model repeatedly measured errors, squared them, averaged them into MSE,"),
                html.P("computed the gradient, and updated w and b."),
                html.P("Over time, the gradient descent path moved toward a point with smaller loss."),
                html.P("The final line is the model's fitted line."),
            ]
        )

    # ---------------- Math Breakdown ----------------
    phase_names = {
        "line": "1/6 Current line",
        "residuals": "2/6 Errors",
        "squared_errors": "3/6 Squared errors",
        "mse": "4/6 MSE",
        "gradient": "5/6 Gradient descent",
        "update": "6/6 Update line",
        "final": "Final state",
    }

    point_rows = []
    for i, (xi, yi, yhat_i, err_i, se_i) in enumerate(
        zip(x, y, y_pred0, errors0, squared_errors0), start=1
    ):
        point_rows.append(
            html.Tr(
                [
                    html.Td(f"Point {i}"),
                    html.Td(f"{xi:.3f}"),
                    html.Td(f"{yi:.3f}"),
                    html.Td(f"{yhat_i:.3f}"),
                    html.Td(f"{err_i:.3f}"),
                    html.Td(f"{se_i:.3f}"),
                ]
            )
        )

    mse_formula = " + ".join([f"{v:.3f}" for v in squared_errors0])
    mse_formula = f"MSE = ({mse_formula}) / {len(squared_errors0)} = {loss0:.5f}"

    results = html.Div(
        [
            html.H4("Variable guide"),
            html.P("x = input value"),
            html.P("y = true output value"),
            html.P("ŷ = predicted output (y-hat)"),
            html.P("w = slope of the line"),
            html.P("b = intercept of the line"),
            html.P("error = y - ŷ"),
            html.P("squared error = (y - ŷ)^2"),
            html.P("MSE = average of squared errors"),
            html.P(f"α = learning rate = {alpha:.3f}"),
            html.P("dw = ∂J/∂w"),
            html.P("db = ∂J/∂b"),

            html.H4("Current phase"),
            html.P(f"{phase_names.get(phase, phase)}"),

            html.H4("Current equation"),
            html.P(f"ŷ = {display_w:.3f}x + {display_b:.3f}"),

            html.H4("Point-by-point calculations"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Point"),
                                html.Th("x"),
                                html.Th("y"),
                                html.Th("ŷ"),
                                html.Th("error = y - ŷ"),
                                html.Th("(y - ŷ)^2"),
                            ]
                        )
                    ),
                    html.Tbody(point_rows),
                ],
                className="lr-math-table",
            ),

            html.H4("Loss calculation"),
            html.P(mse_formula),

            html.H4("Gradient descent"),
            html.P(f"Current MSE J(w, b) = {loss0:.5f}"),
            html.P(f"dw = ∂J/∂w = {dw0:.5f}"),
            html.P(f"db = ∂J/∂b = {db0:.5f}"),
            html.P("These tell us how the loss changes when w or b changes."),
            html.P("The gradient vector is ∇J = (dw, db)."),
            html.P("To reduce the loss, we move in the opposite direction -∇J."),

            html.H4("Update rule"),
            html.P(f"w_new = w - α·dw = {w0:.3f} - {alpha:.3f}·({dw0:.5f}) = {w1:.3f}"),
            html.P(f"b_new = b - α·db = {b0:.3f} - {alpha:.3f}·({db0:.5f}) = {b1:.3f}"),
        ]
    )

    return main_fig, contour_fig, squared_error_fig, loss_fig, explanation, results