import numpy as np


def predict(x, w, b):
    x = np.array(x, dtype=float)
    return w * x + b


def mse_loss(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_gradients(x, y, w, b):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    n = len(x)
    y_pred = predict(x, w, b)

    # Gradient of MSE with respect to w and b
    dw = (2 / n) * np.sum((y_pred - y) * x)
    db = (2 / n) * np.sum(y_pred - y)

    return float(dw), float(db)


def generate_gradient_descent_states(
    x,
    y,
    learning_rate=0.01,
    epochs=8,
    initial_w=0.0,
    initial_b=0.0,
):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    states = []

    w = float(initial_w)
    b = float(initial_b)

    for epoch in range(epochs + 1):
        y_pred = predict(x, w, b)
        errors = y - y_pred
        squared_errors = errors ** 2
        loss = float(np.mean(squared_errors))
        dw, db = compute_gradients(x, y, w, b)

        states.append(
            {
                "epoch": epoch,
                "w": float(w),
                "b": float(b),
                "y_pred": y_pred.tolist(),
                "errors": errors.tolist(),
                "squared_errors": squared_errors.tolist(),
                "loss": float(loss),
                "dw": float(dw),
                "db": float(db),
            }
        )

        if epoch < epochs:
            w = w - learning_rate * dw
            b = b - learning_rate * db

    return states


def generate_animation_frames(states, subframes_per_phase=14):
    """
    Each iteration is divided into clear teaching phases:
    1) current line
    2) errors
    3) squared errors
    4) MSE
    5) gradient explanation
    6) parameter update
    """
    if not states or len(states) < 2:
        return []

    frames = []

    if subframes_per_phase < 2:
        subframes = [1.0]
    else:
        subframes = np.linspace(0, 1, subframes_per_phase)

    for epoch in range(len(states) - 1):
        current_state = states[epoch]
        next_state = states[epoch + 1]

        previous_losses = [s["loss"] for s in states[:epoch]]

        for progress in subframes:
            frames.append(
                {
                    "epoch": epoch,
                    "phase": "line",
                    "progress": float(progress),
                    "current": current_state,
                    "next": next_state,
                    "previous_losses": previous_losses,
                }
            )

        for progress in subframes:
            frames.append(
                {
                    "epoch": epoch,
                    "phase": "residuals",
                    "progress": float(progress),
                    "current": current_state,
                    "next": next_state,
                    "previous_losses": previous_losses,
                }
            )

        for progress in subframes:
            frames.append(
                {
                    "epoch": epoch,
                    "phase": "squared_errors",
                    "progress": float(progress),
                    "current": current_state,
                    "next": next_state,
                    "previous_losses": previous_losses,
                }
            )

        for progress in subframes:
            frames.append(
                {
                    "epoch": epoch,
                    "phase": "mse",
                    "progress": float(progress),
                    "current": current_state,
                    "next": next_state,
                    "previous_losses": previous_losses,
                }
            )

        for progress in subframes:
            frames.append(
                {
                    "epoch": epoch,
                    "phase": "gradient",
                    "progress": float(progress),
                    "current": current_state,
                    "next": next_state,
                    "previous_losses": previous_losses + [current_state["loss"]],
                }
            )

        for progress in subframes:
            frames.append(
                {
                    "epoch": epoch,
                    "phase": "update",
                    "progress": float(progress),
                    "current": current_state,
                    "next": next_state,
                    "previous_losses": previous_losses + [current_state["loss"]],
                }
            )

    final_state = states[-1]

    frames.append(
        {
            "epoch": final_state["epoch"],
            "phase": "final",
            "progress": 1.0,
            "current": final_state,
            "next": final_state,
            "previous_losses": [s["loss"] for s in states[:-1]],
        }
    )

    return frames