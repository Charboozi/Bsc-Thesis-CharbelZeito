import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=100, initial_w=0.0, initial_b=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = initial_w
        self.b = initial_b
        self.history = []

    def predict(self, x):
        return self.w * x + self.b

    def compute_loss(self, x, y):
        y_pred = self.predict(x)
        return np.mean((y_pred - y) ** 2)

    def fit(self, x, y):
        n = len(x)
        self.history = []

        for epoch in range(self.epochs):
            y_pred = self.predict(x)

            dw = (2 / n) * np.sum((y_pred - y) * x)
            db = (2 / n) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            loss = self.compute_loss(x, y)

            self.history.append({
                "epoch": epoch,
                "w": self.w,
                "b": self.b,
                "loss": loss,
                "dw": dw,
                "db": db,
                "y_pred": self.predict(x).tolist(),
            })

        return self