from dash import Dash, html
from app.layout import create_layout

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "ML Visualizer"

app.layout = create_layout()

server = app.server

if __name__ == "__main__":
    app.run(debug=True)