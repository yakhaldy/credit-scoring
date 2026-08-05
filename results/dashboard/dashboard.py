"""Dash dashboard for the credit-scoring model (bonus deliverable).

Enter a customer id (from application_train.csv or application_test.csv)
and get back the predicted probability of default, a human-readable
client profile, the SHAP contribution of each feature to that specific
score, and a comparison of the client against the train population.

Reuses the model-loading and figure-building logic from scripts/explain.py
so the dashboard and the standalone PDF reports stay consistent.

Run from the project root:
    python results/dashboard/dashboard.py
then open http://127.0.0.1:8050 in a browser.
"""

import importlib.util
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


def _load_explain_module():
    """Load scripts/explain.py by path, so this file needs no sys.path hacks
    or package scaffolding to reuse its model-loading and figure-building logic."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "explain.py"
    spec = importlib.util.spec_from_file_location("explain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


explain = _load_explain_module()

BUNDLE = explain.load_model()
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["features"]
TRAIN_POP = explain.load_processed("train")
TEST_POP = explain.load_processed("test")

app = Dash(__name__)
app.title = "Credit Scoring Dashboard"


def build_shap_bar(explanation_row, feature_names, top_n=15):
    values = explanation_row.values
    order = np.argsort(np.abs(values))[-top_n:]
    feats = np.array(feature_names)[order]
    vals = values[order]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color=colors))
    fig.update_layout(
        title="Top feature contributions (SHAP, log-odds scale)",
        xaxis_title="SHAP value — pushes the score up (red) or down (blue)",
        margin={"l": 220, "t": 60, "b": 40},
        height=520,
    )
    return fig


app.layout = html.Div(
    [
        html.H1("Credit Scoring — Customer Explainability Dashboard"),
        html.P("Enter a SK_ID_CURR to compute its score and explain the prediction."),
        html.Div(
            [
                dcc.Dropdown(
                    id="dataset",
                    options=[
                        {"label": "Train (has TARGET)", "value": "train"},
                        {"label": "Test (no TARGET)", "value": "test"},
                    ],
                    value="train",
                    clearable=False,
                    style={"width": "220px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                dcc.Input(
                    id="client_id",
                    type="number",
                    placeholder="SK_ID_CURR",
                    value=100002,
                    style={"marginLeft": "10px", "verticalAlign": "middle"},
                ),
                html.Button(
                    "Score client",
                    id="submit",
                    n_clicks=0,
                    style={"marginLeft": "10px", "verticalAlign": "middle"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(id="error-message", style={"color": "#c0392b", "fontWeight": "bold"}),
        html.Div(id="score-card", style={"fontSize": "20px", "fontWeight": "bold", "margin": "15px 0"}),
        dcc.Graph(id="profile-table"),
        dcc.Graph(id="shap-bar"),
        dcc.Graph(id="comparison-chart"),
    ],
    style={"maxWidth": "1200px", "margin": "auto", "fontFamily": "Arial, sans-serif", "padding": "20px"},
)


@app.callback(
    [
        Output("score-card", "children"),
        Output("profile-table", "figure"),
        Output("shap-bar", "figure"),
        Output("comparison-chart", "figure"),
        Output("error-message", "children"),
    ],
    Input("submit", "n_clicks"),
    [State("client_id", "value"), State("dataset", "value")],
)
def update_dashboard(_n_clicks, client_id, dataset):
    empty_fig = go.Figure()
    if client_id is None:
        return "", empty_fig, empty_fig, empty_fig, "Enter a customer id."

    processed = TRAIN_POP if dataset == "train" else TEST_POP
    client_row = processed[processed["SK_ID_CURR"] == client_id]
    if client_row.empty:
        return "", empty_fig, empty_fig, empty_fig, f"Client {client_id} not found in the {dataset} set."

    client_features = client_row[FEATURES]
    score, client_explanation = explain.compute_explanation(MODEL, client_features)

    true_label = None
    raw_row = None
    try:
        raw_row = explain.load_raw_row(client_id, dataset)
        if "TARGET" in raw_row.index:
            true_label = int(raw_row["TARGET"])
    except ValueError:
        pass

    outcome = explain.outcome_label(true_label)
    predicted = explain.DEFAULT_LABEL if score >= 0.5 else explain.NO_DEFAULT_LABEL
    score_text = (
        f"SK_ID_CURR {client_id} — predicted probability of default: {score:.1%} "
        f"(predicted: {predicted} | actual: {outcome})"
    )

    profile_fig = explain.build_profile_table(raw_row, score, true_label) if raw_row is not None else empty_fig
    shap_fig = build_shap_bar(client_explanation[0], FEATURES)
    comparison_fig = explain.build_comparison_figure(client_row.iloc[0], TRAIN_POP)

    return score_text, profile_fig, shap_fig, comparison_fig, ""


if __name__ == "__main__":
    app.run(debug=True)
