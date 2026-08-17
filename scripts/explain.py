import argparse
import io
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from matplotlib.backends.backend_pdf import PdfPages
from pandas.errors import PerformanceWarning
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=PerformanceWarning)

DATA_DIR = "data"
PROCESSED_DIR = "data/processed"
MODEL_PATH = "results/model/my_own_model.pkl"
RESULTS_PATH = "results/clients_outputs/"


RAW_DISPLAY_COLS = [
    "SK_ID_CURR",
    "CODE_GENDER",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "NAME_CONTRACT_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

COMPARISON_VARS = [
    "AGE_YEARS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "EXT_SOURCE_MEAN",
]

DEFAULT_LABEL = "Default"
NO_DEFAULT_LABEL = "No default"
UNKNOWN_LABEL = "Unknown (test set)"


def outcome_label(true_label):
    if true_label == 1:
        return DEFAULT_LABEL
    if true_label == 0:
        return NO_DEFAULT_LABEL
    return UNKNOWN_LABEL


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_processed(dataset):
    return pd.read_pickle(f"{PROCESSED_DIR}/{dataset}.pkl")


def load_raw_row(client_id, dataset):
    path = f"{DATA_DIR}/application_{dataset}.csv"
    cols = list(RAW_DISPLAY_COLS)
    if dataset == "train":
        cols = cols + ["TARGET"]
    raw = pd.read_csv(path, usecols=cols)
    row = raw[raw["SK_ID_CURR"] == client_id]
    if row.empty:
        # raise ValueError(f"Client {client_id} not found in application_{dataset}.csv")
        print(f"Client {client_id} not found in application_{dataset}.csv")
        return None
    return row.iloc[0]


def compute_explanation(model, client_features):
    """Returns (score, shap.Explanation for the single client)."""
    score = model.predict_proba(client_features)[:, 1][0]
    explainer = shap.TreeExplainer(model)
    explanation = explainer(client_features)
    return score, explanation


def fig_to_array(fig, width=1000, height=550, scale=2):
    png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    return plt.imread(io.BytesIO(png_bytes), format="png")


def build_profile_table(raw_row, score, true_label=None):
    def fmt(val, kind="raw"):
        if pd.isna(val):
            return "N/A"
        if kind == "money":
            return f"{val:,.0f}"
        if kind == "years":
            return f"{val:.1f}"
        if kind == "pct":
            return f"{val:.3f}"
        return str(val)

    age_years = -raw_row["DAYS_BIRTH"] / 365
    years_employed = (
        None if raw_row["DAYS_EMPLOYED"] == 365243 else -raw_row["DAYS_EMPLOYED"] / 365
    )

    rows = [
        ("Predicted probability of default", f"{score:.1%}"),
        ("Actual outcome (TARGET)", outcome_label(true_label)),
        ("Gender", fmt(raw_row["CODE_GENDER"])),
        ("Age (years)", fmt(age_years, "years")),
        ("Years employed", fmt(years_employed, "years") if years_employed is not None else "N/A (not employed)"),
        ("Education", fmt(raw_row["NAME_EDUCATION_TYPE"])),
        ("Family status", fmt(raw_row["NAME_FAMILY_STATUS"])),
        ("Income type", fmt(raw_row["NAME_INCOME_TYPE"])),
        ("Occupation", fmt(raw_row["OCCUPATION_TYPE"])),
        ("Number of children", fmt(raw_row["CNT_CHILDREN"])),
        ("Family members", fmt(raw_row["CNT_FAM_MEMBERS"])),
        ("Loan type", fmt(raw_row["NAME_CONTRACT_TYPE"])),
        ("Annual income", fmt(raw_row["AMT_INCOME_TOTAL"], "money")),
        ("Credit amount", fmt(raw_row["AMT_CREDIT"], "money")),
        ("Loan annuity", fmt(raw_row["AMT_ANNUITY"], "money")),
        ("Goods price", fmt(raw_row["AMT_GOODS_PRICE"], "money")),
        ("EXT_SOURCE_1", fmt(raw_row["EXT_SOURCE_1"], "pct")),
        ("EXT_SOURCE_2", fmt(raw_row["EXT_SOURCE_2"], "pct")),
        ("EXT_SOURCE_3", fmt(raw_row["EXT_SOURCE_3"], "pct")),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Variable", "Value"],
                    "fill_color": "#2c3e50",
                    "font": {"color": "white", "size": 13},
                    "align": "left",
                },
                cells={
                    "values": [[r[0] for r in rows], [r[1] for r in rows]],
                    "fill_color": [["#f8f9f9"] * len(rows), ["white"] * len(rows)],
                    "align": "left",
                    "font": {"size": 12},
                    "height": 26,
                },
            )
        ]
    )
    fig.update_layout(
        title=f"Client profile — SK_ID_CURR {int(raw_row['SK_ID_CURR'])}",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def build_comparison_figure(client_row, train_pop):
    n = len(COMPARISON_VARS)
    fig = make_subplots(rows=1, cols=n, subplot_titles=COMPARISON_VARS)

    for i, var in enumerate(COMPARISON_VARS, start=1):
        # Clip to the 1st-99th percentile so a few extreme outliers don't
        # flatten the box plots (same convention as the EDA notebook).
        col_data = train_pop[var].dropna()
        p1, p99 = col_data.quantile(0.01), col_data.quantile(0.99)
        for target_val, color, label in [(0, "#27ae60", NO_DEFAULT_LABEL), (1, "#e74c3c", DEFAULT_LABEL)]:
            data = train_pop.loc[train_pop["TARGET"] == target_val, var].dropna().clip(p1, p99)
            fig.add_trace(
                go.Box(y=data, name=label, marker_color=color, showlegend=(i == 1), boxpoints=False),
                row=1,
                col=i,
            )
        client_val = client_row[var]
        if pd.notna(client_val):
            fig.add_trace(
                go.Scatter(
                    x=[NO_DEFAULT_LABEL],
                    y=[np.clip(client_val, p1, p99)],
                    mode="markers",
                    marker={"color": "black", "size": 13, "symbol": "diamond"},
                    name="This client",
                    showlegend=(i == 1),
                ),
                row=1,
                col=i,
            )

    fig.update_layout(
        title="This client vs. the train population (by outcome)",
        height=500,
        width=1400,
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
    )
    return fig


def render_client_report(client_id, dataset, output_path, true_label_override=None):
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]

    processed = load_processed(dataset)
    client_processed = processed[processed["SK_ID_CURR"] == client_id]
    if client_processed.empty:
        # raise ValueError(f"Client {client_id} not found in processed {dataset} set")
        print(f"Client {client_id} not found in processed {dataset} set")
        return
    client_features = client_processed[features]

    raw_row = load_raw_row(client_id, dataset)
    true_label = true_label_override
    if true_label is None and "TARGET" in raw_row.index:
        true_label = int(raw_row["TARGET"])

    score, explanation = compute_explanation(model, client_features)

    train_pop = load_processed("train")

    profile_fig = build_profile_table(raw_row, score, true_label)
    comparison_fig = build_comparison_figure(client_processed.iloc[0], train_pop)

    with PdfPages(output_path) as pdf:
        # Page 1 — client profile table (Plotly -> static image)
        plt.figure(figsize=(11, 8.5))
        plt.imshow(fig_to_array(profile_fig, width=900, height=750))
        plt.axis("off")
        pdf.savefig()
        plt.close()

        # Page 2 — SHAP force plot (base value -> f(x), pushed by each feature)
        shap.plots.force(
            explanation.base_values[0], explanation.values[0], client_features.iloc[0], matplotlib=True, show=False
        )
        plt.suptitle(f"SHAP force plot — SK_ID_CURR {client_id} (score={score:.1%})", y=1.4)
        pdf.savefig(bbox_inches="tight")
        plt.close()

        # Page 3 — SHAP waterfall (same values, full ranked breakdown — more
        # readable than the force plot once more than a handful of features matter)
        plt.figure(figsize=(11, 8.5))
        shap.plots.waterfall(explanation[0], max_display=15, show=False)
        plt.title(f"SHAP contributions — SK_ID_CURR {client_id} (score={score:.1%})", x=0.5, y=1.05)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 4 — comparison vs population (Plotly -> static image)
        plt.figure(figsize=(11, 5))
        plt.imshow(fig_to_array(comparison_fig, width=1400, height=500))
        plt.axis("off")
        pdf.savefig()
        plt.close()

    predicted = DEFAULT_LABEL if score >= 0.5 else NO_DEFAULT_LABEL
    actual = outcome_label(true_label)
    print(f"Client {client_id} ({dataset}): score={score:.4f} -> predicted={predicted} | actual={actual}")
    print(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a local interpretability PDF for one client.")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.output = RESULTS_PATH + args.output

    render_client_report(args.client_id, args.dataset, args.output)


if __name__ == "__main__":
    main()
