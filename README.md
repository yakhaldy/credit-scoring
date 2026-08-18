# Credit Scoring — Home Credit Default Risk

**Username:** `yakhaldyma032026` — `yakhaldy 01EDU_MA_03_2026`
(as recorded in [`username.txt`](username.txt), unmodified since day 1)

## Project

An interpretable probability-of-default model for Home Credit's loan
applicants. Given a loan application, the model outputs the probability that
the client will have payment difficulties, and — just as importantly — *why*.

Two audiences are served by design:

- **Compliance / regulators** — a global feature-importance ranking of what the
  model actually relies on (`results/model/feature_importance.png`), plus a
  learning curve showing the train/validation gap
  (`results/model/learning_curve.png`).
- **Customer service** — a per-client SHAP breakdown explaining how each
  feature pushed a given applicant's score up or down, as a PDF report
  (`results/clients_outputs/`) or as an interactive dashboard.

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
— `application_train.csv` / `application_test.csv` plus 6 supplementary tables
(`bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`,
`credit_card_balance`, `installments_payments`). Column-by-column definitions
ship with the data in `data/HomeCredit_columns_description.csv`.

## Results

| Metric | Value |
| --- | --- |
| Validation AUC | **0.7796** |
| Model | LightGBM (`LGBMClassifier`, `class_weight="balanced"`) |
| Features | 283 (engineered + one-hot encoded) |
| Best iteration | 1046 / 2000 (early stopping, 100 rounds patience) |
| Validation split | Stratified 80/20 held-out split of `application_train.csv` |

This clears the project's 0.55 minimum and 0.62 target comfortably. Metrics and
the full feature list are written to `results/model/model_report.txt` by
`train.py` on every run.

The Kaggle submission file is `results/prediction.csv` (48,744 scored
applications) — upload it to the competition to read the public-leaderboard AUC
on the unlabelled test set.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Data** — the raw CSVs are gitignored (too large for the repo). Download
[home-credit-default-risk.zip](https://assets.01-edu.org/ai-branch/project5/home-credit-default-risk.zip)
and unzip it into `data/` so that `data/application_train.csv`,
`data/bureau.csv`, etc. exist at the paths the scripts read.

All commands below are run **from the repository root** — the scripts use
relative paths such as `data/` and `results/`.

## How to run

### 1. Feature engineering

Reads the 8 raw CSVs, aggregates the 6 supplementary tables per client
(`SK_ID_CURR`), engineers ratios and flags (`AGE_YEARS`,
`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EXT_SOURCE_MEAN`, days-late on
installments, bureau DPD counts…), one-hot encodes the categoricals, downcasts
to 32-bit to keep memory in check, and writes the processed matrices:

```bash
python scripts/preprocess.py
# -> data/processed/train.pkl
# -> data/processed/test.pkl
```

`scripts/preprocess-v2.py` is a variant of the same pipeline with extra
`inspect_dataframe()` diagnostics printed at each stage — useful when
debugging the feature build, not required for the pipeline.

### 2. Training

Stratified 80/20 split, LightGBM with early stopping on validation AUC, saves
the model bundle (model + feature list + validation AUC) and the two required
charts:

```bash
python scripts/train.py
# -> results/model/my_own_model.pkl
# -> results/model/learning_curve.png
# -> results/model/feature_importance.png
# -> results/model/model_report.txt
```

### 3. Prediction

Scores `application_test.csv` and writes the Kaggle submission file:

```bash
python scripts/predict.py
# -> results/prediction.csv
```

### 4. Local interpretability (per-client report)

SHAP explanation + client profile + comparison against the train population,
as a 4-page PDF, for any client id:

```bash
python scripts/explain.py --client_id 100002 --dataset train  --output client_train.pdf
python scripts/explain.py --client_id 100067 --dataset test   --output client_test.pdf
```

- `--client_id` — any `SK_ID_CURR` present in the chosen dataset.
- `--dataset` — `train` (has a known `TARGET`, so the report shows
  predicted *vs* actual) or `test` (outcome unknown).
- `--output` — **filename only**; it is written into
  `results/clients_outputs/`.

Each PDF contains: the client profile table, a SHAP force plot, a ranked SHAP
waterfall of the top 15 contributions, and box plots placing the client against
the defaulting / non-defaulting populations.

Two pre-generated examples are committed in `results/clients_outputs/`:
`client_train.pdf` (train set, actual outcome shown) and `client_test.pdf`
(test set, outcome unknown).

### 5. Exploratory data analysis

```bash
jupyter notebook results/feature_engineering/EDA.ipynb
```

Target imbalance, feature distributions, `EXT_SOURCE_*` separation power,
default rates per category, and correlations with the target — exported
alongside as PNGs in the same folder. `EDA-V2.ipynb` is an earlier iteration
kept for reference.

### 6. Dashboard (bonus)

The same score / profile / SHAP / comparison views as `explain.py`, interactive
in the browser, for any client id:

```bash
python results/dashboard/dashboard.py
# -> open http://127.0.0.1:8050
```

Pick the train or test set from the dropdown, enter an `SK_ID_CURR`, and press
**Score client**. It reuses `scripts/explain.py` directly, so the dashboard and
the PDFs can never drift apart.

## Project structure

```
data/                             raw CSVs (gitignored) + processed/ pickles (generated, gitignored)
scripts/
  preprocess.py                   feature engineering pipeline -> data/processed/*.pkl
  preprocess-v2.py                same pipeline with per-stage inspection output
  train.py                        training, learning curve, global feature importance
  predict.py                      scoring of application_test.csv -> submission
  explain.py                      local interpretability (SHAP + Plotly) for one client
results/
  prediction.csv                  Kaggle submission
  model/                          my_own_model.pkl, learning_curve.png,
                                  feature_importance.png, model_report.txt
  feature_engineering/            EDA.ipynb, EDA-V2.ipynb + exported EDA charts
  clients_outputs/                per-client SHAP + Plotly PDF reports
  dashboard/dashboard.py          Dash app (bonus) — interactive version of explain.py
requirements.txt                  pinned dependencies
username.txt                      submission identity
```

Run order matters: `preprocess.py` → `train.py` → `predict.py` / `explain.py` /
`dashboard.py`. Every step after the first depends on the artifacts produced by
the ones before it.
