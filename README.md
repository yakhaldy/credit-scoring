# Credit Scoring — Home Credit Default Risk

Username: see `username.txt` (`yakhaldy MA_03_2026`, unmodified since day 1)

## Project

An interpretable probability-of-default model for Home Credit's loan
applicants. Two audiences are served by design:
- **Compliance / regulators** — a global feature-importance ranking of
  what the model actually relies on (`results/model/feature_importance.png`).
- **Customer service** — a per-client SHAP breakdown explaining *why* a
  given applicant received their score (`results/clients_outputs/`).

Dataset: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
(`application_{train,test}.csv` + 5 supplementary tables — see
`readme_data.md`).

## Results

- **Validation AUC: 0.7915** (stratified 80/20 held-out split of
  `application_train.csv`), well above the 0.55 minimum and 0.62 target.
- Model: LightGBM, 349 engineered features, early-stopped at iteration
  1611/2000. Full methodology, overfitting-prevention measures, and
  limitations: `results/model/model_report.txt`.
- Kaggle submission file: `results/prediction.csv` (upload this to the
  competition to read the public-leaderboard AUC on the unlabelled test
  set).

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data: unzip `data/home-credit-default-risk.zip` into `data/` so that
`data/application_train.csv`, `data/bureau.csv`, etc. exist at the paths
read by the scripts below [DataSets](https://assets.01-edu.org/ai-branch/project5/home-credit-default-risk.zip) .

1. **Feature engineering** — reads the 7 raw CSVs, aggregates the 5
   supplementary tables per client, engineers ratios/flags, one-hot
   encodes categoricals, and writes `data/processed/{train,test}.pkl`:
   ```bash
   python scripts/preprocess.py
   ```

2. **Training** — stratified 80/20 split, LightGBM with early stopping,
   saves the model and the two required charts:
   ```bash
   python scripts/train.py
   # -> results/model/my_own_model.pkl
   # -> results/model/learning_curve.png
   # -> results/model/feature_importance.png
   ```

3. **Prediction** — scores `application_test.csv` and writes the Kaggle
   submission file:
   ```bash
   python scripts/predict.py
   # -> results/prediction.csv
   ```

4. **Local interpretability** — SHAP explanation + client profile +
   comparison-to-population, for any client id:
   ```bash
   python scripts/explain.py --client_id 100002 --dataset train --output results/clients_outputs/client1_correct_train.pdf
   ```
   Three pre-generated examples are already in `results/clients_outputs/`:
   - `client1_correct_train.pdf` — SK_ID_CURR 100002 (train, model correct)
   - `client2_wrong_train.pdf` — SK_ID_CURR 435011 (train, model wrong —
     see `model_report.txt` §3 for the analysis of why)
   - `client_test.pdf` — SK_ID_CURR 100067 (test set, borderline score)

5. **Exploratory Data Analysis**: `results/feature_engineering/EDA.ipynb`
   (open with `jupyter notebook`).

6. **Dashboard (bonus)** — same score/profile/SHAP/comparison as `explain.py`,
   interactive in a browser, for any client id:
   ```bash
   python results/dashboard/dashboard.py
   # -> open http://127.0.0.1:8050
   ```

## Project structure

```
data/                          raw CSVs (gitignored) + data/processed/ (generated, gitignored)
results/
  prediction.csv                Kaggle submission
  model/                        trained model, learning curve, feature importance, model_report.txt
  feature_engineering/EDA.ipynb exploratory data analysis
  clients_outputs/              per-client SHAP + Plotly PDF reports
  dashboard/dashboard.py        Dash app (bonus) — interactive version of explain.py
scripts/
  preprocess.py                 feature engineering pipeline
  train.py                      training, learning curve, global feature importance
  predict.py                    scoring of application_test.csv
  explain.py                    local interpretability (SHAP + Plotly) for one client
```
