"""Score application_test.csv with the trained model and write the Kaggle submission file.

Loads the model trained by train.py (results/model/my_own_model.pkl) and the
engineered test features (data/processed/test.pkl, built by preprocess.py),
predicts the probability of default for every SK_ID_CURR, and writes
results/prediction.csv in the format expected by the Home Credit Default Risk
Kaggle competition (SK_ID_CURR, TARGET).

Note: application_test.csv has no TARGET column, so no AUC can be computed on
it locally. The AUC printed below is the held-out validation AUC computed by
train.py on a stratified split of application_train.csv — that is the number
reported and defended at the audit. To get the real test-set AUC, submit
results/prediction.csv to the Kaggle competition and read the public
leaderboard score.
"""

import pickle

import pandas as pd

PROCESSED_DIR = "data/processed"
MODEL_PATH = "results/model/my_own_model.pkl"
OUTPUT_PATH = "results/prediction.csv"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def main():
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]
    val_auc = bundle["val_auc"]

    test = pd.read_pickle(f"{PROCESSED_DIR}/test.pkl")
    X_test = test[features]

    preds = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"], "TARGET": preds})
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Held-out validation AUC (application_train.csv split, from train.py): {val_auc:.4f}")
    print(f"Predictions saved to {OUTPUT_PATH} -> shape {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
