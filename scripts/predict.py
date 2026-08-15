

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
