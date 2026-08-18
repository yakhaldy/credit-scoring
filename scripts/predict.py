

import pickle

import pandas as pd
from sklearn.metrics import roc_auc_score


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


    train = pd.read_pickle(f"{PROCESSED_DIR}/train.pkl")
    X_train = train[features]
    y_train = train["TARGET"]
    train_preds = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_preds)
    print(f"AUC on validation set:: {train_auc:.4f}")


if __name__ == "__main__":
    main()
