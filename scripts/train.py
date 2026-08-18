
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed"
MODEL_DIR = "results/model"
ID_TARGET_COLS = ["SK_ID_CURR", "TARGET"]


def load_train_data():
    df = pd.read_pickle(f"{PROCESSED_DIR}/train.pkl")
    y = df["TARGET"].astype(int)
    X = df.drop(columns=ID_TARGET_COLS)
    return X, y


def train_model(X_train, y_train, X_val, y_val):
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        importance_type="gain",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric="auc",
        eval_names=["train", "valid"],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)],
    )
    return model


def plot_learning_curve(model, path):
    results = model.evals_result_
    train_auc = results["train"]["auc"]
    valid_auc = results["valid"]["auc"]
    best_iter = model.best_iteration_

    plt.figure(figsize=(9, 6))
    plt.plot(train_auc, label="Train AUC")
    plt.plot(valid_auc, label="Validation AUC")
    plt.axvline(best_iter, color="red", linestyle="--", label=f"Early stopping (iter {best_iter})")
    plt.xlabel("Boosting iteration")
    plt.ylabel("AUC")
    plt.title("Learning curve — LightGBM (Train vs Validation AUC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, path, top_n=25):
    importances = model.feature_importances_
    order = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(9, 10))
    plt.barh(np.array(feature_names)[order], importances[order])
    plt.xlabel("Importance (gain)")
    plt.title(f"Top {top_n} Feature Importances — LightGBM")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    X, y = load_train_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

    model = train_model(X_train, y_train, X_val, y_val)

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"\nValidation AUC: {auc:.4f}")
    print(f"Best iteration: {model.best_iteration_}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_learning_curve(model, f"{MODEL_DIR}/learning_curve.png")
    plot_feature_importance(model, X.columns.tolist(), f"{MODEL_DIR}/feature_importance.png")

    with open(f"{MODEL_DIR}/my_own_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": X.columns.tolist(), "val_auc": auc}, f)

    with open(f"{MODEL_DIR}/model_report.txt", "w") as f:
        f.write(f"Validation AUC: {auc:.4f}\n")
        f.write(f"Best iteration: {model.best_iteration_}\n")
        f.write(f"Number of features: {len(X.columns)}\n")
        f.write(f"Feature names: {', '.join(X.columns.tolist())}\n")
    

    print(f"\nSaved model to {MODEL_DIR}/my_own_model.pkl")
    print(f"Saved {MODEL_DIR}/model_report.txt")
    print(f"Saved {MODEL_DIR}/learning_curve.png")
    print(f"Saved {MODEL_DIR}/feature_importance.png")


if __name__ == "__main__":
    main()
