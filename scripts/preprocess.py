"""Feature engineering pipeline for the Home Credit Default Risk dataset.

Loads the main application table plus the 5 supplementary tables (bureau,
bureau_balance, previous_application, POS_CASH_balance, credit_card_balance,
installments_payments), aggregates each supplementary table per SK_ID_CURR,
merges everything, and saves the resulting train/test feature matrices to
data/processed/ for train.py and predict.py to consume.
"""

import os
import re

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_DIR = "data/processed"


def inspect_dataframe(df, name):
    print(f"\n{'='*40}")
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print("\nInfo:")
    df.info()
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nSample (5 rows):\n{df.head(5)}")


def load_raw_data():
    train = pd.read_csv(f"{DATA_DIR}/application_train.csv")
    test = pd.read_csv(f"{DATA_DIR}/application_test.csv")
    bureau = pd.read_csv(f"{DATA_DIR}/bureau.csv")
    bureau_balance = pd.read_csv(f"{DATA_DIR}/bureau_balance.csv")
    prev = pd.read_csv(f"{DATA_DIR}/previous_application.csv")
    pos = pd.read_csv(f"{DATA_DIR}/POS_CASH_balance.csv")
    cc = pd.read_csv(f"{DATA_DIR}/credit_card_balance.csv")
    inst = pd.read_csv(f"{DATA_DIR}/installments_payments.csv")
    return train, test, bureau, bureau_balance, prev, pos, cc, inst


def engineer_application_features(df):
    """Ratios and flags derived from the main application table."""
    # DAYS_EMPLOYED == 365243 is a known placeholder for "not employed" (pensioners, etc.)
    days_employed_anom = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    days_employed_clean = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]

    new_cols = {
        "DAYS_EMPLOYED": days_employed_clean,
        "DAYS_EMPLOYED_ANOM": days_employed_anom,
        "AGE_YEARS": -df["DAYS_BIRTH"] / 365,
        "YEARS_EMPLOYED": -days_employed_clean / 365,
        "CREDIT_INCOME_RATIO": df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"],
        "ANNUITY_INCOME_RATIO": df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"],
        "CREDIT_GOODS_RATIO": df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"],
        "CREDIT_ANNUITY_RATIO": df["AMT_CREDIT"] / df["AMT_ANNUITY"],
        "INCOME_PER_PERSON": df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"],
        "EMPLOYED_BIRTH_RATIO": days_employed_clean / df["DAYS_BIRTH"],
        "EXT_SOURCE_MEAN": df[ext_cols].mean(axis=1),
        "EXT_SOURCE_MIN": df[ext_cols].min(axis=1),
        "EXT_SOURCE_MAX": df[ext_cols].max(axis=1),
        "EXT_SOURCE_STD": df[ext_cols].std(axis=1),
        "EXT_SOURCE_PROD": df[ext_cols].prod(axis=1),
        "DOC_COUNT": df[doc_cols].sum(axis=1),
    }
    df = df.drop(columns=["DAYS_EMPLOYED"]).assign(**new_cols)

    return df


def aggregate_bureau(bureau, bureau_balance):
    """One row per SK_ID_CURR summarizing credit-bureau history."""
    bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BB_DPD_COUNT=("STATUS", lambda s: s.isin(["1", "2", "3", "4", "5"]).sum()),
    )
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau["CREDIT_ACTIVE_FLAG"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.int8)

    num_agg = {
        "DAYS_CREDIT": ["mean", "min", "max"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "DAYS_CREDIT_ENDDATE": ["mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "AMT_CREDIT_SUM": ["mean", "sum", "max"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "CREDIT_ACTIVE_FLAG": ["mean", "sum"],
        "BB_MONTHS_COUNT": ["mean", "sum"],
        "BB_DPD_COUNT": ["mean", "sum"],
    }
    agg = bureau.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["BUREAU_" + "_".join(c).upper() for c in agg.columns]
    agg["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_previous_application(prev):
    """One row per SK_ID_CURR summarizing past Home Credit applications."""
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    prev["APPROVED_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    prev["REFUSED_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)

    num_agg = {
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"],
        "APP_CREDIT_RATIO": ["mean"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "DAYS_DECISION": ["mean", "min"],
        "CNT_PAYMENT": ["mean"],
        "APPROVED_FLAG": ["mean", "sum"],
        "REFUSED_FLAG": ["mean", "sum"],
    }
    agg = prev.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["PREV_" + "_".join(c).upper() for c in agg.columns]
    agg["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_pos_cash(pos):
    """One row per SK_ID_CURR summarizing POS/cash loan monthly snapshots."""
    pos["LATE_PAYMENT"] = (pos["SK_DPD"] > 0).astype(np.int8)

    num_agg = {
        "MONTHS_BALANCE": ["mean", "min"],
        "SK_DPD": ["mean", "max"],
        "SK_DPD_DEF": ["mean", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
        "LATE_PAYMENT": ["mean", "sum"],
    }
    agg = pos.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["POS_" + "_".join(c).upper() for c in agg.columns]
    agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_credit_card(cc):
    """One row per SK_ID_CURR summarizing credit-card monthly balances."""
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]

    num_agg = {
        "AMT_BALANCE": ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "UTILIZATION": ["mean", "max"],
        "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
        "SK_DPD": ["mean", "max"],
        "CNT_DRAWINGS_CURRENT": ["mean"],
    }
    agg = cc.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["CC_" + "_".join(c).upper() for c in agg.columns]
    agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_installments(inst):
    """One row per SK_ID_CURR summarizing installment repayment behavior."""
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["LATE_FLAG"] = (inst["DAYS_LATE"] > 0).astype(np.int8)

    num_agg = {
        "PAYMENT_DIFF": ["mean", "sum"],
        "PAYMENT_RATIO": ["mean"],
        "DAYS_LATE": ["mean", "max"],
        "LATE_FLAG": ["mean", "sum"],
        "AMT_INSTALMENT": ["mean", "sum"],
        "AMT_PAYMENT": ["mean", "sum"],
    }
    agg = inst.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["INSTAL_" + "_".join(c).upper() for c in agg.columns]
    agg["INSTAL_COUNT"] = inst.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def build_features():
    print("Loading raw CSV files...")
    train, test, bureau, bureau_balance, prev, pos, cc, inst = load_raw_data()
    print(f"  application_train: {train.shape}, application_test: {test.shape}")

    # Concatenate train+test so categorical one-hot encoding produces identical
    # columns on both sides (no train/test column mismatch to realign later).
    train = train.assign(IS_TRAIN=1)
    test = test.assign(IS_TRAIN=0, TARGET=np.nan)
    df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)

    print("Engineering application-level features...")
    df = engineer_application_features(df)

    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    # LightGBM rejects special/JSON characters in feature names (one-hot columns
    # can contain ": , / etc." from raw category values, e.g. "Industry: type 3").
    df.columns = [re.sub(r"\W+", "_", c) for c in df.columns]

    print("Aggregating bureau + bureau_balance...")
    bureau_agg = aggregate_bureau(bureau, bureau_balance)
    print("Aggregating previous_application...")
    prev_agg = aggregate_previous_application(prev)
    print("Aggregating POS_CASH_balance...")
    pos_agg = aggregate_pos_cash(pos)
    print("Aggregating credit_card_balance...")
    cc_agg = aggregate_credit_card(cc)
    print("Aggregating installments_payments...")
    inst_agg = aggregate_installments(inst)

    print("Merging all tables on SK_ID_CURR...")
    for agg in (bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg):
        df = df.merge(agg, on="SK_ID_CURR", how="left")

    df = df.replace([np.inf, -np.inf], np.nan)

    train_df = df[df["IS_TRAIN"] == 1].drop(columns=["IS_TRAIN"]).reset_index(drop=True)
    test_df = df[df["IS_TRAIN"] == 0].drop(columns=["IS_TRAIN", "TARGET"]).reset_index(drop=True)

    return train_df, test_df


def main():
    train_df, test_df = build_features()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_pickle(f"{OUTPUT_DIR}/train.pkl")
    test_df.to_pickle(f"{OUTPUT_DIR}/test.pkl")

    print(f"\nSaved {OUTPUT_DIR}/train.pkl -> shape {train_df.shape}")
    print(f"Saved {OUTPUT_DIR}/test.pkl -> shape {test_df.shape}")
    print(f"Missing TARGET in train: {train_df['TARGET'].isnull().sum()}")


if __name__ == "__main__":
    main()
