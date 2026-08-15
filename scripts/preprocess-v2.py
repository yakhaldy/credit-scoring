import os
import re
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

# Wide, repeatedly-assigned dataframes (350+ columns) trigger pandas'
# block-consolidation heuristic; it's a perf note, not a correctness issue.
warnings.filterwarnings("ignore", category=PerformanceWarning)

DATA_DIR = "data"
OUTPUT_DIR = "data/processed"


def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if not pd.api.types.is_numeric_dtype(col_type):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(col_type)[:3] == "int":
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df


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
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    new_cols = {
        "AGE_YEARS": -df["DAYS_BIRTH"] / 365,
        "CREDIT_INCOME_RATIO": df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"],
        "ANNUITY_INCOME_RATIO": df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"],
        "EXT_SOURCE_MEAN": df[ext_cols].mean(axis=1),
    }
    df = df.assign(**new_cols)

    return df


def aggregate_bureau(bureau):
    num_agg = {
        "AMT_CREDIT_SUM": ["mean"],
        "AMT_CREDIT_SUM_DEBT": ["mean"],
        "CREDIT_DAY_OVERDUE": ["mean"],
    }
    agg = bureau.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["BUREAU_" + "_".join(c).upper() for c in agg.columns]
    agg["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_previous_application(prev):
    num_agg = {
        "AMT_CREDIT": ["mean"],
        "AMT_ANNUITY": ["mean"],
        "DAYS_DECISION": ["mean"],
    }
    agg = prev.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["PREV_" + "_".join(c).upper() for c in agg.columns]
    agg["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_pos_cash(pos):
    num_agg = {
        "SK_DPD": ["mean"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
    }
    agg = pos.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["POS_" + "_".join(c).upper() for c in agg.columns]
    agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_credit_card(cc):
    num_agg = {
        "AMT_BALANCE": ["mean"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
    }
    agg = cc.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["CC_" + "_".join(c).upper() for c in agg.columns]
    agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def aggregate_installments(inst):
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]

    num_agg = {
        "AMT_INSTALMENT": ["mean"],
        "DAYS_LATE": ["mean"],
    }
    agg = inst.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["INSTAL_" + "_".join(c).upper() for c in agg.columns]
    agg["INSTAL_COUNT"] = inst.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def build_features():
    print("Loading raw CSV files...")
    train, test, bureau, _, prev, pos, cc, inst = load_raw_data()
    print(f"  application_train: {train.shape}, application_test: {test.shape}")

    train = train.assign(IS_TRAIN=1)
    test = test.assign(IS_TRAIN=0, TARGET=np.nan)
    df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)

    print("Engineering application-level features (simplified set)...")
    df = engineer_application_features(df)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    
    df.columns = [re.sub(r"\W+", "_", c) for c in df.columns]

    print("Aggregating bureau...")
    bureau_agg = aggregate_bureau(bureau)
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
    df = reduce_mem_usage(df)

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
