import pandas as pd


def inspect_dataframe(df, name):
    print(f"\n{'='*40}")
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nInfo:")
    df.info()
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nSample (5 rows):\n{df.head(5)}")

    





test




def main():
    application_train = pd.read_csv("data/application_train.csv")
    application_test = pd.read_csv("data/application_test.csv")

    bureau = pd.read_csv("data/bureau.csv")
    bureau_balance = pd.read_csv("data/bureau_balance.csv")

    previous_application = pd.read_csv("data/previous_application.csv")

    pos_cash_balance = pd.read_csv("data/POS_CASH_balance.csv")
    credit_card_balance = pd.read_csv("data/credit_card_balance.csv")

    installments_payments = pd.read_csv("data/installments_payments.csv")

    columns_description = pd.read_csv("data/HomeCredit_columns_description.csv", encoding="latin1")

    datasets = {
        "application_train": application_train,
        "application_test": application_test,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": previous_application,
        "pos_cash_balance": pos_cash_balance,
        "credit_card_balance": credit_card_balance,
        "installments_payments": installments_payments,
        "columns_description": columns_description,
    }

    for name, df in datasets.items():
        inspect_dataframe(df, name)






if __name__ == "__main__":
    main()