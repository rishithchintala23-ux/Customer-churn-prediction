import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df
