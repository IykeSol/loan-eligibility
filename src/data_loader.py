"""Fetches the loan dataset from an online source, with a local cache."""

import os

import pandas as pd

DATA_URLS = [
    "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv",
    "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv",
]

LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "loan_train.csv")


def load_dataset() -> pd.DataFrame:
    """Load the loan dataset, preferring the local cache and falling back to online mirrors."""
    if os.path.exists(LOCAL_PATH):
        return pd.read_csv(LOCAL_PATH)

    last_err = None
    for url in DATA_URLS:
        try:
            df = pd.read_csv(url)
            os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
            df.to_csv(LOCAL_PATH, index=False)
            return df
        except Exception as err:  # noqa: BLE001 - try the next mirror
            last_err = err
    raise RuntimeError(f"Could not fetch the loan dataset from any source: {last_err}")
