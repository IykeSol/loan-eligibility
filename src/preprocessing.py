"""Cleaning and feature engineering for the loan dataset."""

import numpy as np
import pandas as pd

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
ENGINEERED = ["TotalIncome", "EMI", "IncomeToEMI", "LoanToIncome"]

FEATURES = CATEGORICAL + NUMERIC + ENGINEERED
TARGET = "Loan_Status"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and normalise categories."""
    df = df.copy()
    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    if "LoanAmount" in df.columns:
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    if "Loan_Amount_Term" in df.columns:
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0).astype(int).astype(str)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived affordability features."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = (df["LoanAmount"] * 1000) / term  # LoanAmount is in thousands
    df["EMI"] = df["EMI"].fillna(df["EMI"].median())
    df["IncomeToEMI"] = df["TotalIncome"] / df["EMI"].replace(0, np.nan)
    df["IncomeToEMI"] = df["IncomeToEMI"].fillna(df["IncomeToEMI"].median())
    df["LoanToIncome"] = (df["LoanAmount"] * 1000) / df["TotalIncome"].replace(0, np.nan)
    df["LoanToIncome"] = df["LoanToIncome"].fillna(df["LoanToIncome"].median())
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Full preparation pipeline: clean then engineer."""
    return engineer(clean(df))


def applicant_to_frame(applicant: dict) -> pd.DataFrame:
    """Convert a single applicant dict (form input) into a prepared one-row DataFrame."""
    row = pd.DataFrame([applicant])
    return prepare(row)
