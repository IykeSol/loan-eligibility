"""Train the credit risk model and generate every figure used by the app and the report.

Run:  python train.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import load_dataset
from src.ml_model import train
from src.preprocessing import CATEGORICAL, ENGINEERED, NUMERIC, prepare
from src.rule_engine import forward_chain

FIG_DIR = os.path.join("reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = {"Y": "#2e7d32", "N": "#c62828"}


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def eda_figures(df, data):
    print("Generating exploratory figures...")

    # 1. Approval distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Loan_Status"].value_counts()
    ax.bar(["Approved", "Rejected"], [counts.get("Y", 0), counts.get("N", 0)],
           color=["#2e7d32", "#c62828"])
    for i, v in enumerate([counts.get("Y", 0), counts.get("N", 0)]):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("Loan Application Outcomes")
    ax.set_ylabel("Applications")
    save(fig, "01_approval_distribution.png")

    # 2. Credit history vs approval
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(data["Credit_History"], df["Loan_Status"], normalize="index") * 100
    ct.plot(kind="bar", ax=ax, color=["#c62828", "#2e7d32"], rot=0)
    ax.set_title("Approval Rate by Credit History")
    ax.set_xlabel("Credit History (0 = past issues, 1 = clean)")
    ax.set_ylabel("% of applications")
    ax.legend(["Rejected", "Approved"])
    save(fig, "02_credit_history_impact.png")

    # 3. Income distribution by outcome
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tmp = data.copy()
    tmp["Outcome"] = np.where(df["Loan_Status"] == "Y", "Approved", "Rejected")
    sns.kdeplot(data=tmp[tmp["TotalIncome"] < 20000], x="TotalIncome", hue="Outcome",
                fill=True, common_norm=False, ax=ax,
                palette={"Approved": "#2e7d32", "Rejected": "#c62828"})
    ax.set_title("Household Income Distribution by Outcome")
    ax.set_xlabel("Total Monthly Income")
    save(fig, "03_income_distribution.png")

    # 4. Loan amount by outcome (boxplot)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=tmp, x="Outcome", y="LoanAmount", hue="Outcome", ax=ax,
                palette={"Approved": "#2e7d32", "Rejected": "#c62828"}, legend=False)
    ax.set_title("Requested Loan Amount by Outcome")
    ax.set_ylabel("Loan Amount (thousands)")
    save(fig, "04_loan_amount_by_outcome.png")

    # 5. Property area approval rates
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(data["Property_Area"], df["Loan_Status"], normalize="index")["Y"] * 100
    ct.sort_values().plot(kind="barh", ax=ax, color="#1565c0")
    ax.set_title("Approval Rate by Property Area")
    ax.set_xlabel("% approved")
    save(fig, "05_property_area_approval.png")

    # 6. Correlation heatmap of numeric + engineered features
    fig, ax = plt.subplots(figsize=(9, 7))
    num = data[NUMERIC + ENGINEERED].copy()
    num["Approved"] = (df["Loan_Status"] == "Y").astype(int)
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlations")
    save(fig, "06_correlation_heatmap.png")

    # 7. Education & marriage
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["Education", "Married"]):
        ct = pd.crosstab(data[col], df["Loan_Status"], normalize="index")["Y"] * 100
        ct.plot(kind="bar", ax=ax, color="#6a1b9a", rot=0)
        ax.set_title(f"Approval Rate by {col}")
        ax.set_ylabel("% approved")
    save(fig, "07_education_married.png")


def evaluation_figures(name, model, scores, holdout):
    print("Generating evaluation figures...")
    X_test, y_test = holdout
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "best_model": name,
        "cv_scores": scores,
        "holdout": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        },
    }
    with open(os.path.join("reports", "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))

    # 8. Model comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    names, vals = list(scores.keys()), [v * 100 for v in scores.values()]
    bars = ax.bar(names, vals, color=["#90a4ae" if n != name else "#1565c0" for n in names])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(60, 90)
    ax.set_title("Cross-Validated Accuracy by Algorithm")
    ax.set_ylabel("Accuracy (%)")
    save(fig, "08_model_comparison.png")

    # 9. Confusion matrix
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Rejected", "Approved"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {name}")
    save(fig, "09_confusion_matrix.png")

    # 10. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1565c0", lw=2, label=f"AUC = {metrics['holdout']['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    save(fig, "10_roc_curve.png")

    # 11. Feature importance (from the fitted pipeline when available)
    clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]
    feat_names = pre.get_feature_names_out()
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    if importances is not None:
        order = np.argsort(importances)[-12:]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        labels = [feat_names[i].split("__")[-1] for i in order]
        ax.barh(labels, importances[order], color="#00838f")
        ax.set_title(f"What Drives the Decision — {name}")
        ax.set_xlabel("Relative importance")
        save(fig, "11_feature_importance.png")

    return metrics


def rule_engine_figure(df, data):
    print("Generating rule-engine figure...")
    # Run every historical application through the forward-chaining engine
    records = data.to_dict("records")
    risk_scores = [forward_chain(r).risk_score for r in records]
    tmp = pd.DataFrame({
        "risk": risk_scores,
        "Outcome": np.where(df["Loan_Status"] == "Y", "Approved", "Rejected"),
    })
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=tmp, x="risk", hue="Outcome", bins=25, multiple="stack",
                 palette={"Approved": "#2e7d32", "Rejected": "#c62828"}, ax=ax)
    ax.axvline(35, ls="--", color="#2e7d32")
    ax.axvline(60, ls="--", color="#c62828")
    ax.set_title("Expert-System Risk Scores vs Real Outcomes")
    ax.set_xlabel("Rule-based risk score (0 = safest)")
    save(fig, "12_rule_engine_validation.png")

    agree = ((tmp["risk"] < 60) == (tmp["Outcome"] == "Approved")).mean()
    print(f"  rule engine agreement with historical outcomes: {agree:.1%}")


def main():
    print("Loading dataset...")
    df = load_dataset()
    data = prepare(df)
    print(f"  {len(df)} applications, {df['Loan_Status'].eq('Y').mean():.0%} approved")

    eda_figures(df, data)

    print("Training models...")
    name, model, scores, holdout = train(df)
    print(f"  best model: {name}")

    evaluation_figures(name, model, scores, holdout)
    rule_engine_figure(df, data)
    print("Done. Model saved to models/loan_model.joblib, figures in reports/figures/")


if __name__ == "__main__":
    main()
