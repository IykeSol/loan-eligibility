"""Machine-learning model: training, persistence and prediction."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import CATEGORICAL, ENGINEERED, FEATURES, NUMERIC, TARGET, prepare

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.joblib")


def build_candidates():
    """Candidate pipelines that share the same preprocessing."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC + ENGINEERED),
        ]
    )
    return {
        "Logistic Regression": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))]),
        "Random Forest": Pipeline(
            [("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42))]
        ),
        "Gradient Boosting": Pipeline(
            [("pre", pre), ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))]
        ),
    }


def train(df: pd.DataFrame):
    """Train all candidates, keep the best by cross-validated accuracy.

    Returns (best_name, best_pipeline, scores, holdout) where holdout is
    (X_test, y_test) for later evaluation plots.
    """
    data = prepare(df)
    X = data[FEATURES]
    y = (data[TARGET] == "Y").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scores = {}
    best_name, best_score, best_model = None, -np.inf, None
    for name, pipe in build_candidates().items():
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        scores[name] = float(cv.mean())
        if scores[name] > best_score:
            best_name, best_score, best_model = name, scores[name], pipe

    best_model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"name": best_name, "model": best_model, "cv_scores": scores}, MODEL_PATH)
    return best_name, best_model, scores, (X_test, y_test)


def load_model():
    """Load the persisted model bundle, or None when not trained yet."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_probability(bundle, applicant_frame: pd.DataFrame) -> float:
    """Probability of approval for a prepared single-applicant frame."""
    proba = bundle["model"].predict_proba(applicant_frame[FEATURES])[0, 1]
    return float(proba)
