"""Reproducible, leakage-conscious research baseline (not for employment decisions)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data_quality import validate_hr_dataset

# Fixed whitelist avoids identifiers, protected attributes, derived risk scores,
# and fields that may be unavailable at inference time.
NUMERIC = ("Age", "DistanceFromHome", "JobLevel", "MonthlyIncome",
           "NumCompaniesWorked", "TotalWorkingYears", "TrainingTimesLastYear",
           "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
           "YearsWithCurrManager")
CATEGORICAL = ("BusinessTravel", "Department", "EducationField", "JobRole",
               "OverTime")
FEATURES = NUMERIC + CATEGORICAL

def _metrics(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", zero_division=0)
    return {"average_precision": float(average_precision_score(y, probabilities)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "brier_score": float(brier_score_loss(y, probabilities)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1)}

def evaluate_baseline(frame: pd.DataFrame, seed: int = 42) -> dict:
    report = validate_hr_dataset(frame)
    if not report.passed:
        raise ValueError(f"Invalid source data: {report}")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    X = frame.loc[:, FEATURES].copy()
    y = frame["Attrition"].eq("Yes").astype(int)
    if y.value_counts().min() < 10:
        raise ValueError("At least 10 examples per target class required")
    # 60/20/20 split. All preprocessing fits on training data only.
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), list(NUMERIC)),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("encode", OneHotEncoder(handle_unknown="ignore"))]),
         list(CATEGORICAL)),
    ])
    model = Pipeline([("preprocess", preprocessing),
                      ("classifier", LogisticRegression(
                          max_iter=2000, class_weight="balanced", random_state=seed))])
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    # Select threshold only on validation, never on test.
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [precision_recall_fscore_support(
        y_val, (val_proba >= t).astype(int), average="binary", zero_division=0)[2]
        for t in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])
    test_proba = model.predict_proba(X_test)[:, 1]
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    return {
        "status": "research_only_not_validated_for_employment_decisions",
        "seed": seed, "features": list(FEATURES),
        "split_sizes": {"train": len(y_train), "validation": len(y_val),
                        "test": len(y_test)},
        "threshold_selected_on_validation": threshold,
        "logistic_test": _metrics(y_test, test_proba, threshold),
        "dummy_test": _metrics(y_test, dummy_proba, 0.5),
        "note": "New class-weighted baseline; NOT a reproduction of the legacy SMOTEENN model.",
    }

def save_report(report: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
