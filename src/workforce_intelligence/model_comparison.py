"""Controlled research comparison on identical data splits and feature set.

This is not an exact reproduction of the legacy notebook and must not be used
to make decisions about individual employees.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_quality import validate_hr_dataset
from .model_validation import CATEGORICAL, FEATURES, NUMERIC, _metrics


def compare_models(frame: pd.DataFrame, seed: int = 42) -> dict:
    quality = validate_hr_dataset(frame)
    if not quality.passed:
        raise ValueError(f"Invalid source data: {quality}")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = frame.loc[:, FEATURES].copy()
    y = frame["Attrition"].eq("Yes").astype(int)
    if y.value_counts().min() < 10:
        raise ValueError("At least 10 examples per target class required")

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed
    )

    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(NUMERIC)),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(CATEGORICAL)),
    ], sparse_threshold=0)

    candidates = {
        "class_weighted_logistic": ImbPipeline([
            ("preprocess", clone(preprocessing)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "smoteenn_logistic": ImbPipeline([
            ("preprocess", clone(preprocessing)),
            ("smoteenn", SMOTEENN(sampling_strategy=0.6, random_state=seed)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
    }
    thresholds = np.linspace(0.1, 0.9, 81)
    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        validation_f1 = [
            precision_recall_fscore_support(
                y_val, (val_prob >= threshold).astype(int),
                average="binary", zero_division=0
            )[2] for threshold in thresholds
        ]
        selected = float(thresholds[int(np.argmax(validation_f1))])
        test_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "threshold_selected_on_validation": selected,
            "test_metrics": _metrics(y_test, test_prob, selected),
        }

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    results["dummy_prior"] = {
        "threshold_selected_on_validation": None,
        "test_metrics": _metrics(y_test, dummy.predict_proba(X_test)[:, 1], 0.5),
    }
    return {
        "status": "research_only_not_validated_for_employment_decisions",
        "seed": seed,
        "features": list(FEATURES),
        "split_sizes": {"train": len(y_train), "validation": len(y_val),
                        "test": len(y_test)},
        "models": results,
        "limitations": [
            "One-hot categorical columns are interpolated by standard SMOTE; synthetic "
            "rows may not correspond to valid categorical records.",
            "SMOTEENN is applied only inside the training pipeline.",
            "Thresholds selected on validation; test evaluated once per candidate.",
            "Comparing candidates on the same test set is exploratory, not an unbiased "
            "estimate of the performance of a subsequently selected winner.",
            "Not a reproduction of legacy notebook: features and split differ.",
        ],
    }


def save_comparison(report: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
