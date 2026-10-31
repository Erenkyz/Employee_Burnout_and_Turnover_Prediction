"""Repeated holdout stability and calibration diagnostics for the research baseline.

Each repeat trains on 60%, tunes its decision threshold on 20%, and measures
on the remaining 20%. Test observations overlap between repeats, so this is
a stability diagnostic, NOT an independent external validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_quality import validate_hr_dataset
from .model_validation import CATEGORICAL, FEATURES, NUMERIC, _metrics


def _expected_calibration_error(y_true, probabilities, bins=10):
    """Equal-width-bin ECE, a descriptive diagnostic sensitive to bin choice."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    return float(sum(
        np.mean(bucket == i) * abs(float(np.mean(y[bucket == i])) -
                                   float(np.mean(p[bucket == i])))
        for i in range(bins) if np.any(bucket == i)
    ))


def evaluate_stability(frame: pd.DataFrame, seeds=(42, 43, 44, 45, 46)):
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
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one unique seed")

    runs = []
    for seed in seeds:
        X_train, X_hold, y_train, y_hold = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)
        preprocess = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), list(NUMERIC)),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), list(CATEGORICAL)),
        ])
        model = Pipeline([
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed)),
        ])
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = [
            precision_recall_fscore_support(
                y_val, (val_proba >= threshold).astype(int),
                average="binary", zero_division=0)[2]
            for threshold in thresholds
        ]
        threshold = float(thresholds[int(np.argmax(f1_scores))])
        probabilities = model.predict_proba(X_test)[:, 1]
        fraction_positive, mean_predicted = calibration_curve(
            y_test, probabilities, n_bins=10, strategy="uniform")
        runs.append({
            "seed": seed,
            "split_sizes": {"train": len(y_train), "validation": len(y_val),
                            "test": len(y_test)},
            "threshold_selected_on_validation": threshold,
            "test_metrics": _metrics(y_test, probabilities, threshold),
            "test_ece_10_bins": _expected_calibration_error(y_test, probabilities),
            "calibration_curve": {
                "mean_predicted": mean_predicted.tolist(),
                "fraction_positive": fraction_positive.tolist(),
            },
        })
    keys = list(runs[0]["test_metrics"]) + ["ece_10_bins"]
    summary = {}
    for key in keys:
        values = [
            run["test_ece_10_bins"] if key == "ece_10_bins"
            else run["test_metrics"][key] for run in runs
        ]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {
        "status": "research_only_not_validated_for_employment_decisions",
        "features": list(FEATURES),
        "seeds": list(seeds),
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Overlapping test sets across seeds: repeat metrics are correlated.",
            "No hyperparameter selection on repeat test metrics is permitted.",
            "ECE is sensitive to binning; calibration curves omit empty bins.",
            "This evaluates uncalibrated probabilities; no calibration model is fitted.",
            "Synthetic IBM HR data is not evidence of real-world suitability.",
        ],
    }


def save_stability_report(report, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
