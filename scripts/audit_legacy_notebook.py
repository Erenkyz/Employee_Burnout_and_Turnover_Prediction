"""Static notebook audit: identify split/resampling order for manual review.

This does not execute notebook cells or establish absence of leakage.
Usage: python scripts/audit_legacy_notebook.py
"""
import argparse
import json
from pathlib import Path
import re

PATTERNS = {
    "split": r"train_test_split|StratifiedKFold|cross_val_score|cross_validate",
    "resampling": r"SMOTEENN|SMOTE|fit_resample|RandomOverSampler",
    "preprocessing_fit": r"fit_transform|\.fit\(",
    "threshold": r"threshold|precision_recall_curve|roc_curve",
    "evaluation": r"classification_report|confusion_matrix|roc_auc_score|average_precision_score",
    "target_derived": r"burnout_risk|employee_risk|Attrition",
}
def audit_notebook(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    findings = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        matched = [name for name, pattern in PATTERNS.items()
                   if re.search(pattern, source, flags=re.IGNORECASE)]
        if matched:
            findings.append({"cell": index, "signals": matched,
                             "source_excerpt": source[:1200]})
    return {"notebook": str(path), "cells_with_signals": findings,
            "warning": "Keyword audit only. Requires manual data-flow review; no automatic leakage verdict."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Employee_Burnout_and_Turnover_Prediction.ipynb")
    parser.add_argument("--output", default="data/legacy_notebook_audit.json")
    args = parser.parse_args()
    report = audit_notebook(args.input)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in report["cells_with_signals"]:
        print(f"Cell {item['cell']}: {', '.join(item['signals'])}")
    print(f"Audit saved to {destination}; manual review required.")

if __name__ == "__main__":
    main()
