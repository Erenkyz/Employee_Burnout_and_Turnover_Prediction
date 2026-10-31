"""Run repeated holdout stability and calibration diagnostics."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workforce_intelligence.model_stability import evaluate_stability, save_stability_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="HR-Employee-Attrition.csv")
    parser.add_argument("--output", default="data/model_stability_metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    report = evaluate_stability(pd.read_csv(args.input), seeds=args.seeds)
    save_stability_report(report, args.output)
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved full report to {args.output}")


if __name__ == "__main__":
    main()
