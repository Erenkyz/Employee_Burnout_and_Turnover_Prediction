"""Run: python scripts/evaluate_baseline.py --input HR-Employee-Attrition.csv"""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workforce_intelligence.model_validation import evaluate_baseline, save_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="HR-Employee-Attrition.csv")
    parser.add_argument("--output", default="data/baseline_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = evaluate_baseline(pd.read_csv(args.input), seed=args.seed)
    save_report(report, args.output)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
