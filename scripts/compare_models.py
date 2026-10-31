"""Run a controlled research comparison of two logistic pipelines."""
import argparse
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workforce_intelligence.model_comparison import compare_models, save_comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="HR-Employee-Attrition.csv")
    parser.add_argument("--output", default="data/model_comparison_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = compare_models(pd.read_csv(args.input), seed=args.seed)
    save_comparison(report, args.output)
    print(f"Saved comparison to {Path(args.output)}")
    for name, result in report["models"].items():
        metrics = result["test_metrics"]
        print(f"{name}: F1={metrics['f1']:.3f}, PR-AUC={metrics['average_precision']:.3f}, "
              f"ROC-AUC={metrics['roc_auc']:.3f}, Brier={metrics['brier_score']:.3f}")

if __name__ == "__main__":
    main()
