"""Run from repository root: python scripts/build_warehouse.py."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from workforce_intelligence.warehouse import load_warehouse, department_kpis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "HR-Employee-Attrition.csv"))
    parser.add_argument("--database", default=str(ROOT / "data" / "workforce.db"))
    args = parser.parse_args()
    count = load_warehouse(args.source, args.database)
    print(f"Loaded {count} records into {args.database}")
    print(department_kpis(args.database).to_string(index=False))

if __name__ == "__main__":
    main()
