"""Split a CSV into 80% train / 20% test using scikit-learn.

Usage:
    python split_csv.py --file_path data/input.csv [--output data] [--seed 42]
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="80/20 split of a CSV file.")
    parser.add_argument("--file_path", required=True, help="Path to the input CSV")
    parser.add_argument("--output", "--output_dir", dest="output_dir", default=None,
                        help="Where to write the splits (default: alongside the input)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    path = Path(args.file_path)
    df = pd.read_csv(path)

    train, test = train_test_split(df, test_size=0.2, random_state=args.seed)

    out_dir = Path(args.output_dir) if args.output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{path.stem}_train.csv"
    test_path = out_dir / f"{path.stem}_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"{len(df)} rows -> {len(train)} train ({train_path}), {len(test)} test ({test_path})")


if __name__ == "__main__":
    main()
