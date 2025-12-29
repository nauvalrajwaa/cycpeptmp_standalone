#!/usr/bin/env python3
"""Post-process prediction CSVs: normalize, classify (0/1), rank, and save sorted output.

Usage:
  python scripts/process_predictions.py path/to/new_prediction.csv [--threshold -6]
If no threshold provided, -6 is used (same as model's cutoff).
Normalization uses config `data.lower_limit` and `data.upper_limit` (maps lower->0, upper->1).
"""
import sys
import os
import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_csv', help='Path to prediction CSV')
    parser.add_argument('--threshold', type=float, default=-6.0, help='Cutoff for high permeability (>= threshold => 1)')
    parser.add_argument('--out', help='Output path (optional)')
    args = parser.parse_args()

    if not os.path.exists(args.pred_csv):
        print(f"Error: file not found: {args.pred_csv}")
        sys.exit(2)

    # Load config limits if available
    cfg_path = 'config/CycPeptMP.json'
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        lower = cfg.get('data', {}).get('lower_limit', -8.0)
        upper = cfg.get('data', {}).get('upper_limit', -4.0)
    else:
        lower, upper = -8.0, -4.0

    df = pd.read_csv(args.pred_csv)
    if 'pred' not in df.columns:
        print('Error: input csv has no "pred" column')
        sys.exit(2)

    # Add normalized score mapping [lower,upper] -> [0,1]
    denom = float(upper) - float(lower)
    if denom == 0:
        df['normalized'] = 0.0
    else:
        df['normalized'] = (df['pred'] - lower) / denom
        df['normalized'] = df['normalized'].clip(0.0, 1.0)

    # Binary class: 1 if pred >= threshold else 0
    df['class'] = (df['pred'] >= args.threshold).astype(int)
    df['class_label'] = df['class'].map({1: 'HIGH', 0: 'LOW'})

    # Rank by normalized score (1 is best)
    df = df.sort_values(by='normalized', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    # Output
    out_path = args.out if args.out else os.path.splitext(args.pred_csv)[0] + '_sorted.csv'
    df.to_csv(out_path, index=False)
    print(f'Wrote sorted results to: {out_path}')


if __name__ == '__main__':
    main()
