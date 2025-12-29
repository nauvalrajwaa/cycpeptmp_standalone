#!/usr/bin/env python3
"""Filter sequences by pLDDT from a design metrics CSV.

Outputs a plain text file with one sequence per line (suitable for --file in predict.py).

Usage:
  python scripts/filter_plddt.py --input from_evobind/designmetrics.csv --threshold 60 --output scripts/test_filtered.txt
"""
import argparse
import os
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Filter sequences by pLDDT and write sequences to a txt file')
    parser.add_argument('--input', '-i', default='from_evobind/designmetrics.csv', help='Input CSV file (default: from_evobind/designmetrics.csv)')
    parser.add_argument('--threshold', '-t', type=float, default=60.0, help='pLDDT threshold (default: 60.0)')
    parser.add_argument('--op', choices=['gte','gt','lte','lt'], default='gte', help='pLDDT comparison operator (default: gte)')
    parser.add_argument('--loss', type=float, default=None, help='Optional loss threshold to filter by')
    parser.add_argument('--loss-op', choices=['gte','gt','lte','lt'], default='lte', help='Loss comparison operator (default: lte)')
    parser.add_argument('--output', '-o', default='scripts/test_filtered.txt', help='Output txt file (one sequence per line)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Error: input file not found: {args.input}', file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.input)

    if 'plddt' not in df.columns or 'sequence' not in df.columns:
        print('Error: input CSV must contain "plddt" and "sequence" columns', file=sys.stderr)
        sys.exit(2)

    if args.op == 'gte':
        mask = df['plddt'] >= args.threshold
    elif args.op == 'gt':
        mask = df['plddt'] > args.threshold
    elif args.op == 'lte':
        mask = df['plddt'] <= args.threshold
    else:
        mask = df['plddt'] < args.threshold

    # Optional loss-based filtering (combined with pLDDT using logical AND)
    if args.loss is not None:
        if 'loss' not in df.columns:
            print('Error: loss threshold requested but input CSV has no "loss" column', file=sys.stderr)
            sys.exit(2)

        if args.loss_op == 'gte':
            loss_mask = df['loss'] >= args.loss
        elif args.loss_op == 'gt':
            loss_mask = df['loss'] > args.loss
        elif args.loss_op == 'lte':
            loss_mask = df['loss'] <= args.loss
        else:
            loss_mask = df['loss'] < args.loss

        mask = mask & loss_mask

    selected = df.loc[mask, 'sequence'].astype(str).str.strip()

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    selected_list = [s for s in selected.tolist() if s]
    # remove duplicates while preserving order
    seen = set()
    unique_seqs = []
    for s in selected_list:
        if s not in seen:
            seen.add(s)
            unique_seqs.append(s)
    selected_list = unique_seqs

    if not selected_list:
        print('Warning: no sequences matched the pLDDT/loss thresholds; wrote empty file.')

    with open(args.output, 'w') as f:
        for seq in selected_list:
            f.write(seq + '\n')

    print(f'Wrote {len(selected_list)} unique sequences to: {args.output}')


if __name__ == '__main__':
    main()
