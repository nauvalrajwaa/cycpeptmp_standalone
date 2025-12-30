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
    parser.add_argument('--sort-by', choices=['plddt','loss'], default=None, help='Optional sort for output CSV/txt: "plddt" (high->low) or "loss" (low->high)')
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

    # Get filtered dataframe
    df_filtered = df.loc[mask].copy()
    
    # Deduplicate by sequence while preserving order
    df_filtered = df_filtered.drop_duplicates(subset='sequence', keep='first')

    # Optional sorting: plddt (desc) or loss (asc)
    if args.sort_by is not None:
        if args.sort_by == 'plddt':
            if 'plddt' in df_filtered.columns:
                df_filtered = df_filtered.sort_values(by='plddt', ascending=False)
            else:
                print('Warning: sort-by plddt requested but "plddt" column not present; skipping sort.')
        elif args.sort_by == 'loss':
            if 'loss' in df_filtered.columns:
                df_filtered = df_filtered.sort_values(by='loss', ascending=True)
            else:
                print('Warning: sort-by loss requested but "loss" column not present; skipping sort.')
    
    if df_filtered.empty:
        print('Warning: no sequences matched the pLDDT/loss thresholds; wrote empty files.')
    
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write txt file (sequences only)
    sequences = df_filtered['sequence'].astype(str).str.strip().tolist()
    with open(args.output, 'w') as f:
        for seq in sequences:
            if seq:
                f.write(seq + '\n')

    # Write csv file (with metadata columns)
    csv_output = args.output.rsplit('.', 1)[0] + '.csv'
    cols_to_save = []
    for col in ['iteration', 'loss', 'plddt', 'sequence']:
        if col in df_filtered.columns:
            cols_to_save.append(col)
    
    if cols_to_save:
        df_filtered[cols_to_save].to_csv(csv_output, index=False)
        print(f'Wrote {len(df_filtered)} unique sequences to: {args.output}')
        print(f'Wrote {len(df_filtered)} rows with metadata to: {csv_output}')
    else:
        print(f'Wrote {len(df_filtered)} unique sequences to: {args.output}')


if __name__ == '__main__':
    main()
