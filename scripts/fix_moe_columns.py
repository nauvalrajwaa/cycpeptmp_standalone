#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

EXPECTED_FILES = [
    'peptide_moe_2D.csv',
    'peptide_moe_3D.csv',
    'monomer_moe_2D.csv',
    'monomer_moe_3D.csv'
]

SMILES_VARIANTS = ['SMILES', 'Smiles', 'smiles', 'SMILE', 'Smi', 'SMI', 'mol', 'mol_charged', 'Mol', '$File']


def find_smiles_column(df):
    for c in SMILES_VARIANTS:
        if c in df.columns:
            return c
    return None


def load_unique_monomer(data_dir):
    candidates = [
        os.path.join(data_dir, 'unique_monomer.csv'),
        os.path.join(data_dir, 'monomer_table.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None


def load_peptide_db(data_dir):
    p = os.path.join(data_dir, 'CycPeptMPDB_Peptide_All.csv')
    if os.path.exists(p):
        try:
            return pd.read_csv(p, low_memory=False)
        except Exception:
            pass
    return None


def fix_file(path, unique_df, peptide_df=None):
    df = pd.read_csv(path)

    # Detect and normalize SMILES-like column
    col = find_smiles_column(df)
    if col is not None:
        if col != 'SMILES':
            df = df.rename(columns={col: 'SMILES'})
            print(f"RENAMED: {os.path.basename(path)} column '{col}' -> 'SMILES'")
        else:
            print(f"OK: {os.path.basename(path)} has SMILES column '{col}'")
    else:
        print(f"No SMILES-like column in {os.path.basename(path)}; will try to populate from DBs if available")

    # Accept and normalize ID-like columns
    id_candidates = ['ID', 'ID_org', 'MID', 'id']
    for c in id_candidates:
        if c in df.columns and c != 'ID':
            df = df.rename(columns={c: 'ID'})
            print(f"RENAMED: {os.path.basename(path)} column '{c}' -> 'ID'")
            break

    # If we have SMILES but no ID, try to map SMILES -> ID using unique_monomer
    if 'SMILES' in df.columns and 'ID' not in df.columns and unique_df is not None:
        # find SMILES column in unique_df
        u_sm = None
        for c in unique_df.columns:
            if c.lower() == 'smiles':
                u_sm = c
                break
        if u_sm is not None and 'ID' in unique_df.columns:
            mapping = dict(zip(unique_df[u_sm].astype(str), unique_df['ID'].astype(str)))
            df['ID'] = df['SMILES'].map(mapping)
            matched = df['ID'].notna().sum()
            if matched:
                print(f"Mapped {matched} rows in {os.path.basename(path)} to IDs via unique_monomer.csv")

    # For peptide files, try peptide DB mapping if still missing IDs
    if 'SMILES' in df.columns and 'ID' not in df.columns and peptide_df is not None:
        # peptide DB uses 'SMILES' and 'CycPeptMPDB_ID' as identifier
        p_sm = None
        id_col = None
        for c in peptide_df.columns:
            if c.lower() == 'smiles':
                p_sm = c
            if c.lower() in ('cycpeptmpdb_id', 'id', 'peptide_id'):
                id_col = c
        if p_sm is not None and id_col is not None:
            mapping = dict(zip(peptide_df[p_sm].astype(str), peptide_df[id_col].astype(str)))
            # only fill where ID is missing
            df['ID'] = df.get('ID')
            missing_mask = df['ID'].isna()
            df.loc[missing_mask, 'ID'] = df.loc[missing_mask, 'SMILES'].map(mapping)
            matched2 = df['ID'].notna().sum()
            if matched2:
                print(f"Mapped {matched2} rows in {os.path.basename(path)} to peptide IDs via peptide DB")

    # If still no ID, attempt positional assignment when unique_df length matches
    if 'ID' not in df.columns or df['ID'].isna().all():
        if unique_df is not None and 'ID' in unique_df.columns and len(df) == len(unique_df):
            df['ID'] = unique_df['ID'].astype(str).to_list()
            print(f"ASSIGNED: IDs from unique_monomer.csv to {os.path.basename(path)} by position")
        else:
            # last resort: create sequential external IDs
            df['ID'] = df.get('ID')
            # fill any remaining NA with sequential ext IDs
            na_idx = df['ID'].isna()
            if na_idx.any():
                start = 1
                for i in df[na_idx].index:
                    df.at[i, 'ID'] = f'EXT{start}'
                    start += 1
                print(f"WARNING: created sequential external IDs for {os.path.basename(path)} (EXT#)")

    # Ensure file saved with any applied modifications
    df.to_csv(path, index=False)

    # success if SMILES exists now
    return 'SMILES' in df.columns


def main():
    ap = argparse.ArgumentParser(description='Validate and fix MOE CSVs in a run desc folder')
    ap.add_argument('run_desc', help='path to run desc folder (e.g. runs/run_xxx/desc)')
    ap.add_argument('--data', default='data', help='project data directory containing unique_monomer.csv')
    args = ap.parse_args()

    run_desc = args.run_desc
    if not os.path.isdir(run_desc):
        print('Run desc folder not found:', run_desc)
        sys.exit(2)

    unique_df = load_unique_monomer(args.data)
    peptide_df = load_peptide_db(args.data)
    if unique_df is not None:
        print('Loaded', os.path.join(args.data, 'unique_monomer.csv'))
    else:
        print('No unique_monomer.csv found under', args.data)
    if peptide_df is not None:
        print('Loaded peptide DB', os.path.join(args.data, 'CycPeptMPDB_Peptide_All.csv'))

    results = {}
    for fn in EXPECTED_FILES:
        p = os.path.join(run_desc, fn)
        if os.path.exists(p):
            results[fn] = fix_file(p, unique_df, peptide_df)
        else:
            print('Missing expected file:', fn)
            results[fn] = False

    print('\nSummary:')
    for k, v in results.items():
        print(f' - {k}:', 'OK' if v else 'FIX/LATER')

    missing = [k for k, v in results.items() if not v]
    if missing:
        print('\nSome files need manual attention or different column names. Paste a header if you want me to adapt further.')
        sys.exit(1)
    else:
        print('\nAll expected MOE CSVs look OK or were fixed.')
        sys.exit(0)


if __name__ == '__main__':
    main()
