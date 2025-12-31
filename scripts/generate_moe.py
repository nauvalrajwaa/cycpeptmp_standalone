#!/usr/bin/env python3
"""Wrapper to run MOE (moebatch) jobs to generate MOE descriptor CSVs.

This script runs MOE jobs (SVL) to produce 2D/3D descriptor CSVs for peptides and monomers.
It supports running by `--run-id` (uses runs/<run_id>/sdf/*.sdf) or by explicit SDF paths.

Defaults assume `moebatch -job <svl> -in <sdf> -out <csv>` works for your MOE installation.
Adjust job filenames with the flags if your setup differs.

Example:
  python scripts/generate_moe.py --run-id run_20251230_204756 --out-dir /tmp/moe_csvs \
    --job-peptide-2d compute_peptide_2D.svl --job-peptide-3d compute_peptide_3D.svl \
    --job-monomer-2d compute_monomer_2D.svl --job-monomer-3d compute_monomer_3D.svl

Or point directly to SDFs:
  python scripts/generate_moe.py --peptide-sdf runs/run_xxx/sdf/peptide.sdf --monomer-sdf runs/run_xxx/sdf/monomer.sdf --out-dir /tmp/moe_csvs
"""
import argparse
import os
import shutil
import subprocess
import sys
import datetime
import json

# make sure repo utils are importable
sys.path.append(os.getcwd())
from utils import utils_function
from utils import generate_conformation


def find_moebatch(provided=None):
    if provided:
        if os.path.exists(provided):
            return provided
        print(f'Provided moebatch path not found: {provided}')
        return None
    mb = shutil.which('moebatch')
    return mb


def run_job(moebatch, jobfile, input_sdf, out_csv, dry_run=False):
    cmd = [moebatch, '-job', jobfile, '-in', input_sdf, '-out', out_csv]
    print('Running:', ' '.join(cmd))
    if dry_run:
        return 0
    try:
        res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(res.stdout.decode(errors='ignore'))
        if res.returncode != 0:
            print('moebatch stderr:', res.stderr.decode(errors='ignore'), file=sys.stderr)
        return res.returncode
    except FileNotFoundError:
        print('Error: moebatch executable not found.', file=sys.stderr)
        return 127


def main():
    p = argparse.ArgumentParser(description='Generate MOE descriptor CSVs using moebatch')
    p.add_argument('--run-id', help='Run id under runs/ (uses runs/<run_id>/sdf/*.sdf)')
    p.add_argument('--peptide-sdf', help='Path to peptide.sdf')
    p.add_argument('--monomer-sdf', help='Path to monomer.sdf')
    p.add_argument('--sequences-file', help='Plain txt file with one AA sequence per line. If provided, SDFs will be generated.')
    p.add_argument('--sequence', help='Single AA sequence string to generate SDF for (alternative to --sequences-file)')
    p.add_argument('--out-dir', required=True, help='Directory to write MOE CSVs to')
    p.add_argument('--moebatch', help='Path to moebatch executable (optional)')
    p.add_argument('--job-peptide-2d', default='compute_peptide_2D.svl', help='MOE SVL job for peptide 2D')
    p.add_argument('--job-peptide-3d', default='compute_peptide_3D.svl', help='MOE SVL job for peptide 3D')
    p.add_argument('--job-monomer-2d', default='compute_monomer_2D.svl', help='MOE SVL job for monomer 2D')
    p.add_argument('--job-monomer-3d', default='compute_monomer_3D.svl', help='MOE SVL job for monomer 3D')
    p.add_argument('--dry-run', action='store_true', help='Print commands without running')
    args = p.parse_args()

    # If a run-id is provided, use its sdf folder
    peptide_sdf = args.peptide_sdf
    monomer_sdf = args.monomer_sdf
    temp_run_created = False
    if args.run_id and not (args.sequences_file or args.sequence):
        base = os.path.join('runs', args.run_id)
        peptide_sdf = os.path.join(base, 'sdf', 'peptide.sdf')
        monomer_sdf = os.path.join(base, 'sdf', 'monomer.sdf')

    # If sequences provided, generate a temporary run folder (SDFs) using the same pipeline logic
    if args.sequences_file or args.sequence:
        # create a run folder under runs/
        run_id = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S_moe')
        base = os.path.join('runs', run_id)
        dir_data = os.path.join(base, 'data')
        dir_sdf = os.path.join(base, 'sdf')
        os.makedirs(dir_data, exist_ok=True)
        os.makedirs(dir_sdf, exist_ok=True)

        # build new_data dataframe from sequences
        sequences = []
        if args.sequence:
            sequences.append(args.sequence.strip())
        if args.sequences_file:
            if not os.path.exists(args.sequences_file):
                print(f'Error: sequences file not found: {args.sequences_file}', file=sys.stderr)
                sys.exit(2)
            with open(args.sequences_file, 'r') as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        sequences.append(s)

        if not sequences:
            print('Error: no sequences provided to generate SDFs', file=sys.stderr)
            sys.exit(2)

        # map 1-letter AA to SMILES (same mapping as predict.py)
        AA_TO_SMILES = {
            'A': 'CN[C@@H](C)C=O',
            'C': 'CN[C@H](C=O)CS',
            'D': 'CN[C@H](C=O)CC(=O)O',
            'E': 'CN[C@H](C=O)CCC(=O)O',
            'F': 'CN[C@H](C=O)Cc1ccccc1',
            'G': 'CNCC=O',
            'H': 'CN[C@H](C=O)Cc1c[nH]cn1',
            'I': 'CC[C@H](C)[C@@H](C=O)NC',
            'K': 'CN[C@H](C=O)CCCCN',
            'L': 'CN[C@H](C=O)CC(C)C',
            'M': 'CN[C@H](C=O)CCSC',
            'N': 'CN[C@H](C=O)CC(N)=O',
            'P': 'CN1CCC[C@H]1C=O',
            'Q': 'CN[C@H](C=O)CCC(N)=O',
            'R': 'CN[C@H](C=O)CCCNC(=N)N',
            'S': 'CN[C@H](C=O)CO',
            'T': 'CN[C@H](C=O)[C@@H](C)OC',
            'V': 'CN[C@H](C=O)C(C)C',
            'W': 'CN[C@H](C=O)Cc1c[nH]c2ccccc12',
            'Y': 'CN[C@H](C=O)Cc1ccc(O)cc1'
        }

        data_rows = []
        for idx, seq in enumerate(sequences):
            sequ = seq.upper().strip()
            monomers = []
            valid = True
            for aa in sequ:
                if aa in AA_TO_SMILES:
                    monomers.append(AA_TO_SMILES[aa])
                else:
                    print(f'Warning: Unknown amino acid "{aa}" in sequence {sequ}; skipping sequence')
                    valid = False
                    break
            if not valid:
                continue
            row = {
                'ID': f'pept{idx+1}',
                'ID_org': f'seq_{idx+1}_{sequ}',
                'SMILES': '.'.join(monomers),
                'Monomer_number': len(monomers),
                'Monomer_number_in_main_chain': len(monomers),
                'shape': 'Unknown',
                'permeability': 0
            }
            for i, m in enumerate(monomers):
                row[f'Substructure-{i+1}'] = m
            data_rows.append(row)

        if not data_rows:
            print('No valid sequences produced SDFs.', file=sys.stderr)
            sys.exit(2)

        import pandas as pd
        new_data_df = pd.DataFrame(data_rows)
        new_data_df.to_csv(os.path.join(dir_data, 'new_data.csv'), index=False)

        # generate unique monomers and enum_smiles using utils
        utils_function.get_unique_monomer(new_data_df, os.path.join(dir_data, 'unique_monomer.csv'))
        config_dummy = {}
        # Try to read existing config if present
        cfg_path = os.path.join('config', 'CycPeptMP.json')
        if os.path.exists(cfg_path):
            try:
                config_dummy = json.load(open(cfg_path, 'r'))
            except Exception:
                config_dummy = {}
        utils_function.enumerate_smiles(new_data_df, config_dummy, os.path.join(dir_data, 'enum_smiles.csv'))

        # generate conformations
        df_enu = None
        import pandas as pd
        df_enu = pd.read_csv(os.path.join(dir_data, 'enum_smiles.csv'))
        generate_conformation.generate_peptide_conformation(config_dummy, df_enu, os.path.join(dir_sdf, 'peptide.sdf'))
        df_monomer = pd.read_csv(os.path.join(dir_data, 'unique_monomer.csv'))
        generate_conformation.generate_monomer_conformation(config_dummy, df_monomer, os.path.join(dir_sdf, 'monomer.sdf'))

        peptide_sdf = os.path.join(dir_sdf, 'peptide.sdf')
        monomer_sdf = os.path.join(dir_sdf, 'monomer.sdf')
        temp_run_created = True
    else:
        peptide_sdf = args.peptide_sdf
        monomer_sdf = args.monomer_sdf

    if not peptide_sdf or not monomer_sdf:
        print('Error: you must provide either --run-id or both --peptide-sdf and --monomer-sdf', file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(peptide_sdf):
        print(f'Error: peptide SDF not found: {peptide_sdf}', file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(monomer_sdf):
        print(f'Error: monomer SDF not found: {monomer_sdf}', file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    moebatch = find_moebatch(args.moebatch)
    if moebatch is None:
        print('Error: moebatch not found on PATH and no --moebatch provided.', file=sys.stderr)
        sys.exit(2)

    jobs = [
        (args.job_peptide_2d, peptide_sdf, os.path.join(out_dir, 'peptide_moe_2D.csv')),
        (args.job_peptide_3d, peptide_sdf, os.path.join(out_dir, 'peptide_moe_3D.csv')),
        (args.job_monomer_2d, monomer_sdf, os.path.join(out_dir, 'monomer_moe_2D.csv')),
        (args.job_monomer_3d, monomer_sdf, os.path.join(out_dir, 'monomer_moe_3D.csv')),
    ]

    failed = []
    for jobfile, inp, outcsv in jobs:
        if args.dry_run:
            print('[dry-run] would run:', jobfile, inp, '->', outcsv)
            continue

        rc = run_job(moebatch, jobfile, inp, outcsv, dry_run=args.dry_run)
        if rc != 0:
            failed.append((jobfile, rc))

    print('\nMOE generation summary:')
    for _, _, outcsv in jobs:
        status = 'OK' if os.path.exists(outcsv) else 'MISSING'
        print(f'  {os.path.basename(outcsv)}: {status}')

    if failed:
        print('\nSome MOE jobs failed:')
        for jobfile, rc in failed:
            print(f'  {jobfile}: return code {rc}')
        sys.exit(1)

    print('All MOE files generated (or present) under:', out_dir)


if __name__ == '__main__':
    main()
