#!/usr/bin/env python3
"""Run MOE workflow via the provided bash scripts (run_moe_2d.sh, run_moe_3d.sh).

This script will:
 - create a run folder (or use --run-id) and generate peptide/monomer SDFs from sequences
 - call the two user-provided bash scripts to produce MDB outputs
 - leave MDBs in the specified output folder for the user to convert to CSV manually

Important: The bash scripts are not modified by this script. We call them and set
`RUN_DIR` and `OUT_DIR` via the environment. The bash scripts should respect these
environment variables (they already do if they use ${RUN_DIR:-...} style defaults).

Usage examples:
  python scripts/generate_moe.py --sequences-file my_seqs.txt --out-dir out/moe_results
  python scripts/generate_moe.py --run-id run_20260102_132527_moe --out-dir out/moe_results

"""
import argparse
import os
import subprocess
import sys
import datetime
import json

# ensure repo utils are importable
sys.path.append(os.getcwd())
from utils import utils_function
from utils import generate_conformation


def generate_sdfs_from_sequences(sequences, base_run):
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

    dir_data = os.path.join(base_run, 'data')
    dir_sdf = os.path.join(base_run, 'sdf')
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_sdf, exist_ok=True)

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
        print('No valid sequences to generate SDFs.', file=sys.stderr)
        sys.exit(2)

    import pandas as pd
    df = pd.DataFrame(data_rows)
    df.to_csv(os.path.join(dir_data, 'new_data.csv'), index=False)

    cfg_path = os.path.join('config', 'CycPeptMP.json')
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path, 'r'))
        except Exception:
            cfg = {}

    utils_function.get_unique_monomer(df, os.path.join(dir_data, 'unique_monomer.csv'))
    utils_function.enumerate_smiles(df, cfg, os.path.join(dir_data, 'enum_smiles.csv'))

    df_enu = pd.read_csv(os.path.join(dir_data, 'enum_smiles.csv'))
    generate_conformation.generate_peptide_conformation(cfg, df_enu, os.path.join(dir_sdf, 'peptide.sdf'))
    df_monomer = pd.read_csv(os.path.join(dir_data, 'unique_monomer.csv'))
    generate_conformation.generate_monomer_conformation(cfg, df_monomer, os.path.join(dir_sdf, 'monomer.sdf'))

    return os.path.join(dir_sdf, 'peptide.sdf'), os.path.join(dir_sdf, 'monomer.sdf')


def call_bash_script(script_path, env_overrides, dry_run=False):
    cmd_env = os.environ.copy()
    cmd_env.update(env_overrides)
    cmd = ['bash', script_path]
    print('Calling:', ' '.join(cmd), 'with env:', {k: env_overrides[k] for k in env_overrides})
    if dry_run:
        return 0, 'dry-run'
    try:
        proc = subprocess.run(cmd, env=cmd_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
        return proc.returncode, proc.stdout
    except Exception as e:
        return 1, str(e)


def main():
    p = argparse.ArgumentParser(description='Run provided MOE bash scripts to create MDB outputs.')
    p.add_argument('--run-id', help='Use an existing run folder under runs/<run-id> (will use its sdf/*).')
    p.add_argument('--sequences-file', help='Plain txt file with one AA sequence per line to generate SDFs.')
    p.add_argument('--sequence', help='Single AA sequence string to generate SDF for (alternative to --sequences-file)')
    p.add_argument('--out-dir', required=True, help='Directory to write MDB/CSV outputs to (OUT_DIR passed to bash scripts)')
    p.add_argument('--moe-bin', help='Path to moebatch executable (optional). Will be passed to bash scripts as MOE_BIN env var.')
    p.add_argument('--scripts-dir', default='scripts', help='Directory containing run_moe_2d.sh and run_moe_3d.sh')
    p.add_argument('--dry-run', action='store_true', help='Print actions without executing the bash scripts')
    args = p.parse_args()

    # Determine run folder and SDFs
    if args.run_id and not (args.sequences_file or args.sequence):
        base_run = os.path.join('runs', args.run_id)
        if not os.path.exists(base_run):
            print(f'Error: run folder not found: {base_run}', file=sys.stderr)
            sys.exit(2)
        dir_sdf = os.path.join(base_run, 'sdf')
        peptide_sdf = os.path.join(dir_sdf, 'peptide.sdf')
        monomer_sdf = os.path.join(dir_sdf, 'monomer.sdf')
        if not os.path.exists(peptide_sdf) or not os.path.exists(monomer_sdf):
            print('Error: expected SDFs not present under the run folder.', file=sys.stderr)
            sys.exit(2)
    else:
        sequences = []
        if args.sequence:
            sequences.append(args.sequence)
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
            print('Error: no sequences provided; use --sequences-file or --sequence or --run-id', file=sys.stderr)
            sys.exit(2)

        # create a new run folder
        run_id = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S_moe')
        base_run = os.path.join('runs', run_id)
        os.makedirs(base_run, exist_ok=True)
        peptide_sdf, monomer_sdf = generate_sdfs_from_sequences(sequences, base_run)
        print('Generated SDFs under:', os.path.join(base_run, 'sdf'))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # paths to the user-provided bash scripts
    script_2d = os.path.join(args.scripts_dir, 'run_moe_2d.sh')
    script_3d = os.path.join(args.scripts_dir, 'run_moe_3d.sh')
    if not os.path.exists(script_2d) or not os.path.exists(script_3d):
        print('Error: expected bash scripts not found under', args.scripts_dir, file=sys.stderr)
        sys.exit(2)

    # Prepare environment overrides for the bash scripts. These env vars will be available
    # inside the scripts; the scripts should be written to respect them (use ${VAR:-default}).
    env_overrides = {
        'RUN_DIR': os.path.abspath(os.path.join(base_run)),
        'OUT_DIR': os.path.abspath(out_dir),
    }
    if args.moe_bin:
        env_overrides['MOE_BIN'] = args.moe_bin

    # Call 2D then 3D scripts
    print('=== Executing 2D import script ===')
    rc2, out2 = call_bash_script(script_2d, env_overrides, dry_run=args.dry_run)
    print('2D script return code:', rc2)

    print('\n=== Executing 3D descriptor script ===')
    rc3, out3 = call_bash_script(script_3d, env_overrides, dry_run=args.dry_run)
    print('3D script return code:', rc3)

    # Summarize outputs in OUT_DIR
    print('\nMOE run summary (files in OUT_DIR):')
    try:
        for root, _, files in os.walk(out_dir):
            for f in files:
                fp = os.path.join(root, f)
                size = os.path.getsize(fp)
                print(' ', os.path.relpath(fp, out_dir), '-', size, 'bytes')
    except Exception as e:
        print('Could not list OUT_DIR contents:', e)

    print('\nNote: This script does not convert MDB -> CSV. Please convert MDBs to CSVs manually and place them into the descriptor output folder when ready.')


if __name__ == '__main__':
    main()
