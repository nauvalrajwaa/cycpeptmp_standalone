import json
import os
import pandas as pd
import numpy as np
import argparse
from rdkit import Chem
import torch
import torch.nn as nn
import sys
import warnings
import datetime
import shutil
import subprocess

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# 1. SETUP & IMPORTS
# ------------------------------------------------------------------------------
sys.path.append(os.getcwd())

try:
    from utils import utils_function
    from utils import calculate_descriptors
    from utils import generate_conformation
    from utils import generate_atom_input
    from utils import generate_monomer_input
    from utils import generate_peptide_input
    from model import model_utils
except ImportError as e:
    print("Error: Could not import required modules.")
    print("Make sure you are in the folder containing 'utils/' and 'model/' directories.")
    print(f"Details: {e}")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# 2. DEFINITIONS & MAPPING
# ------------------------------------------------------------------------------
AA_TO_SMILES = {
    'A': 'CN[C@@H](C)C=O',                  # Alanine
    'C': 'CN[C@H](C=O)CS',                  # Cysteine
    'D': 'CN[C@H](C=O)CC(=O)O',             # Aspartic Acid
    'E': 'CN[C@H](C=O)CCC(=O)O',            # Glutamic Acid
    'F': 'CN[C@H](C=O)Cc1ccccc1',           # Phenylalanine
    'G': 'CNCC=O',                          # Glycine
    'H': 'CN[C@H](C=O)Cc1c[nH]cn1',         # Histidine
    'I': 'CC[C@H](C)[C@@H](C=O)NC',         # Isoleucine
    'K': 'CN[C@H](C=O)CCCCN',               # Lysine
    'L': 'CN[C@H](C=O)CC(C)C',              # Leucine
    'M': 'CN[C@H](C=O)CCSC',                # Methionine
    'N': 'CN[C@H](C=O)CC(N)=O',             # Asparagine
    'P': 'CN1CCC[C@H]1C=O',                 # Proline
    'Q': 'CN[C@H](C=O)CCC(N)=O',            # Glutamine
    'R': 'CN[C@H](C=O)CCCNC(=N)N',          # Arginine
    'S': 'CN[C@H](C=O)CO',                  # Serine
    'T': 'CN[C@H](C=O)[C@@H](C)OC',         # Threonine (O-methylated)
    'V': 'CN[C@H](C=O)C(C)C',               # Valine
    'W': 'CN[C@H](C=O)Cc1c[nH]c2ccccc12',   # Tryptophan
    'Y': 'CN[C@H](C=O)Cc1ccc(O)cc1'         # Tyrosine
}

def sequence_to_data(sequences):
    """Converts a list of 1-letter sequences into the DataFrame format."""
    data_rows = []
    
    for idx, seq in enumerate(sequences):
        seq = seq.upper().strip()
        monomers = []
        valid_seq = True
        
        for aa in seq:
            if aa in AA_TO_SMILES:
                monomers.append(AA_TO_SMILES[aa])
            else:
                print(f"Warning: Unknown amino acid '{aa}' in sequence {seq}. Skipping.")
                valid_seq = False
                break
        
        if not valid_seq:
            continue
            
        row = {
            'ID': f'pept{idx+1}',
            'ID_org': f'seq_{idx+1}_{seq}',
            'SMILES': '.'.join(monomers),
            'Monomer_number': len(monomers),
            'Monomer_number_in_main_chain': len(monomers),
            'shape': 'Unknown',
            'permeability': 0
        }
        
        for i, m in enumerate(monomers):
            row[f'Substructure-{i+1}'] = m
            
        data_rows.append(row)
        
    return pd.DataFrame(data_rows)

def sanitize_config(cfg):
    """Recursively disable MOE descriptors in the config dictionary."""
    if isinstance(cfg, dict):
        keys = list(cfg.keys())
        for k in keys:
            v = cfg[k]
            if 'moe' in k.lower() and v is True:
                cfg[k] = False
            
            if isinstance(v, list):
                new_list = [x for x in v if 'moe' not in str(x).lower()]
                cfg[k] = new_list
            
            elif isinstance(v, (dict, list)):
                sanitize_config(v)
    elif isinstance(cfg, list):
        for item in cfg:
            sanitize_config(item)

# ------------------------------------------------------------------------------
# 3. MAIN PIPELINE
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict permeability for cyclic peptides.")
    parser.add_argument('-i', '--input', type=str, help="Single amino acid sequence (e.g., 'ACDEF').")
    parser.add_argument('-f', '--file', type=str, help="File containing sequences (one per line).")
    parser.add_argument('--outdir', type=str, default='runs', help="Base output directory for runs (default: 'runs').")
    parser.add_argument('--resume', type=str, default=None, help='Path to an existing run folder to resume prediction (skips preprocessing).')
    parser.add_argument('--moe-dir', type=str, default=None, help='Directory containing MOE CSVs to integrate (peptide_moe_2D.csv, peptide_moe_3D.csv, monomer_moe_2D.csv, monomer_moe_3D.csv).')
    parser.add_argument('--moebatch', type=str, default=None, help='Path to moebatch executable (Windows or WSL).')
    parser.add_argument('--batch-size', type=int, default=None, help='Optional batch size for prediction DataLoader (default: all samples)')
    args = parser.parse_args()

    sequences = []
    if args.input:
        sequences.append(args.input)
    elif args.file:
        with open(args.file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        print("No input provided. Using example: 'AYMV'")
        sequences = ["AYMV"]

    print(f"Processing {len(sequences)} sequence(s)...")

    new_data = sequence_to_data(sequences)
    if new_data.empty:
        print("No valid data generated.")
        return

    # --- UNIQUE RUN FOLDER ---
    skip_preprocessing = False
    if args.resume:
        # Use provided run folder directly
        base_run = args.resume.rstrip('/')
        if not os.path.exists(base_run):
            print(f'Error: resume path not found: {base_run}')
            return
        dir_data = f'{base_run}/data'
        dir_sdf = f'{base_run}/sdf'
        dir_desc = f'{base_run}/desc'
        dir_model_input = f'{base_run}/model_input'
        dir_predicted = f'{base_run}/predicted'
        skip_preprocessing = True
        print(f'Resuming from existing run folder: {base_run}')
        # load new_data from existing run (if present)
        input_csv_path = f'{dir_data}/new_data.csv'
        if os.path.exists(input_csv_path):
            new_data = pd.read_csv(input_csv_path)
        else:
            print(f'Error: cannot find new_data.csv under {dir_data} to resume.')
            return
    else:
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        print(f"Run ID: {run_id}")

        # Use a single run folder to keep outputs organized per run:
        # <outdir>/{run_id}/data, <outdir>/{run_id}/sdf, <outdir>/{run_id}/desc, <outdir>/{run_id}/model_input, <outdir>/{run_id}/predicted
        base_run = f"{args.outdir}/{run_id}"
        dir_data = f'{base_run}/data'
        dir_sdf = f'{base_run}/sdf'
        dir_desc = f'{base_run}/desc'
        dir_model_input = f'{base_run}/model_input'
        dir_predicted = f'{base_run}/predicted'

        paths = [dir_data, dir_sdf, dir_desc, dir_model_input, dir_predicted]
        for p in paths:
            os.makedirs(p, exist_ok=True)

        input_csv_path = f'{dir_data}/new_data.csv'
        new_data.to_csv(input_csv_path, index=False)
    
    config_path = 'config/CycPeptMP.json'
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    config = json.load(open(config_path, 'r'))

    # Determine whether MOE descriptors are available (either provided by user or moebatch present)
    moes_provided = False
    if args.moe_dir:
        if os.path.exists(args.moe_dir):
            moes_provided = True
            print(f'Using MOE CSVs from: {args.moe_dir}')
        else:
            print(f'Warning: provided --moe-dir does not exist: {args.moe_dir}. Ignoring.')
            args.moe_dir = None

    # WSL-aware moebatch detection. Accepts explicit --moebatch path (Windows or WSL),
    # tries to convert common Windows paths to WSL (/mnt/c/...) and falls back to which().
    def resolve_moebatch(path_arg=None):
        if path_arg:
            # If the provided path exists as-is, use it
            if os.path.exists(path_arg):
                return path_arg
            # Try converting Windows-style path (C:\...) to /mnt/c/... for WSL
            p = path_arg.replace('\\', '/').strip('"')
            if len(p) >= 2 and p[1] == ':' and (p[2] == '/' or p[2] == '\\' or p[2] == ''):
                drive = p[0].lower()
                rest = p[2:]
                if rest.startswith('/'):
                    rest = rest[1:]
                candidate = os.path.join('/mnt', drive, rest)
                if os.path.exists(candidate):
                    return candidate
            # Finally, try to find the binary by basename in PATH
            bn = os.path.basename(path_arg)
            found = shutil.which(bn)
            if found:
                return found
            return None

        # No explicit path: try common names and locations
        found = shutil.which('moebatch') or shutil.which('moebatch.exe')
        if found:
            return found
        # Look under common Windows program folders mounted under /mnt
        candidates = [
            '/mnt/c/Program Files/MOE/moebatch.exe',
            '/mnt/c/Program Files (x86)/MOE/moebatch.exe',
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    moebatch_path = resolve_moebatch(getattr(args, 'moebatch', None))
    if moebatch_path:
        print(f"moebatch detected at: {moebatch_path}")
        moes_available = True
    else:
        moes_available = False

    # If neither moebatch nor user-supplied MOE CSVs exist, disable MOE in config
    if not moes_provided and not moes_available:
        print("Adjusting config to ignore MOE descriptors (MOE not available)...")
        sanitize_config(config)
    else:
        print("MOE descriptors will be used (either from --moe-dir or moebatch present).")

    if not skip_preprocessing:
        print(f"--- Starting Pipeline (Output folder: {run_id}) ---")

        # --- Step 1: Data Preparation ---
        print("Generating unique monomers...")
        utils_function.get_unique_monomer(new_data, f'{dir_data}/unique_monomer.csv')
        
        print("Enumerating SMILES...")
        utils_function.enumerate_smiles(new_data, config, f'{dir_data}/enum_smiles.csv')
        df_enu = pd.read_csv(f'{dir_data}/enum_smiles.csv')
        
        print("Generating peptide conformations...")
        generate_conformation.generate_peptide_conformation(config, df_enu, f'{dir_sdf}/peptide.sdf')
        
        print("Generating monomer conformations...")
        df_monomer = pd.read_csv(f'{dir_data}/unique_monomer.csv')
        generate_conformation.generate_monomer_conformation(config, df_monomer, f'{dir_sdf}/monomer.sdf')
        
        # --- Step 2: Descriptors ---
        print("Calculating descriptors...")
        calculate_descriptors.calc_rdkit_descriptors(new_data['SMILES'].tolist(), f'{dir_desc}/peptide_rdkit.csv')
        calculate_descriptors.calc_rdkit_descriptors(df_monomer['SMILES'].tolist(), f'{dir_desc}/monomer_rdkit.csv')
        
        calculate_descriptors.calc_mordred_2Ddescriptors(new_data['SMILES'].tolist(), f'{dir_desc}/peptide_mordred_2D.csv')
        calculate_descriptors.calc_mordred_2Ddescriptors(df_monomer['SMILES'].tolist(), f'{dir_desc}/monomer_mordred_2D.csv')
        
        mols_pep = Chem.SDMolSupplier(f'{dir_sdf}/peptide.sdf')
        calculate_descriptors.calc_mordred_3Ddescriptors(mols_pep, f'{dir_desc}/peptide_mordred_3D.csv')
        
        mols_mono = Chem.SDMolSupplier(f'{dir_sdf}/monomer.sdf')
        calculate_descriptors.calc_mordred_3Ddescriptors(mols_mono, f'{dir_desc}/monomer_mordred_3D.csv')
        
        # Attempt to generate MOE descriptors if MOE (moebatch) is available.
        if moebatch_path:
            print(f"Found 'moebatch' at {moebatch_path}; attempting to run MOE descriptor script...")
            script = os.path.join('utils', 'MOE_3D_descriptors.sh')
            if os.path.exists(script):
                try:
                    env = os.environ.copy()
                    env['MOEBATCH_PATH'] = moebatch_path
                    ret = subprocess.run(['bash', script, run_id], env=env).returncode
                    if ret != 0:
                        print(f"MOE script exited with code {ret}. Continuing without MOE descriptors.")
                except Exception as e:
                    print(f"MOE script execution failed: {e}. Continuing without MOE descriptors.")
            else:
                print(f"MOE script not found at {script}. Continuing without MOE descriptors.")
        else:
            print("Warning: moebatch not found. Proceeding without MOE descriptors.")

        print("Merging descriptors...")
        # If user provided a MOE directory, copy its CSVs into the run desc folder before merging
        if args.moe_dir:
            for fname in ['peptide_moe_2D.csv', 'peptide_moe_3D.csv', 'monomer_moe_2D.csv', 'monomer_moe_3D.csv']:
                src = os.path.join(args.moe_dir, fname)
                dst = os.path.join(dir_desc, fname)
                try:
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                        print(f'Copied MOE file: {src} -> {dst}')
                    else:
                        print(f'Warning: expected MOE file not found in --moe-dir: {src}')
                except Exception as e:
                    print(f'Could not copy {src} to {dst}: {e}')

        calculate_descriptors.merge_descriptors(config, f'{dir_desc}/', f'{dir_data}/')
        
        # --- Step 3: Model Inputs ---
        print("Generating model inputs...")
        set_name = 'new'
        
        generate_atom_input.generate_atom_input(config, new_data, df_enu, mols_pep, dir_model_input, set_name)
        
        df_mono_2D = pd.read_csv(f'{dir_desc}/monomer_2D.csv')
        df_mono_3D = pd.read_csv(f'{dir_desc}/monomer_3D.csv')
        generate_monomer_input.generate_monomer_input(config, new_data, df_mono_2D, df_mono_3D, dir_model_input, set_name)
        
        df_pep_2D = pd.read_csv(f'{dir_desc}/peptide_2D.csv')
        df_pep_3D = pd.read_csv(f'{dir_desc}/peptide_3D.csv')
        generate_peptide_input.generate_peptide_input(config, new_data, df_enu, df_pep_2D, df_pep_3D, dir_model_input, set_name)
    else:
        # When resuming, ensure we have model_input and set_name
        set_name = 'new'
        print('Skipping preprocessing; proceeding to prediction using existing model_input.')

    # --- Step 4: Prediction ---
    print("Running Prediction...")
    MODEL_TYPE = 'Fusion'
    REPLICA_NUM = 60
    
    dataset_new = model_utils.load_dataset(dir_model_input, MODEL_TYPE, REPLICA_NUM, set_name)
    model_utils.set_seed(config['data']['seed'])
    best_trial = config['model']
    
    dfs = []
    for cv in range(3):
        model_path = f'weight/{MODEL_TYPE}/{MODEL_TYPE}-{REPLICA_NUM}_cv{cv}.cpt'
        
        if not os.path.exists(model_path):
             print(f"  Warning: Checkpoint not found at {model_path}")
             continue

        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
        except Exception:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        model = model_utils.create_model(best_trial, DEVICE, config['model']['use_auxiliary'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(DEVICE)
        
        batch_size = args.batch_size if args.batch_size is not None else (len(dataset_new) if len(dataset_new) > 0 else 1)
        dataloader = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=False)
        
        # Request raw (unclipped) model outputs for inspection, then apply clipping ourselves
        ids, exps, preds = model_utils.predict_valid(
            DEVICE, model, dataloader, None, istrain=False,
            use_auxiliary=config['model']['use_auxiliary'],
            gamma_layer=config['model']['gamma_layer'],
            gamma_subout=config['model']['gamma_subout'],
            clip_outputs=False
        )

        now_pred = pd.DataFrame({'pred_raw': np.array(preds).flatten()})
        now_pred['ID'] = ids
        # Apply clipping according to config limits to get the final 'pred' column
        lower = config['data']['lower_limit']
        upper = config['data']['upper_limit']
        now_pred['pred'] = now_pred['pred_raw'].clip(lower, upper)

        # ensure numeric and aggregate across augmented reps
        now_pred['pred_raw'] = pd.to_numeric(now_pred['pred_raw'], errors='coerce')
        now_pred['pred'] = pd.to_numeric(now_pred['pred'], errors='coerce')

        now_grouped = now_pred.groupby('ID').mean().reset_index()
        dfs.append(now_grouped.set_index('ID'))
    
    if not dfs:
        print("Error: No predictions generated.")
        return

    pred_mean = sum(dfs) / len(dfs)
    pred_mean = pred_mean.reset_index()

    pred_mean['ID'] = pred_mean['ID'].astype(int)
    new_data['join_id'] = new_data['ID'].astype(str).str.extract(r'(\d+)').astype(int)

    result = pd.merge(pred_mean, new_data, left_on='ID', right_on='join_id')

    # Include both raw model outputs and clipped predictions in the final CSV
    # 'pred_raw' = averaged raw model output; 'pred' = averaged clipped output
    final_output = result[['ID_org', 'pred_raw', 'pred', 'SMILES']]

    print("\n=== FINAL PREDICTIONS (raw vs clipped) ===")
    print(final_output[['ID_org', 'pred_raw', 'pred']].to_string(index=False))
    
    output_file = f'{dir_predicted}/{set_name}_prediction.csv'
    final_output.to_csv(output_file, index=False)
    print(f"\nResult saved to: {output_file}")

    # Post-process (normalize/classify/rank) the produced CSV into a sorted file.
    try:
        import subprocess, sys
        proc_script = os.path.join('scripts', 'process_predictions.py')
        if os.path.exists(proc_script):
            subprocess.run([sys.executable, proc_script, output_file], check=False)
            sorted_path = os.path.splitext(output_file)[0] + '_sorted.csv'
            if os.path.exists(sorted_path):
                print(f'Sorted/classified results saved to: {sorted_path}')
        else:
            print('Post-processing script not found; skipping sorted output generation.')
    except Exception as e:
        print(f'Post-processing failed: {e}')

if __name__ == "__main__":
    main()