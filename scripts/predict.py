#!/usr/bin/env python3
"""Predict CLI for CycPeptMP

Usage example:
  python scripts/predict.py \
    --checkpoint weight/Fusion/Fusion-60_cv0.cpt \
    --input-folder model/input \
    --replica 60 --set Test --model-type Fusion \
    --output predictions.csv
"""
import os
import sys
import json
import argparse
import json
import pandas as pd

# Ensure repo root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# utils imports (import lazily where needed to avoid hard dependency at runtime)


def load_checkpoint(model, path, device):
    # import torch locally to avoid relying on a global `torch` name
    import torch
    import torch.nn as nn

    ckpt = torch.load(path, map_location=device)
    # saved state dict was from DataParallel (model.module.state_dict())
    try:
        model = nn.DataParallel(model)
        model.module.load_state_dict(ckpt['model_state_dict'])
    except Exception:
        # fallback: try loading directly into model
        model.load_state_dict(ckpt['model_state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, default=None, help='Path to model checkpoint (required for prediction unless --preprocess-only or --convert-only is used)')
    parser.add_argument('--input-folder', default='model/input')
    parser.add_argument('--run-id', default=None, help='Subfolder under --input-folder to store generated .npz (avoids overwriting).')
    parser.add_argument('--unique-run', action='store_true', help='Create a timestamped unique subfolder under --input-folder for this run')
    parser.add_argument('--replica', type=int, default=60)
    parser.add_argument('--set', dest='set_name', default='Test')
    parser.add_argument('--model-type', default='Fusion', choices=['Fusion','Trans','CNN','MLP'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='predictions.csv')
    # Conversion options
    parser.add_argument('--csv-input', default=None, help='CSV file with ID and sequence tokens (monomer symbols).')
    parser.add_argument('--seq-delim', default=None, help='Delimiter for sequence tokens (default: whitespace).')
    parser.add_argument('--id-col', default='ID', help='Column name for sample ID in input CSV.')
    parser.add_argument('--seq-col', default='Sequence', help='Column name for sequence in input CSV.')
    parser.add_argument('--one-letter', action='store_true', help='Treat sequences as contiguous one-letter codes (e.g. MEGVN => M E G V N). Overrides --seq-delim.')
    parser.add_argument('--check-env', action='store_true', help='Print Python, rdkit, numpy, and torch versions then exit.')
    parser.add_argument('--convert-only', action='store_true', help='Only perform conversion, skip prediction.')
    parser.add_argument('--run-preprocessing', action='store_true', help='Run descriptor merge and generate input .npz files after conversion (requires RDKit/Mordred).')
    parser.add_argument('--preprocess-only', action='store_true', help='Run preprocessing (descriptor merge + .npz generation) then exit; does not require --checkpoint.')
    parser.add_argument('--auto-fix-descriptors', action='store_true', help='Try to auto-fill or drop invalid SMILES in desc/new_data after descriptor merge.')
    args = parser.parse_args()

    # Environment check: print versions and exit
    if getattr(args, 'check_env', False):
        import platform
        print('Python executable:', sys.executable)
        print('Python version:', platform.python_version())
        try:
            import rdkit
            print('rdkit:', rdkit.__version__)
        except Exception:
            print('rdkit: not installed')
        try:
            import numpy as _npv
            print('numpy:', _npv.__version__)
        except Exception:
            print('numpy: not installed')
        try:
            import torch as _tv
            print('torch:', _tv.__version__)
        except Exception:
            print('torch: not installed')
        return

    # If requested, create a run-specific subfolder under the input folder so
    # generated .npz files do not overwrite previous runs. This will adjust
    # `args.input_folder` for the rest of the script.
    if getattr(args, 'unique_run', False) or getattr(args, 'run_id', None):
        run_id = None
        if getattr(args, 'unique_run', False) and not getattr(args, 'run_id', None):
            from datetime import datetime
            run_id = datetime.now().strftime('%Y%m%dT%H%M%S')
        else:
            run_id = args.run_id
        if run_id:
            args.input_folder = os.path.join(args.input_folder, str(run_id))
            os.makedirs(args.input_folder, exist_ok=True)
            print('Using run-specific input folder:', args.input_folder)

    # Load config directly so conversion can run without torch installed
    config = json.load(open('config/CycPeptMP.json', 'r'))
    best_trial = config['model']

    # If user provided a CSV, convert to model/input format (data/new_data) first
    if args.csv_input is not None:
        # Build new_data dataframe compatible with repo structure
        df_in = pd.read_csv(args.csv_input)
        seq_delim = args.seq_delim
        mono_ref = pd.read_csv('data/unique_monomer.csv', low_memory=False)
        symbol_to_smiles = dict(zip(mono_ref['Symbol'].astype(str), mono_ref['SMILES'].astype(str)))

        rows = []
        used_monos_set = set()
        for _, r in df_in.iterrows():
            sid = r.get(args.id_col)
            seq = r.get(args.seq_col)
            if pd.isna(seq):
                tokens = []
            else:
                if args.one_letter:
                    s = str(seq).replace(' ', '').strip().upper()
                    tokens = list(s) if s else []
                else:
                    if seq_delim:
                        tokens = str(seq).split(seq_delim)
                    else:
                        tokens = str(seq).split()

            subs = [symbol_to_smiles.get(t, '') for t in tokens]
            used_monos_set.update([s for s in subs if s])
            # Prepare a row similar to data/new_data/new_data.csv
            max_sub = 15
            subdict = {f'Substructure-{i+1}': (subs[i] if i < len(subs) else '') for i in range(max_sub)}
            combined_smiles = '.'.join([s for s in subs if s])
            rows.append({
                'ID': sid,
                'ID_org': sid,
                'SMILES': combined_smiles,
                'Monomer_number': len(tokens),
                'Monomer_number_in_main_chain': len(tokens),
                'shape': 'Unknown',
                'permeability': 0,
                **subdict
            })

        df_new = pd.DataFrame(rows)
        os.makedirs('data/new_data', exist_ok=True)
        df_new.to_csv('data/new_data/new_data.csv', index=False)
        # Build unique_monomer.csv in data/new_data by filtering global list
        df_mono_used = mono_ref[mono_ref['SMILES'].isin(list(used_monos_set))]
        df_mono_used.to_csv('data/new_data/unique_monomer.csv', index=False)

        # Try to enumerate SMILES (creates data/new_data/enum_smiles.csv) if RDKit is available
        try:
            from utils import utils_function
            utils_function.enumerate_smiles(df_new[['ID','SMILES']].rename(columns={'ID':'ID','SMILES':'SMILES'}), config, 'data/new_data/enum_smiles.csv')
            print('Wrote data/new_data/new_data.csv and data/new_data/enum_smiles.csv')
        except Exception as e:
            print('Wrote data/new_data/new_data.csv. Skipped enumerate_smiles (RDKit or utils_function import failed):', e)

        if args.run_preprocessing:
            try:
                # Merge descriptors (uses files under desc/new_data)
                from utils import calculate_descriptors, generate_monomer_input, generate_atom_input, generate_peptide_input
                calculate_descriptors.merge_descriptors(config, 'desc/new_data', 'data/new_data')

                # Load descriptor files
                df_mono_2D = pd.read_csv('desc/new_data/monomer_2D.csv', low_memory=False)
                df_mono_3D = pd.read_csv('desc/new_data/monomer_3D.csv', low_memory=False)
                df_pep_2D = pd.read_csv('desc/new_data/peptide_2D.csv', low_memory=False)
                df_pep_3D = pd.read_csv('desc/new_data/peptide_3D.csv', low_memory=False)

                # Quick SMILES validation to catch missing/invalid entries early
                try:
                    from rdkit import Chem
                except Exception:
                    raise

                bad = []
                def _check(df, dfname):
                    if 'SMILES' not in df.columns:
                        return
                    for idx, smi in df['SMILES'].items():
                        if pd.isna(smi) or str(smi).strip() == '':
                            bad.append((dfname, idx, smi))
                        else:
                            try:
                                if Chem.MolFromSmiles(str(smi)) is None:
                                    bad.append((dfname, idx, smi))
                            except Exception:
                                bad.append((dfname, idx, smi))

                _check(df_mono_2D, 'monomer_2D')
                _check(df_mono_3D, 'monomer_3D')
                _check(df_pep_2D, 'peptide_2D')
                _check(df_pep_3D, 'peptide_3D')

                if bad:
                    print('Found invalid SMILES entries in descriptor files.')
                    for dfname, idx, smi in bad:
                        print(f"  {dfname} row {idx}: {smi}")
                    if getattr(args, 'auto_fix_descriptors', False):
                        print('Attempting auto-fix: fill from mapping files then drop remaining rows')
                        import shutil
                        # attempt to fill desc/new_data/* from data/new_data/unique_monomer.csv and data/unique_monomer.csv
                        def _autofix(path):
                            import pandas as _pd
                            import os as _os
                            if not _os.path.exists(path):
                                return 0,0
                            dfc = _pd.read_csv(path, low_memory=False)
                            before = len(dfc)
                            mask_missing = dfc['SMILES'].isna() | (dfc['SMILES'].astype(str).str.strip().str.lower()=='nan') | (dfc['SMILES'].astype(str).str.strip()=='')
                            if mask_missing.sum()==0:
                                return 0,0
                            maps = []
                            for mpath in ['data/new_data/unique_monomer.csv','data/unique_monomer.csv']:
                                if _os.path.exists(mpath):
                                    mdf = _pd.read_csv(mpath, low_memory=False)
                                    if 'Symbol' in mdf.columns and 'SMILES' in mdf.columns:
                                        maps.append(dict(zip(mdf['Symbol'].astype(str), mdf['SMILES'].astype(str))))
                            filled = 0
                            for i,row in dfc[mask_missing].iterrows():
                                sym = str(row.get('Symbol','')).strip()
                                if not sym:
                                    continue
                                for mp in maps:
                                    if sym in mp and str(mp[sym]).strip()!='':
                                        dfc.at[i,'SMILES'] = mp[sym]
                                        filled += 1
                                        break
                            mask_missing = dfc['SMILES'].isna() | (dfc['SMILES'].astype(str).str.strip().str.lower()=='nan') | (dfc['SMILES'].astype(str).str.strip()=='')
                            dropped = int(mask_missing.sum())
                            if dropped>0:
                                dfc = dfc[~mask_missing]
                            # backup and write
                            shutil.copy(path, path+'.bak')
                            dfc.to_csv(path, index=False)
                            return filled, dropped

                        f1,d1 = _autofix('desc/new_data/monomer_3D.csv')
                        print(f'auto-fixed monomer_3D: filled={f1} dropped={d1}')
                        # re-load files and re-check
                        df_mono_3D = pd.read_csv('desc/new_data/monomer_3D.csv', low_memory=False)
                        bad = []
                        _check(df_mono_2D, 'monomer_2D')
                        _check(df_mono_3D, 'monomer_3D')
                        _check(df_pep_2D, 'peptide_2D')
                        _check(df_pep_3D, 'peptide_3D')
                        if bad:
                            print('Auto-fix incomplete; aborting preprocessing. Remaining invalid SMILES:')
                            for dfname, idx, smi in bad:
                                print(f"  {dfname} row {idx}: {smi}")
                            raise ValueError('Invalid SMILES detected; auto-fix did not resolve all issues')
                        print('Auto-fix succeeded; continuing preprocessing')
                    else:
                        print('Aborting preprocessing. Use --auto-fix-descriptors to attempt automatic fixes.')
                        raise ValueError('Invalid SMILES detected; fix descriptor inputs before preprocessing')

                # Generate inputs (.npz)
                generate_monomer_input.generate_monomer_input(config, df_new, df_mono_2D, df_mono_3D, args.input_folder, args.set_name)

                # Prepare mols for atom input from enumerated smiles
                df_enu = pd.read_csv('data/new_data/enum_smiles.csv')
                from rdkit import Chem
                mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in df_enu['SMILES'].tolist()]
                generate_atom_input.generate_atom_input(config, df_new, df_enu, mols, args.input_folder, args.set_name)

                generate_peptide_input.generate_peptide_input(config, df_new, df_enu, df_pep_2D, df_pep_3D, args.input_folder, args.set_name)

                print('Preprocessing complete: model input .npz files generated under', args.input_folder)
            except Exception as e:
                import traceback
                print('Preprocessing failed (missing dependency or runtime error):')
                traceback.print_exc()

        if args.convert_only:
            print('Conversion finished (convert-only). Exiting.')
            return

        if args.preprocess_only:
            print('Preprocessing finished (preprocess-only). Exiting.')
            return

    # At this point we need torch and model_utils for prediction — import lazily so conversion-only works without torch
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from model import model_utils
    except Exception as e:
        print('Failed to import PyTorch/model utilities:', e)
        return

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # create model
    use_aux = bool(best_trial.get('use_auxiliary', False))

    # Preflight: inspect generated .npz files for expected shapes
    try:
        import numpy as _np
        max_atom = config['data']['max_atommun']
        node_npz = f"{args.input_folder}/Trans/{args.replica}/node_{args.replica}_{args.set_name}.npz"
        conf_npz = f"{args.input_folder}/Trans/{args.replica}/conf_{args.replica}_{args.set_name}.npz"
        graph_npz = f"{args.input_folder}/Trans/{args.replica}/graph_{args.replica}_{args.set_name}.npz"
        if not (os.path.exists(node_npz) and os.path.exists(conf_npz) and os.path.exists(graph_npz)):
            print('Expected model input .npz files missing under', args.input_folder)
            print('Check that --run-preprocessing or --preprocess-only completed successfully')
            return

        _node = _np.load(node_npz)
        _conf = _np.load(conf_npz)
        _graph = _np.load(graph_npz)
        print('Input shapes: node.atoms_mask', getattr(_node, 'files', None) and _node['atoms_mask'].shape, 'conf', _conf['conf'].shape, 'graph', _graph['graph'].shape)
        if _conf['conf'].ndim != 3 or _conf['conf'].shape[1] != max_atom or _conf['conf'].shape[2] != max_atom:
            print('Unexpected `conf` shape:', _conf['conf'].shape, 'expected (N,', max_atom, max_atom, ')')
            print('Aborting to avoid runtime error in model. You may need to re-run preprocessing.')
            return
    except Exception as e:
        print('Preflight check failed:', e)
        return
    model = model_utils.create_model(best_trial, device, use_aux)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    # prepare dataset and dataloader
    dataset = model_utils.load_dataset(args.input_folder, args.model_type, args.replica, args.set_name, _10cv=False)
    batch_size = int(best_trial.get('params_batch_size', 64))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # predict
    ids, exps, preds = model_utils.predict_valid(device, model, dataloader, criterion=None, istrain=False,
                                                use_auxiliary=use_aux,
                                                gamma_layer=best_trial.get('gamma_layer', None),
                                                gamma_subout=best_trial.get('gamma_subout', None))

    # save CSV
    import csv
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','exp','pred'])
        import numpy as _np
        for i,e,p in zip(ids, exps, preds):
            # p may be a list/array like [value], handle both cases
            if isinstance(p, (list, tuple)) or (_np and isinstance(p, _np.ndarray)):
                try:
                    val = float(_np.array(p).ravel()[0])
                except Exception:
                    val = float(p[0])
            else:
                val = float(p)
            writer.writerow([i, e, val])

    print(f'Wrote predictions to {args.output}')


if __name__ == '__main__':
    main()
