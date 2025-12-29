```markdown
Predict script

Usage

1. Convert sequences → preprocess → predict (end‑to‑end, safe non‑overwriting run):

```bash
python scripts/predict.py \
  --csv-input your_input.csv --one-letter \
  --run-preprocessing --auto-fix-descriptors --unique-run --input-folder model/input \
  --checkpoint weight/Fusion/Fusion-60_cv0.cpt --replica 60 --set new \
  --model-type Fusion --device cpu --output predictions_endtoend.csv
```

2. Quick predict using existing preprocessed `.npz` under `model/input`:

```bash
python scripts/predict.py --checkpoint weight/Fusion/Fusion-60_cv0.cpt --input-folder model/input --replica 60 --set Test --model-type Fusion --output preds.csv
```

# Predict pipeline and evobind filter

## Summary of recent changes

- `scripts/predict.py` updates:
  - Outputs organized under a per-run folder (default base `runs/`); use `--outdir` to change the base.
  - `--run-preprocessing` will write preprocessing outputs under `runs/<run_id>/...` (or a timestamped folder when using `--unique-run`).
  - If `moebatch` is not found, MOE descriptors are skipped and missing descriptor columns are filled from `config` means/stds so the pipeline runs without MOE.
  - After predicting, the script calls `scripts/process_predictions.py` to produce a normalized, classified, and sorted CSV (`*_sorted.csv`).

- `scripts/filter_evobind.py` (previously `filter_plddt.py`):
  - Filters `from_evobind/designmetrics.csv` (or any CSV with `plddt` and `sequence`) by pLDDT and optional `loss` threshold.
  - Supports comparison operators (`gte`, `gt`, `lte`, `lt`) for both `plddt` and `loss`.
  - Removes duplicate sequences (preserves first occurrence) and writes a plain text file with one sequence per line (ready to pass to `scripts/predict.py --file`).

## Quick usage

- End-to-end predict (run preprocessing + predict):

```bash
python scripts/predict.py \
  --csv-input your_input.csv --one-letter \
  --run-preprocessing --unique-run --outdir runs \
  --checkpoint weight/Fusion/Fusion-60_cv0.cpt --replica 60 --model-type Fusion \
  --device cpu --output predicted/new_prediction.csv
```

Outputs will be under `runs/<run_id>/predicted/`; a sorted file `new_prediction_sorted.csv` is created automatically.

- Quick predict using an existing preprocessed run folder:

```bash
python scripts/predict.py \
  --checkpoint weight/Fusion/Fusion-60_cv0.cpt \
  --input-folder runs/<run_id>/model_input --replica 60 --set Test --model-type Fusion \
  --outdir runs --output predicted/new_prediction.csv
```

- Filter `from_evobind/designmetrics.csv` (example):

```bash
python scripts/filter_evobind.py \
  --input from_evobind/designmetrics.csv \
  --threshold 70 --op gte \
  --loss 0.5 --loss-op lte \
  --output from_evobind/test_filtered.txt
```

Then run predictions on the filtered unique sequences:

```bash
python scripts/predict.py --file from_evobind/test_filtered.txt --outdir runs
```

## Notes and tips

- RDKit (and Mordred if used) should be installed in the active Python environment when running preprocessing.
- MOE (`moebatch`) is optional; if present the MOE 3D descriptor step will run. If absent, the pipeline fills missing descriptor columns using `config/CycPeptMP.json` defaults.
- If you see `nan` predictions, inspect the run's `model_input` `.npz` files under the run folder for missing or NaN features.

## Contact

- If you want the README adjusted or examples for other checkpoints/ensembles, tell me which checkpoint and I can add the exact command.
