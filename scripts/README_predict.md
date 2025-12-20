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

Flags of interest
- `--csv-input`: CSV with `ID` and `Sequence` (or use `--id-col`/`--seq-col`).
- `--one-letter`: treat sequence as contiguous one-letter codes (e.g. MEGVN).
- `--run-preprocessing`: run descriptor merge + `.npz` generation before predicting.
- `--auto-fix-descriptors`: attempt to fix/fill invalid SMILES in `desc/new_data`.
- `--unique-run`: create a timestamped subfolder under `--input-folder` to avoid overwriting previous runs.
- `--run-id <name>`: explicit subfolder name under `--input-folder` for this run.

Notes
- RDKit/Mordred and other descriptor tools are required for `--run-preprocessing`.
- When using `--run-preprocessing`, generated `.npz` are written under `<input-folder>/<run-id>` (or timestamped folder with `--unique-run`).
- If predictions are `nan`, check generated `.npz` under the run folder for NaNs (feature maps or descriptor gaps).

```
Predict script

Usage

1. Prepare input files as in `model/input/...` (see `Newdata.ipynb` for generation).
2. Run prediction with a checkpoint:

```bash
python scripts/predict.py --checkpoint weight/Fusion/Fusion-60_cv0.cpt --input-folder model/input --replica 60 --set Test --model-type Fusion --output preds.csv
```

Notes
- RDKit is recommended to install via conda-forge.
- The script expects preprocessed `.npz` input files under the input folder.