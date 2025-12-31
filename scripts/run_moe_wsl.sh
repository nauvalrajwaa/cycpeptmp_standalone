#!/usr/bin/env bash
# WSL wrapper to run MOE (Windows moebatch.exe) from WSL using `generate_moe.py`.
#
# Usage examples:
#  # If moebatch.exe is at C:\Users\You\moebatch.exe and you have a run folder
#  ./scripts/run_moe_wsl.sh --win-path 'C:\\Users\\You\\moebatch.exe' --run-id run_20251230_204756 --out-dir /mnt/c/Users/You/moe_out
#
#  # Or default to Windows user home moebatch.exe
#  ./scripts/run_moe_wsl.sh --run-id run_20251230_204756 --out-dir /mnt/c/Users/You/moe_out
#
set -euo pipefail

show_help() {
  cat <<EOF
Usage: $0 [--win-path <Windows path to moebatch.exe>] --run-id <run_id> --out-dir <out_dir> [--dry-run] [--job-peptide-2d JOB] [--job-peptide-3d JOB] [--job-monomer-2d JOB] [--job-monomer-3d JOB]

This wrapper converts a Windows moebatch.exe path into a WSL path and calls scripts/generate_moe.py with --moebatch set.
If --win-path is omitted, defaults to C:\\Users\\$USER\\moebatch.exe (adjust if needed).
Note: provide Windows paths with backslashes escaped or quoted.
EOF
}

WIN_PATH=""
RUN_ID=""
OUT_DIR=""
DRY_RUN=0
JOB_PEP_2D="compute_peptide_2D.svl"
JOB_PEP_3D="compute_peptide_3D.svl"
JOB_MON_2D="compute_monomer_2D.svl"
JOB_MON_3D="compute_monomer_3D.svl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --win-path)
      WIN_PATH="$2"; shift 2;;
    --run-id)
      RUN_ID="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    --job-peptide-2d)
      JOB_PEP_2D="$2"; shift 2;;
    --job-peptide-3d)
      JOB_PEP_3D="$2"; shift 2;;
    --job-monomer-2d)
      JOB_MON_2D="$2"; shift 2;;
    --job-monomer-3d)
      JOB_MON_3D="$2"; shift 2;;
    -h|--help)
      show_help; exit 0;;
    *)
      echo "Unknown arg: $1"; show_help; exit 2;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "Error: --run-id is required" >&2
  show_help
  exit 2
fi
if [[ -z "$OUT_DIR" ]]; then
  echo "Error: --out-dir is required" >&2
  show_help
  exit 2
fi

# Default Windows moebatch path in user home if not provided
if [[ -z "$WIN_PATH" ]]; then
  WIN_PATH="C:\\Users\\$USER\\moebatch.exe"
  echo "No --win-path given; defaulting to: $WIN_PATH"
fi

# Convert Windows path to WSL path using wslpath
if ! command -v wslpath >/dev/null 2>&1; then
  echo "Error: wslpath not found. This script must run under WSL." >&2
  exit 2
fi

# wslpath expects Windows path with backslashes or drive letter; ensure it's quoted when containing spaces
WSL_MOEBATCH=$(wslpath -u "$WIN_PATH" 2>/dev/null || true)
if [[ -z "$WSL_MOEBATCH" ]]; then
  echo "Could not convert Windows path to WSL path: $WIN_PATH" >&2
  echo "Make sure the path is valid and use an absolute Windows path like C:\\\\Users\\\\You\\\\moebatch.exe" >&2
  exit 2
fi

echo "Using moebatch executable (WSL path): $WSL_MOEBATCH"

PEP_SDF="runs/${RUN_ID}/sdf/peptide.sdf"
MON_SDF="runs/${RUN_ID}/sdf/monomer.sdf"

if [[ ! -f "$PEP_SDF" ]]; then
  echo "Error: peptide.sdf not found at $PEP_SDF" >&2
  exit 2
fi
if [[ ! -f "$MON_SDF" ]]; then
  echo "Error: monomer.sdf not found at $MON_SDF" >&2
  exit 2
fi

CMD=(python3 scripts/generate_moe.py --run-id "$RUN_ID" --out-dir "$OUT_DIR" --moebatch "$WSL_MOEBATCH" --job-peptide-2d "$JOB_PEP_2D" --job-peptide-3d "$JOB_PEP_3D" --job-monomer-2d "$JOB_MON_2D" --job-monomer-3d "$JOB_MON_3D")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run; command would be: ${CMD[*]}"
  exit 0
fi

echo "Running generate_moe.py with Windows moebatch via WSL..."
"${CMD[@]}"
