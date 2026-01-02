#!/usr/bin/env python3
"""Create a funnel-style chart showing progressive filtering counts.

Usage examples:
    python scripts/funnel_chart.py --initial 256 --struct-count 150 --final 5
    python scripts/funnel_chart.py --design-csv from_evobind/designmetrics_8.csv --out-dir plot --name funnel_8

Outputs PNG and PDF in the output directory.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import pandas as pd
import numpy as np


def compute_structural_count(design_csv: Path):
    df = pd.read_csv(design_csv)
    cond = (df.get("loss", pd.Series([1]*len(df))) < 0.5) & \
           (df.get("plddt", pd.Series([0]*len(df))) >= 50) & \
           (df.get("plddt") <= 70)
    return int(cond.sum())


def plot_funnel(counts, labels, out_prefix: Path, dpi=300, figsize=(6, 3)):
    # PERBAIKAN 1: figsize tinggi dikurangi jadi 3 (sebelumnya 4)
    # Ini langsung membuang ruang vertikal yang tidak perlu
    
    max_count = max(counts) if counts else 1
    n = len(counts)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Lebar bar tetap panjang (2.5)
    max_bar_width = 2.5 
    height = 0.9
    spacing = 1.05

    # PERBAIKAN 2: Kalkulasi Y-LIMIT yang presisi (Tight)
    # Bar paling atas ada di y=0. Batas atas cukup +0.6
    top_limit = 0.6
    
    # Bar paling bawah ada di index -(n-1). 
    # Kita hitung posisi tepatnya biar tidak ada sisa ruang bawah.
    last_bar_y = -((n - 1) * spacing)
    bottom_limit = last_bar_y - 0.6  # Beri sedikit margin 0.6 ke bawah
    
    ax.set_ylim(bottom_limit, top_limit)
    
    # X-LIMIT tetap compact (-3.2 sampai 3.2)
    ax.set_xlim(-3.2, 3.2)
    ax.axis('off')

    cmap = mpl.colormaps.get_cmap('RdYlGn_r') if hasattr(mpl, 'colormaps') else mpl.cm.get_cmap('RdYlGn_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    
    # Posisi teks
    left_label_x = -1.4
    right_label_x = 1.4
    
    for i, (c, lab) in enumerate(zip(counts, labels)):
        frac = c / max_count if max_count > 0 else 0
        
        calc_width = max_bar_width * frac
        width = calc_width if calc_width > 0.08 else 0.08
        
        left = -width / 2
        color = cmap(norm(c))
        
        rect = patches.FancyBboxPatch((left, -i * spacing - height/2), width, height,
                                      boxstyle="round,pad=0.02", linewidth=0.6, 
                                      facecolor=color, edgecolor='k', alpha=0.95)
        ax.add_patch(rect)
        
        # Label Kiri
        ax.text(left_label_x, -i * spacing, lab, 
                va='center', ha='right', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none'))
        
        pct = (c / counts[0] * 100) if counts[0] > 0 else 0
        
        # Label Kanan
        ax.text(right_label_x, -i * spacing, f"{c} ({pct:.1f}%)", 
                va='center', ha='left', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none'))

    # Gunakan tight_layout dengan pad kecil
    plt.tight_layout(pad=0.5)
    
    out_png = out_prefix.with_suffix('.png')
    out_pdf = out_prefix.with_suffix('.pdf')
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)
    return out_png, out_pdf


def main():
    p = argparse.ArgumentParser(description="Generate funnel/stacked-bar chart for filtering pipeline")
    p.add_argument("--initial", type=int, help="Initial designs count (overrides design-csv)")
    p.add_argument("--design-csv", help="Design metrics CSV (to compute structural filter)")
    p.add_argument("--struct-count", type=int, help="Pre-computed structural-pass count")
    p.add_argument("--plddt", type=float, help="Optional pLDDT value to display in the structural label")
    p.add_argument("--predict-csv", help="Prediction CSV (sorted) to use for final candidates (uses column 'class'==1)")
    p.add_argument("--final", type=int, default=5, help="Final candidates count")
    p.add_argument("--out-dir", default="plot", help="Output directory")
    p.add_argument("--name", default="funnel", help="Output filename prefix")
    p.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / args.name

    initial = args.initial
    if initial is None and args.design_csv:
        try:
            df = pd.read_csv(args.design_csv)
            initial = int(len(df))
        except Exception:
            initial = None
    if initial is None:
        initial = 256

    if args.struct_count is not None:
        struct_count = args.struct_count
    elif args.design_csv:
        struct_count = compute_structural_count(Path(args.design_csv))
    else:
        struct_count = max(int(initial * 0.6), 1)

    final = args.final

    # If a prediction CSV is provided, use it to set the second and third values
    # second value = total sequences in the prediction CSV
    # third value = count of rows with class_label == 'HIGH' (fallback to class==1)
    if args.predict_csv:
        try:
            df_pred = pd.read_csv(args.predict_csv)
            total_preds = len(df_pred)
            if 'class_label' in df_pred.columns:
                high_count = int((df_pred['class_label'] == 'HIGH').sum())
            elif 'class' in df_pred.columns:
                high_count = int((df_pred['class'] == 1).sum())
            else:
                high_count = int(final)
            struct_count = int(total_preds)
            final = int(high_count)
        except Exception:
            # if reading fails, keep previously computed values
            pass

    counts = [initial, struct_count, final]
    
    if args.plddt is not None:
        struct_label = f"Structural filter\n(loss<0.5, pLDDT={args.plddt})"
    else:
        struct_label = f"Structural filter\n(loss<0.5, pLDDT 50-70)"

    labels = [
        "Initial designs",
        struct_label,
        "Final candidates",
    ]

    out_png, out_pdf = plot_funnel(counts, labels, out_prefix, dpi=args.dpi)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()