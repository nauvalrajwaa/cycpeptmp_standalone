#!/usr/bin/env python3
"""Plot loss vs iteration from a designmetrics CSV for paper-quality figures.

Usage:
  python scripts/plot_loss.py from_evobind/designmetrics_8.csv

The script saves a high-resolution PNG and PDF in the output directory.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator


def load_and_process(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Map 'init' to 0, numeric iterations to int
    def iter_to_int(x):
        xs = str(x).strip()
        if xs.lower() == "init":
            return 0
        try:
            return int(xs)
        except Exception:
            # fallback: try float then int
            try:
                return int(float(xs))
            except Exception:
                return None

    df["iter_num"] = df["iteration"].apply(iter_to_int)
    df = df.dropna(subset=["iter_num"])  # drop rows we can't parse
    df["iter_num"] = df["iter_num"].astype(int)
    df = df.sort_values("iter_num")
    return df


def plot_loss(df: pd.DataFrame, out_prefix: Path, dpi: int = 300, figsize=(7, 4.2),
              fit_linear: bool = False, mark_min: bool = False):
    # Use default matplotlib font (clean, paper-appropriate)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=figsize)

    x = df["iter_num"].to_numpy()
    y = df["loss"].to_numpy()

    # Plot raw line (thin) and markers
    ax.plot(x, y, color="#1f77b4", lw=1.2, alpha=0.95, label="Total loss")
    ax.scatter(x, y, color="#1f77b4", s=20, alpha=0.95)

    # Optional linear fit
    if fit_linear:
        # fit y = a*x + b
        coeffs = np.polyfit(x, y, deg=1)
        a, b = coeffs[0], coeffs[1]
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, color="#ff7f0e", lw=1.8, label="Linear fit")
        # Pearson r
        try:
            r = np.corrcoef(x, y)[0, 1]
        except Exception:
            r = float('nan')
        ax.text(0.02, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))

    # Mark minimum loss point if requested
    if mark_min:
        idx_min = int(np.nanargmin(y))
        xm, ym = x[idx_min], y[idx_min]
        ax.plot([xm], [ym], marker="v", color="#2ca02c", markersize=8)
        ax.annotate("min", xy=(xm, ym), xytext=(xm, ym + (max(y) - min(y)) * 0.05),
                    ha="center", fontsize=10,
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    # Axis labels suitable for paper
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Total loss", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0)

    # Legend (only if more than one plotted item)
    if len(ax.get_legend_handles_labels()[0]) > 0:
        ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()

    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)
    return out_png, out_pdf


def main():
    p = argparse.ArgumentParser(description="Plot loss vs iteration from designmetrics CSV")
    p.add_argument("csv", help="Input CSV file (designmetrics)")
    p.add_argument("--out-dir", default="predicted", help="Output directory for figures")
    p.add_argument("--name", default="loss_plot", help="Output filename prefix (no ext)")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI for PNG/PDF (default: 300)")
    p.add_argument("--figsize", type=float, nargs=2, default=(7.0, 4.2), help="Figure size in inches (w h)")
    p.add_argument("--fit-linear", action="store_true", help="Plot linear fit and show Pearson r")
    p.add_argument("--mark-min", action="store_true", help="Mark minimum loss on the plot")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / args.name

    df = load_and_process(csv_path)
    out_png, out_pdf = plot_loss(df, out_prefix, dpi=args.dpi, figsize=tuple(args.figsize),
                                 fit_linear=args.fit_linear, mark_min=args.mark_min)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
