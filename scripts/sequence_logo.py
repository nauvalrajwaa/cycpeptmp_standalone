import argparse
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import os
import sys

def create_peptide_logo(input_file, out_dir, name, col_name='sequence', clean_noise=True):
    # 1. Cek file input
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' tidak ditemukan.")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error baca CSV: {e}")
        sys.exit(1)

    if col_name not in df.columns:
        print(f"Error: Kolom '{col_name}' tidak ditemukan.")
        sys.exit(1)

    # 2. Ambil sequence
    sequences = df[col_name].dropna().astype(str).tolist()
    if not sequences:
        print("Error: Data kosong.")
        sys.exit(1)

    # Filter panjang sequence agar seragam
    lengths = [len(s) for s in sequences]
    most_common_len = max(set(lengths), key=lengths.count)
    sequences = [s for s in sequences if len(s) == most_common_len]
    print(f"Memproses {len(sequences)} sequence (panjang: {most_common_len})...")

    # 3. Buat Matrix
    try:
        # matrix information bits
        ww_counts = logomaker.alignment_to_matrix(sequences=sequences, to_type='information', characters_to_ignore='.-X*')
        
        # --- CLEAN NOISE ---
        # Hapus huruf kecil (sampah) biar rapi
        if clean_noise:
            ww_counts[ww_counts < 0.10] = 0.0
            
    except Exception as e:
        print(f"Error matriks: {e}")
        sys.exit(1)

    # 4. Plotting
    fig_width = max(8, len(sequences[0]) * 0.8) 
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    logo = logomaker.Logo(ww_counts,
                          color_scheme='skylign_protein',
                          vpad=.05,
                          width=.9,
                          ax=ax)

    # Styling
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    
    ax.set_ylabel("Bits", fontsize=14)
    ax.set_xlabel("Position", fontsize=14)
    
    # --- PERUBAHAN DI SINI ---
    # Baris title di bawah ini sudah dihapus/dikomentari
    # ax.set_title(f"Sequence Logo: {name}", fontsize=16) 
    
    # Fix Skala Y (4.5 bits)
    ax.set_ylim([0, 4.5])
    ax.set_xticks(range(len(ww_counts)))

    # 5. Simpan
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output_path = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Sukses! Plot bersih tersimpan di: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('--out-dir', type=str, default='plot')
    parser.add_argument('--name', type=str, default='seq_logo')
    parser.add_argument('--col', type=str, default='sequence')
    parser.add_argument('--keep-noise', action='store_true', help='Jangan hapus huruf-huruf kecil (noise)')

    args = parser.parse_args()
    do_clean = not args.keep_noise

    create_peptide_logo(args.input_file, args.out_dir, args.name, args.col, do_clean)