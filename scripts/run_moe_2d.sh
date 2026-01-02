#!/bin/bash

# =================================================================
# KONFIGURASI
# =================================================================
MOE_BIN="/mnt/c/Program Files/moe2022/bin/moebatch.exe"
RUN_DIR="runs/run_20260102_132527_moe" 
OUT_DIR="out/moe_results" # Folder output

mkdir -p "$OUT_DIR"

# =================================================================
# FUNGSI IMPORT ONLY (SDF -> MDB)
# =================================================================

run_moe_mdb() {
    local BASENAME=$1
    echo "========================================================="
    echo "PROCESSING: IMPORT SDF TO MDB ($BASENAME)"
    echo "========================================================="

    local LINUX_IN="$RUN_DIR/sdf/${BASENAME}.sdf"
    local LINUX_OUT_MDB="$OUT_DIR/${BASENAME}_final.mdb"
    
    # Cek input
    if [ ! -f "$LINUX_IN" ]; then
        echo "Error: Input $LINUX_IN not found."
        return
    fi

    # 1. KONVERSI PATH LINUX -> WINDOWS
    local WIN_IN=$(wslpath -m "$(readlink -f "$LINUX_IN")")
    local WIN_OUT_MDB=$(wslpath -m "$(readlink -f "$LINUX_OUT_MDB")")
    
    # Bersihkan file output lama jika ada
    rm -f "$LINUX_OUT_MDB"

    # -----------------------------------------------------------
    # TAHAP: IMPORT ONLY
    # -----------------------------------------------------------
    echo "   [1/1] Converting SDF to MDB..."
    
    # PERINTAH SVL:
    # 1. db_Open: Buat MDB baru.
    # 2. db_ImportSD: Import struktur dari SDF ke kolom 'mol'.
    # 3. exit: Selesai.
    # (Tidak ada hitung charge, tidak ada hitung descriptor)
    
    "$MOE_BIN" -exec "db_Open ['$WIN_OUT_MDB', 'create']; db_ImportSD ['$WIN_OUT_MDB', '$WIN_IN', 'mol']; exit [];"

    # -----------------------------------------------------------
    # TAHAP: VERIFIKASI
    # -----------------------------------------------------------
    
    if [ -f "$LINUX_OUT_MDB" ]; then
        local SIZE=$(wc -c < "$LINUX_OUT_MDB")
        if [ "$SIZE" -gt 100 ]; then
            echo "   [SUCCESS] MDB saved: $LINUX_OUT_MDB"
        else
            echo "   [WARNING] MDB created but seems empty."
        fi
    else
        echo "   [ERROR] Failed to create MDB."
    fi
    echo ""
}

# =================================================================
# EKSEKUSI
# =================================================================

run_moe_mdb "peptide"
run_moe_mdb "monomer"