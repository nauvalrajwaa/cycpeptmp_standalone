#!/bin/bash

# =================================================================
# KONFIGURASI
# =================================================================
MOE_BIN="${MOE_BIN:-/mnt/c/Program Files/moe2022/bin/moebatch.exe}"
# RUN_DIR should be provided by the caller (env) or fall back to the original example
RUN_DIR="${RUN_DIR:-runs/run_20260102_132527_moe}"
# OUT_DIR where MDB/CSV will be written; can be overridden via env
OUT_DIR="${OUT_DIR:-out/moe_results}" # Folder output

mkdir -p "$OUT_DIR"

# =================================================================
# FUNGSI PROSES MDB ONLY
# =================================================================

run_moe_mdb() {
    local BASENAME=$1
    echo "========================================================="
    echo "PROCESSING 3D DESCRIPTORS (MDB OUTPUT): $BASENAME"
    echo "========================================================="

    local LINUX_IN="$RUN_DIR/sdf/${BASENAME}.sdf"
    # Output langsung diberi nama _final.mdb
    local LINUX_OUT_MDB="$OUT_DIR/${BASENAME}_moe_3D.mdb"
    
    # Cek input
    if [ ! -f "$LINUX_IN" ]; then
        echo "Error: Input $LINUX_IN not found."
        return
    fi

    # 1. KONVERSI PATH LINUX -> WINDOWS
    # Menggunakan wslpath -m (format C:/...) agar kompatibel dengan MOE
    local WIN_IN=$(wslpath -m "$(readlink -f "$LINUX_IN")")
    local WIN_OUT_MDB=$(wslpath -m "$(readlink -f "$LINUX_OUT_MDB")")
    
    # Bersihkan file output lama jika ada
    rm -f "$LINUX_OUT_MDB"

    # -----------------------------------------------------------
    # TAHAP: IMPORT & HITUNG & SAVE
    # -----------------------------------------------------------
    echo "   [1/2] Calculating Descriptors into MDB..."
    
    # List Descriptor (Sesuai script anda sebelumnya)
    local DESC_LIST="'ASA','ASA+','ASA-','ASA_H','ASA_P','CASA+','CASA-','DASA','DCASA','dens','dipole','E','E_ang','E_ele','E_nb','E_oop','E_sol','E_stb','E_str','E_strain','E_tor','E_vdw','FASA+','FASA-','FASA_H','FASA_P','FCASA+','FCASA-','glob','npr1','npr2','pmi','pmi1','pmi2','pmi3','rgyr','std_dim1','std_dim2','std_dim3','vol','VSA','vsurf_A','vsurf_CP','vsurf_CW1','vsurf_CW2','vsurf_CW3','vsurf_CW4','vsurf_CW5','vsurf_CW6','vsurf_CW7','vsurf_CW8','vsurf_D1','vsurf_D2','vsurf_D3','vsurf_D4','vsurf_D5','vsurf_D6','vsurf_D7','vsurf_D8','vsurf_DD12','vsurf_DD13','vsurf_DD23','vsurf_DW12','vsurf_DW13','vsurf_DW23','vsurf_EDmin1','vsurf_EDmin2','vsurf_EDmin3','vsurf_EWmin1','vsurf_EWmin2','vsurf_EWmin3','vsurf_G','vsurf_HB1','vsurf_HB2','vsurf_HB3','vsurf_HB4','vsurf_HB5','vsurf_HB6','vsurf_HB7','vsurf_HB8','vsurf_HL1','vsurf_HL2','vsurf_ID1','vsurf_ID2','vsurf_ID3','vsurf_ID4','vsurf_ID5','vsurf_ID6','vsurf_ID7','vsurf_ID8','vsurf_IW1','vsurf_IW2','vsurf_IW3','vsurf_IW4','vsurf_IW5','vsurf_IW6','vsurf_IW7','vsurf_IW8','vsurf_R','vsurf_S','vsurf_V','vsurf_W1','vsurf_W2','vsurf_W3','vsurf_W4','vsurf_W5','vsurf_W6','vsurf_W7','vsurf_W8','vsurf_Wp1','vsurf_Wp2','vsurf_Wp3','vsurf_Wp4','vsurf_Wp5','vsurf_Wp6','vsurf_Wp7','vsurf_Wp8'"

    # PERINTAH SVL:
    # 1. db_Open: Buat MDB baru.
    # 2. db_ImportSD: Import struktur.
    # 3. PartialChargeMDB: Hitung muatan (diperlukan untuk descriptor energi/elektrostatik).
    # 4. QuaSAR_DescriptorMDB: Hitung semua descriptor di atas.
    # 5. exit: Selesai (MDB otomatis tersimpan).
    
    "$MOE_BIN" -exec "db_Open ['$WIN_OUT_MDB', 'create']; db_ImportSD ['$WIN_OUT_MDB', '$WIN_IN', 'mol']; PartialChargeMDB ['$WIN_OUT_MDB', 'FF', 'mol', 'mol_charged']; QuaSAR_DescriptorMDB ['$WIN_OUT_MDB', 'mol_charged', [$DESC_LIST]]; exit [];"

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