"""
automate_AnggaMaulana.py
========================
Script otomatisasi preprocessing untuk dataset Mall Customer Segmentation.
Tahapan sama dengan notebook eksperimen.

Kriteria 1 - Skilled | Kelas Membangun Sistem Machine Learning

Penggunaan:
    python automate_AnggaMaulana.py
    python automate_AnggaMaulana.py --input path/ke/data.csv --output path/output/
"""

import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_PATH  = "../mall_customers_raw/Mall_Customers.csv"
OUTPUT_DIR  = "./mall_customers_preprocessing"
FEATURES    = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


# ── 1. Load Data ──────────────────────────────────────────────────────
def load_data(input_path: str) -> pd.DataFrame:
    """Memuat dataset dari path yang diberikan."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[load_data]       Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ── 2. Seleksi Fitur ──────────────────────────────────────────────────
def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Memilih kolom fitur yang relevan untuk clustering."""
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom tidak ditemukan: {missing_cols}")
    df_selected = df[features].copy()
    print(f"[select_features] Fitur dipilih: {features}")
    return df_selected


# ── 3. Handle Missing Values ──────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris yang mengandung missing values."""
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    removed = before - after
    if removed > 0:
        print(f"[handle_missing]  {removed} baris dihapus karena missing values")
    else:
        print(f"[handle_missing]  Tidak ada missing values -- {after} baris dipertahankan")
    return df_clean


# ── 4. Handle Duplikasi ───────────────────────────────────────────────
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris yang duplikat."""
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    removed = before - after
    if removed > 0:
        print(f"[handle_duplicates] {removed} baris duplikat dihapus")
    else:
        print(f"[handle_duplicates] Tidak ada duplikasi -- {after} baris dipertahankan")
    return df_clean


# ── 5. Handle Outlier ─────────────────────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Menghapus outlier menggunakan metode IQR."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        print(f"[remove_outliers]  Kolom '{col}': {len(outliers)} outlier ditemukan")

        df_clean = df_clean[
            (df_clean[col] >= lower_bound) &
            (df_clean[col] <= upper_bound)
        ]

    print(f"[remove_outliers]  Shape setelah hapus outlier: {df_clean.shape}")
    return df_clean


# ── 6. Feature Scaling ────────────────────────────────────────────────
def scale_features(df: pd.DataFrame) -> tuple:
    """Melakukan standardisasi fitur menggunakan StandardScaler."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    print(f"[scale_features]  StandardScaler diterapkan -- mean=0, std=1")
    return df_scaled, scaler


# ── 7. Simpan Hasil ───────────────────────────────────────────────────
def save_preprocessed(df: pd.DataFrame, output_dir: str) -> str:
    """Menyimpan data hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mall_customers_preprocessing.csv")
    df.to_csv(output_path, index=False)
    print(f"[save]            Data tersimpan: {output_path}")
    return output_path


# ── Pipeline Utama ────────────────────────────────────────────────────
def run_preprocessing(input_path=INPUT_PATH, output_dir=OUTPUT_DIR, features=None):
    """Menjalankan seluruh pipeline preprocessing secara berurutan."""
    if features is None:
        features = FEATURES

    print("=" * 55)
    print("  PIPELINE PREPROCESSING -- Mall Customer Segmentation")
    print("=" * 55)

    df = load_data(input_path)
    df = select_features(df, features)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = remove_outliers_iqr(df, features)
    df_scaled, scaler = scale_features(df)
    save_preprocessed(df_scaled, output_dir)

    print("=" * 55)
    print(f"  Preprocessing selesai! Shape: {df_scaled.shape}")
    print("=" * 55)

    return df_scaled


# ── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing -- Mall Customer Segmentation"
    )
    parser.add_argument("--input", type=str, default=INPUT_PATH,
                        help="Path ke file CSV dataset mentah")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Folder tujuan output")
    args = parser.parse_args()

    result = run_preprocessing(input_path=args.input, output_dir=args.output)
    print(f"\nPreview 5 baris pertama:")
    print(result.head().to_string())