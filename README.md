# Eksperimen_SML_AnggaMaulana

Repositori eksperimen preprocessing untuk submission Kelas Membangun Sistem Machine Learning (Dicoding).

## Dataset
**Mall Customer Segmentation** — data pelanggan mall berisi usia, pendapatan tahunan, dan spending score.  
Sumber: [Kaggle - vjchoudhary7](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Struktur Folder
```
Eksperimen_SML_AnggaMaulana/
├── README.md
├── .gitignore
├── mall_customers_raw/
│   └── Mall_Customers.csv
└── preprocessing/
    ├── Eksperimen_AnggaMaulana.ipynb
    ├── automate_AnggaMaulana.py
    └── mall_customers_preprocessing/
        └── mall_customers_preprocessing.csv
```

## Pipeline Preprocessing
1. Load Data
2. Seleksi Fitur
3. Handle Missing Values
4. Feature Scaling (StandardScaler)
5. Simpan Output