# Data

The data files are **not included** in this repository: the Kaggle competition rules do not allow redistributing them.

## How to get them

1. Go to the [Rossmann Store Sales – Data](https://www.kaggle.com/c/rossmann-store-sales/data) page. You need a Kaggle account and must accept the competition rules.
2. Download `train.csv`, `test.csv` and `store.csv`.
3. Save them in this folder with these names:

```
data/
├── train_salespredictions.csv
├── test_salespredictions.csv
└── store_salespredictions.csv
```

## Generated files

Running the notebook creates a `clean/` subfolder with the cleaned datasets:

```
data/clean/
├── train_clean.parquet   # 844,338 rows: open days with sales > 0, plus engineered features
└── test_clean.parquet    # 41,088 rows: all test days, plus engineered features
```
