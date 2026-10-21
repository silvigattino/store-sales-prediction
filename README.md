# Store Sales Prediction – Rossmann

Forecasting daily sales for **856 Rossmann drugstores** in Germany, six weeks ahead (August 1 – September 17, 2015), based on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle competition.

## Problem

Rossmann operates thousands of drugstores across Europe. Daily sales depend on promotions, holidays, seasonality, competition and store characteristics. Store managers need reliable forecasts to plan staff schedules, inventory and resources.

**Goal:** build a machine learning model that predicts daily sales per store up to six weeks in advance, minimizing the prediction error.

**Metric:** RMSPE (Root Mean Square Percentage Error). Days with zero sales are not scored.

## Data

The data comes from Kaggle and is **not included in this repository** (see [Getting the data](#getting-the-data)).

| File | Rows | Description |
|---|---|---|
| `train` | 1,017,209 | Daily sales and customers per store, Jan 2013 – Jul 2015 |
| `test` | 41,088 | Days to predict (Aug 1 – Sep 17, 2015), without sales |
| `store` | 1,115 | Store type, assortment, competition and Promo2 information |

## Project structure

```
store-sales-prediction/
├── salespredictions.ipynb   # full workflow: cleaning, EDA, models, forecast
├── data/                    # Kaggle data (not tracked by git)
├── requirements.txt
└── README.md
```

## Workflow

### 1. Data cleaning
- Converted `Date` to datetime and `StateHoliday` to a category.
- Filled the 11 missing `Open` values in test (store 622) with 1.
- Created 4 **closure features** on train + test combined, *before* filtering: `ClosedYesterday`, `ClosedTomorrow`, `DaysSinceClosed`, `DaysToClosed`.
- Merged store information.
- Handled missing values and added competition features: `CompetitionOpenMonths`, `CompetitionActive`, plus missing-data flags.
- Added Promo2 features: `Promo2Active`, `Promo2Month`, `Promo2Weeks`.
- Filtered train to open days with sales > 0 (844,338 rows). Test kept complete; closed days are predicted as 0.

### 2. Exploratory data analysis
| Topic | Finding |
|---|---|
| Sales distribution | Right-skewed (skewness ≈ 1.6). Log transform makes it almost symmetric, so the model predicts `log(Sales)`. |
| Day of week | Monday is the strongest day, Saturday the weakest. Only ~30 stores open on Sundays. |
| Promo | Promos lift sales by **~40%** for a typical store, ranging from +2% to +100% depending on the store. |
| Closures | Sales are ~20% higher the day after a closure. |
| Competition | Distance alone barely matters, but when a new competitor opens, sales drop **5–7%**. |
| Promo2 | The relaunch month does **not** increase sales. |
| Aug–Sep | Every year this period sells ~3% less than June–July (summer holidays), and daily sales follow the promo calendar. |

### 3. Feature selection
23 features covering store identity, calendar, promo, holidays, closures, competition and Promo2. `Customers` was excluded because it is not available for the forecast period (data leakage).

### 4. Models
Validation: the last 6 weeks of train (Jun 19 – Jul 31, 2015), mimicking the real forecast horizon.

| Model | Validation RMSPE |
|---|---|
| Linear Regression (one-hot encoding) | 0.2039 |
| Random Forest | 0.1439 |
| XGBoost | 0.1217 |
| **XGBoost + correction factor (×0.985)** | **0.1200** |

**Hyperparameter tuning:** we changed one parameter at a time (tree depth, column sampling, minimum child weight, learning rate, feature removal). None beat the initial configuration, so it was kept as the final model.

Final XGBoost parameters: `max_depth=10`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.7`, `min_child_weight=3`, 1,250 trees, retrained on the full train set.

### 5. Forecast (Aug – Sep 2015)

Average daily sales per store (open days, Aug 1 – Sep 17):

| | Stores | Jun–Jul | Aug | Sep | Aug–Sep vs Jun–Jul |
|---|---|---|---|---|---|
| 2013 (actual) | 856 | 6,608 | 6,503 | 6,308 | −3.1% |
| 2014 (actual) | 676* | 6,992 | 6,669 | 6,910 | −2.9% |
| **2015 (forecast)** | 856 | 7,096 | **6,767** | **6,743** | **−4.8%** |

\* *180 stores have no data for the second half of 2014.*

## Conclusions

- The forecast is consistent with past behavior: it captures the summer slowdown and the downward trend already visible in July 2015.
- It may be **slightly pessimistic**: the predicted drop versus June–July (−4.8%) is larger than in 2013 and 2014.
- The biggest drivers of sales are **store identity** and the **daily promo**; parameter tuning added little, so future gains should come from better features, such as store-level historical averages or promo lift per store.

## Getting the data

1. Download `train.csv`, `test.csv` and `store.csv` from the [Kaggle competition page](https://www.kaggle.com/c/rossmann-store-sales/data). You need a Kaggle account and must accept the competition rules.
2. Save them in the `data/` folder as `train_salespredictions.csv`, `test_salespredictions.csv` and `store_salespredictions.csv`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook salespredictions.ipynb
```

Update `base_path` in the data loading cell to point to your local `data/` folder.

## Tech stack

Python · pandas · NumPy · Matplotlib · scikit-learn · XGBoost
