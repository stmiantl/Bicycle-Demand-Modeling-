# Bicycle-Demand-Modeling-
This repo is based on a demo dataset depicting supervised learning modeling.
# Bike Demand Interpolation (Hourly) — Supervised ML Pipeline

This repository contains a complete, end-to-end supervised learning workflow for **interpolating missing hourly bike demand** values. It covers data cleaning, EDA, feature engineering, **target transformation** (Box–Cox), **feature selection** (permutation importance), **model selection** (Linear Regression, Random Forest, XGBoost), **hyperparameter optimization**, and evaluation.

> **Use case:** Interpolation (filling missing historical targets), **not** forecasting. Because we only recover values inside the observed period, both earlier and later timestamps can inform the model (while still avoiding target leakage).

---

## Dataset

* **Source (as referenced in the notebook):** Kaggle — *Bicycle Demand Machine Learning Dataset*
  **Link:** [https://www.kaggle.com/datasets/kagglemaster95/bicycle-demand-machine-learning-dataset/data](https://www.kaggle.com/datasets/kagglemaster95/bicycle-demand-machine-learning-dataset/data)

* **Files referenced in the notebook:**

  * `training_data.csv` — includes targets `casual`, `registered`, `count`
  * `test_data.csv` — same schema **without** targets (used to simulate inference)


---

## Repository Structure

```
.
├─ README.md
└─ modeling_notebook.ipynb
└─ requirements.txt

```

---

## Environment & Setup

### Local (Python 3.9+)

```bash
python -m venv .venv
source .venv/bin/activate         # (Windows) .venv\Scripts\activate
pip install -U numpy pandas scipy scikit-learn xgboost matplotlib
```

### Run the notebook

Open `notebooks/modeling_notebook.ipynb` and run cells in order:

1. **EDA & Cleaning**: drop index artifact, cast `datetime`, derive `hour`/`month`, one-hot encode `season`, `holiday`, `workingday`, `weather`, `hr`, `month`.
2. **Target Transform**: `boxcox1p` (λ=0.4) for `casual`, `registered`, `count` → `*_t`.
3. **Baselines**: OLS, RandomForestRegressor, XGBRegressor (5-fold CV).
4. **Feature Selection**: Permutation importance (MSE-based), keep features with positive mean importance per target.
5. **Tuning**: Compact GridSearchCV on RF (`n_estimators`, `max_depth`) for `count_t`, reuse settings for `casual_t`/`registered_t`.
6. **Evaluation**: Train with selected features and report metrics on a carved-out test split.

> **Leakage control:** All scalers/encoders/selection are fit **inside** CV folds or on training splits first, then applied to validation/test.

---

## Modeling Design (Why Interpolation ≠ Forecasting)

* **Interpolation (this project):** Missing values lie **within** the historical window; features at those timestamps are known. **Standard K-Fold** is acceptable (still avoid target leakage).
* **If forecasting:** Enforce **time-ordered** splits (rolling/blocked CV) so the model never “sees” the future during training.

---

## Methods (Brief)

* **Targets:** Box–Cox via `boxcox1p` (λ=0.4) on `casual`, `registered`, `count` → `casual_t`, `registered_t`, `count_t`.
* **Categoricals:** `pd.get_dummies` on `season`, `holiday`, `workingday`, `weather`, `hr`, `month`.
* **Numeric scaling:** Min–Max scaling applied **inside CV**.
* **Models:** OLS, RandomForestRegressor (bagging), XGBRegressor (boosting).
* **Feature selection:** Permutation importance (`scoring='neg_mean_squared_error'`), per target.
* **Hyperparameters (demo):** RF `n_estimators`, `max_depth` tuned via GridSearchCV on `count_t`; settings reused for the other targets.

---

## Results (Carved-Out Split)

* **Chosen baseline:** **RandomForestRegressor** (similar to XGB; stronger than OLS; simpler and interpretable).
* **RMSE (count, original units):** **72** → average hourly error ≈ 72 rentals (root-mean-square sense).
* **R²:** high **0.8s** → strong lift over a mean baseline.
* **Error shape:** Residuals centered near 0; larger errors on **high-demand** hours (likely missing exogenous drivers like events).

> **Caveat:** The carved-out split was involved in selection/tuning, so results are **indicative**. Re-validate on a strictly **held-out, out-of-time** test set once available.

---

## Limitations & Next Steps

* **Feature enrichment:** Add exogenous drivers (events, promotions, transit/traffic anomalies, competitor signals, tourism/footfall).
* **Transforms:** Consider Yeo–Johnson/quantile transforms for skewed predictors; careful causal windowing if moving toward forecasting.
* **Model breadth:** Try Ridge/Lasso/Elastic Net, polynomial regression (with regularization), SVR, LightGBM/CatBoost.
* **Tuning depth:** Extend RF (`min_samples_leaf`, `min_samples_split`, `max_features`, `bootstrap/max_samples`) and broaden XGB tuning.
* **Evaluation:** Re-assess on a **true held-out/out-of-time** set and report RMSE/MAE/R² (original units) plus optional RMSLE.

---

## Why Box–Cox (`boxcox1p`) on Targets?

Targets are right-skewed with high variance and zeros. `boxcox1p(y, λ)` (λ=0.4) stabilizes variance and reduces skew; unlike `log`, it handles **zeros** via `+1`. Even for trees, transforming the **target** can reduce the leverage of extreme values under squared-error loss and lower RMSE after inverse transform.



## Acknowledgments

* Dataset: Kaggle — *Bicycle Demand Machine Learning Dataset*
  [https://www.kaggle.com/datasets/kagglemaster95/bicycle-demand-machine-learning-dataset/data](https://www.kaggle.com/datasets/kagglemaster95/bicycle-demand-machine-learning-dataset/data)

---

*Questions or suggestions? Open an issue or PR.*
