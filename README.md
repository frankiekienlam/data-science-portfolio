# 📊 Data Science Portfolio

**Frankie Lam** | Data Analyst → Data Scientist  
📍 New York, NY | [LinkedIn](https://www.linkedin.com/in/frankielgk/) | [Email](mailto:frankie_lam@outlook.com)

---

> ⚠️ **Important Disclaimer**
>
> All datasets, scenarios, company names, metrics, and business contexts used in this portfolio are **entirely synthetic and independently constructed**. They were simulated from scratch using publicly available research, open-source datasets (Kaggle, UCI, CMS), and domain knowledge derived from publicly documented industry practices.
>
> **No proprietary data, confidential business information, internal systems, unreleased metrics, or trade secrets from any current or former employer — including WebMD or The New York Times — were used, referenced, or reproduced in any form.**
>
> Any resemblance to real internal data is purely coincidental. All work here is my own, built entirely outside of any employment context, and does not represent the views, methods, or intellectual property of any organization I have worked for.

---

## 👋 About Me

Data professional with **8+ years of experience** as a Data Analyst at **WebMD** and **The New York Times** — two of the most data-rich digital media companies in the world.

I bring a rare combination of deep **business domain expertise** and a growing **machine learning skill set**. After nearly a decade of turning data into decisions for editorial teams, product managers, and executive stakeholders, I am now formalizing my transition into Data Science with a structured curriculum covering Python, statistics, and machine learning.

My background means I don't just build models — I understand the business questions behind them.

---

## 🗂️ Portfolio Structure

```
data-science-portfolio/
├── pandas/          # 10-Day Pandas Bootcamp ✅
├── numpy/           # 5-Day NumPy Bootcamp ✅
├── scipy/           # 14-Day SciPy Statistics Bootcamp ✅
├── scikit-learn/    # 30-Day scikit-learn Bootcamp ✅ Complete
└── production/      # Production ML Pipeline — Real-World Datasets
```

---

## 🚀 Production Pipeline — Real-World Datasets

> A production-grade binary classification pipeline built from scratch, tested across four real-world datasets of increasing complexity.

**`classification_LGBM_pipeline.py`** — a fully reusable, dataset-agnostic ML pipeline featuring:

- **PyOD Outlier Ensemble** — IForest + KNN/HBOS + LOF with auto-bypass at 500K+ rows
- **MCAR Diagnostic** — KS-test-based missingness detection with automatic `_IS_MISSING` indicator flags
- **SMOTE / SMOTENC Auto-Selection** — detects categorical features and selects correct oversampling strategy
- **Optuna Hyperparameter Tuning** — Bayesian search with dynamic `min_child_samples`, configurable metric (`auprc`, `roc_auc`, `recall`)
- **Auto-Calibration** — isotonic / sigmoid / none selected based on class ratio
- **TunedThresholdClassifierCV** — business-constraint recall floor with precision floor guard
- **SHAP Explainability** — beeswarm, bar, waterfall, dependency plots with human-readable feature names
- **Pipeline Telemetry** — per-step wall time, RAM delta, peak memory via `psutil`

### Real-World Dataset Results

| Dataset | Rows | ROC-AUC | Recall | Precision | ECE | Key Signal |
|---------|------|---------|--------|-----------|-----|------------|
| **Telco Churn** | 7K | 0.869 | 59% | 66% | 0.041 ✅ | Contract × Tenure |
| **Credit Card Fraud** | 284K | 0.966 | 88% | 29% | — | V14 × V4 |
| **Lending Club Default** | 1.3M | 0.766 | 19% | 79% | 0.026 ✅ | Debt settlement × Sub-grade |
| **CMS GLP-1 Market Intel** | 30M | — | — | — | — | Market intelligence analysis |

---

## 🏥 Featured Project — Digital Health Patient Conversion Model

> *Can we predict whether a user will convert to a diagnosis-seeking (DX) or prescription-seeking (RX) action based on their content consumption behaviour?*

First built in the SciPy bootcamp using `statsmodels` logistic regression, then extended across the scikit-learn arc:

- **D15** — Rebuilt with sklearn Pipeline, calibrated probabilities, revenue threshold curve
- **D29** — Upgraded to XGBoost + SHAP, head-to-head vs baseline, optimal outreach threshold
- **D30** — Capstone: joint MedPulse × HealthLine engagement predictor, full production pipeline with nested CV, calibration audit, and PR curve threshold tuning

All data is fully synthetic. See disclaimer above.

---

## 📚 Completed Bootcamps

### 🐼 Pandas — 10-Day Bootcamp ✅
Data wrangling mastery: groupby, pivot_table, window functions, joins, reshaping, method chaining. Applied to Titanic, SF Salaries, restaurant tips, and country demographics datasets.

---

### 🔢 NumPy — 5-Day Bootcamp ✅
Array operations, broadcasting, linear algebra, and the NumPy ↔ pandas ↔ scikit-learn data pipeline. Foundation for the ML work that followed.

---

### 📈 SciPy — 14-Day Statistics Bootcamp ✅
Applied statistics for digital health and media business problems.

**Week 1 — Theory:** probability distributions · CLT & confidence intervals · A/B testing & hypothesis testing (5-part series) · ANOVA/ANCOVA · regression · time series · Monte Carlo simulation

**Week 2 — Practice on real data:** Heart Disease UCI · News Category · A/B Testing · Student Performance · Web Traffic datasets + Monte Carlo capstone ⭐

**Skills:** `scipy.stats` · `statsmodels` · `pingouin` · bootstrapping · permutation tests · FDR correction · SHAP-free feature analysis · simulation

---

### 🤖 scikit-learn — 30-Day Bootcamp ✅ Complete
Applied ML using synthetic **MedPulse** (digital health) and **HealthLine** (media subscriptions) datasets throughout.

**Week 1 — Core Algorithms ✅ (Days 1–14)**

| Day | Topic |
|-----|-------|
| 1 | sklearn API & ML workflow — Pipeline, leakage detection, CV |
| 2 | Linear & Logistic Regression — Ridge, Lasso, odds ratios |
| 3 | Classification Metrics — confusion matrix, ROC, PR curve, threshold tuning |
| 4 | Cross-Validation & Regression Metrics — KFold, StratifiedKFold, RMSE/MAE/R² |
| 5 | Decision Trees — tree structure, depth sweep, Gini vs permutation importance |
| 6 | Random Forests — OOB score, stability analysis, feature elimination |
| 7 | SVMs & KNN — kernel functions, C sweep, k sweep, log-transform fix |
| 8 | Gradient Boosting — loss curves, staged predictions, lr × n_estimators interaction |
| 9 | XGBoost, LightGBM & SHAP — early stopping, SHAP waterfall/beeswarm, Shapley properties |
| 10 | Preprocessing & Feature Engineering — scaling, encoding, imputation, cyclical features |
| 11 | Pipelines — ColumnTransformer, leakage proof, joblib persistence |
| 12 | Hyperparameter Tuning — GridSearch, RandomizedSearch, validation curves, heatmap |
| 13 | Imbalanced Classes — SMOTE, class_weight, threshold tuning, clinical scenario framing |
| 14 | Model Selection & Bias-Variance — polynomial demo, learning curves, 5-model comparison |

**Week 2 — Applied Projects ✅ (Days 15–30)**

Real datasets (IBM HR Attrition, Ames House Prices, BBC News, Credit Card Fraud) + synthetic MedPulse/HealthLine projects culminating in the D29 DX/RX final model and D30 capstone.

| Day | Topic |
|-----|-------|
| 15 | Digital Health Conversion Model — sklearn Pipeline, calibrated LR, revenue curve |
| 18 | Imbalanced Classes — SMOTE, SMOTENC, ImbPipeline, threshold optimisation, cost-sensitive framing |
| 19 | Full Preprocessing Pipeline — ColumnTransformer, custom transformers, KNN vs median imputation |
| 20 | Cross-Validation Deep Dive — GroupKFold, TimeSeriesSplit, nested CV, user-level leakage detection |
| 21 | Model Comparison — 5-model CV, ROC/PR curves, SMOTENC for mixed-type data |
| 22 | Regression — Ridge/Lasso/GBR on Ames house prices, residual analysis, QQ plots |
| 26 | Stroke Prediction — open practice, full pipeline, SHAP on clinical data |
| 27 | News Popularity — high-cardinality features, regression framing, feature selection |
| 28 | Patient Readmission — clinical imbalance, GroupKFold by patient, recall-constrained scoring |
| 29 | DX/RX Final Model ⭐ — XGBoost + SHAP, nested CV, calibration curves, revenue impact |
| 30 | Capstone: Engagement Predictor ⭐ — nested CV, 3-model comparison, waterfall, PR threshold tuning |

**W1 Skills:** `Pipeline` · `ColumnTransformer` · `GridSearchCV` · `XGBClassifier` · `LGBMClassifier` · SHAP · SMOTE · `cross_validate` · `learning_curve` · `validation_curve` · `joblib`

**W2 Skills:** `SMOTENC` · `ImbPipeline` · `GroupKFold` · `TimeSeriesSplit` · nested CV · `CalibratedClassifierCV` · `TunedThresholdClassifierCV` · `average_precision_score` · `brier_score_loss` · SHAP waterfall/dependency · residual diagnostics · QQ plots · revenue simulation

---

## 🛠️ Technical Skills

| Category | Tools |
|----------|-------|
| **Languages** | Python, SQL (7+ years) |
| **Data manipulation** | pandas, NumPy |
| **Statistics** | SciPy, statsmodels, pingouin |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, SHAP |
| **Imbalanced Learning** | imbalanced-learn, SMOTE, SMOTENC, PyOD |
| **Hyperparameter Tuning** | Optuna (Bayesian), GridSearchCV, RandomizedSearchCV |
| **Workflow** | VS Code, Jupyter Notebooks, Google Colab, GitHub |
| **Cloud** | Google Drive + Colab pipeline (CPU + T4 GPU) |

---

## 💼 Professional Background

**Digital Health Industry** *(Current)*  
Data Analyst — health content performance, user engagement analytics, A/B testing, audience segmentation

**Digital Media Industry**  
Data Analyst — subscription funnel analysis, content recommendation signals, reader behaviour modelling

8 years of translating complex data into decisions for non-technical stakeholders across product, editorial, and executive teams.

---

## 📬 Get in Touch

I am actively building toward a Data Scientist role.  
If you are working on interesting problems in health, media, or consumer technology, I would love to connect.

- **LinkedIn:** [linkedin.com/in/frankielgk/](https://www.linkedin.com/in/frankielgk/)
- **Email:** frankie_lam@outlook.com
- **Kaggle:** [kaggle.com/frankielgk](https://www.kaggle.com/frankielgk)

---

*Last updated: August 2026 | pandas ✅ · NumPy ✅ · SciPy ✅ · scikit-learn ✅ (30/30) · Production Pipeline ✅*
