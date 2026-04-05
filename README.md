# 📊 Data Science Portfolio

**Frankie Lam** | Data Analyst → Data Scientist  
📍 New York, NY & Bay Area, CA | [LinkedIn](https://www.linkedin.com/in/frankielgk/) | [Email](mailto:frankie_lam@outlook.com)

---

## 👋 About Me

Data professional with **8+ years of experience** as a Data Analyst at **WebMD** and **The New York Times** — two of the most data-rich digital media companies in the world.

I bring a rare combination of deep **business domain expertise** and a growing **machine learning skill set**. After nearly a decade of turning data into decisions for editorial teams, product managers, and executive stakeholders, I am now formalizing my transition into Data Science with a structured curriculum covering Python, statistics, and machine learning.

My background means I don't just build models — I understand the business questions behind them.

---

## 🗂️ Portfolio Structure

```
data-science-portfolio/
├── pandas/          # 10-Day Pandas Bootcamp
├── numpy/           # 5-Day NumPy Bootcamp
├── scipy/           # 14-Day SciPy Statistics Bootcamp
└── scikit-learn/    # 30-Day scikit-learn Bootcamp (in progress)
```

---

## 🏥 Featured Project — Digital Health Patient Conversion Model

**`scipy/SciPy_W1_D5_Correlation_and_Regression.ipynb`** (Q6 Portfolio Extension)

> *In digital health platforms, can we predict whether a user will convert to a diagnosis-seeking (DX) or prescription-seeking (RX) action based on their content consumption behaviour?*

This is a **fully synthetic, independently designed project** built to explore a common and well-documented business problem in the digital health industry. All data was simulated from scratch using publicly known statistical properties of health content platforms — no proprietary or employer data of any kind was used.

**What I built:**
- Designed and simulated a realistic 2,000-user session dataset from first principles using domain-informed distributions (Poisson, Beta, Log-normal, Binomial) based on publicly available research on health content engagement
- Engineered features including content type mix, session depth, device type, return visitor status, and time-on-page
- Built a logistic regression model using `statsmodels` with full odds ratio interpretation
- Evaluated model performance using confusion matrix, precision/recall, F1-score, and AUC-ROC

**Key finding:** Drug information article consumption (OR = 8.2) and session depth (OR = 1.08 per page) are the strongest predictors of conversion — consistent with published research on health information-seeking behaviour.

**Coming in scikit-learn Days 22-24:**
- Proper train/test split and k-fold cross-validation
- Full AUC-ROC and precision-recall curves
- Comparison against Random Forest and XGBoost
- Hyperparameter tuning with GridSearchCV

**Skills demonstrated:** `scipy.stats` · `statsmodels` · `numpy` · `pandas` · logistic regression · odds ratios · synthetic data simulation · domain-driven feature engineering

---

## 📚 Completed Bootcamps

### 🐼 Pandas — 10-Day Bootcamp
**`pandas/`**

End-to-end pandas mastery using real-world datasets (Titanic, restaurant tips, salaries, country demographics).

| Day | Topic | Dataset |
|-----|-------|---------|
| 1 | Loading, inspecting, filtering | Titanic |
| 2 | Data cleaning, new columns, np.select() | Restaurant tips |
| 3 | Sorting, ranking, nlargest | Country demographics |
| 4 | GroupBy, aggregation, HAVING equivalent, transform() | SF Salaries |
| 5 | Joins, merges, UNION ALL | Titanic + class lookup |
| 6 | Reshaping, pivot_table, melt, crosstab | Demographics |
| 7 | Multi-table joins, pivot, string ops | Restaurant + time lookup |
| 8 | Window functions, cumsum, shift, rank, YoY growth | SF Salaries |
| 9 | Datetime parsing, dt accessor, time series, messy dates | Campaign data |
| 10 | Capstone — full pipeline, method chaining, export | All datasets |

**Skills demonstrated:** `pd.merge` · `groupby` · `pivot_table` · `melt` · `transform` · `shift` · `cumsum` · `pd.cut` · method chaining · `.assign()` · `.pipe()`

---

### 🔢 NumPy — 5-Day Bootcamp
**`numpy/`**

NumPy foundations through to the scikit-learn bridge, covering the mathematical layer beneath pandas.

| Day | Topic |
|-----|-------|
| 1 | Array creation, attributes, indexing, boolean masking |
| 2 | Reshaping, stacking, copy vs view |
| 3 | Math, aggregations, broadcasting, np.where() |
| 4 | Linear algebra, np.linalg, random numbers, reproducibility |
| 5 | NumPy ↔ pandas bridge, X/y pattern, StandardScaler manual, np.isnan() |

**Skills demonstrated:** broadcasting · vectorized operations · `np.dot` · `np.linalg` · `np.random` · array manipulation · scikit-learn data preparation

---

### 📈 SciPy — 14-Day Statistics Bootcamp
**`scipy/`**

Applied statistics with a focus on real business problems in digital health and media. All examples use publicly available datasets or synthetic data designed to reflect realistic industry scenarios.

**Week 1 — Theory & Application**

| Day | Topic |
|-----|-------|
| 1 | Probability distributions (Normal, Binomial, Poisson, Beta, Log-normal) |
| 2 | Sampling, Central Limit Theorem, confidence intervals, bootstrapping |
| 3 | Hypothesis testing, p-values, t-tests, A/B testing, Type I/II errors |
| 4 | ANOVA, post-hoc tests, ANCOVA as regression, unified linear model |
| 5 | Correlation, linear & multiple regression, logistic regression ⭐ |
| 6 | Time series — decomposition, stationarity, autocorrelation, seasonality |
| 7 | Monte Carlo simulation, uncertainty quantification, power analysis |

**Week 2 — Practice on Real Data**

| Day | Topic | Dataset |
|-----|-------|---------|
| 1 | Distributions practice | Heart Disease UCI (Kaggle) |
| 2 | Sampling & CIs | News Category Dataset (Kaggle) |
| 3 | A/B testing full pipeline | A/B Testing Dataset (Kaggle) |
| 4 | ANOVA & ANCOVA | Student Performance (Kaggle) |
| 5 | Regression end-to-end | News articles dataset |
| 6 | Time series — traffic & seasonality | Web Traffic (Kaggle) |
| 7 | Monte Carlo capstone ⭐ | Simulated digital health & media business scenarios |

**Skills demonstrated:** `scipy.stats` · `statsmodels` · hypothesis testing · A/B testing · ANOVA · ANCOVA · logistic regression · time series decomposition · Monte Carlo simulation · bootstrapping

---

### 🤖 scikit-learn — 30-Day Bootcamp
**`scikit-learn/`** *(In Progress)*

| Week | Focus |
|------|-------|
| 1 | sklearn API, train/test split, cross-validation, linear & logistic regression |
| 2 | Decision trees, random forests, SVMs, gradient boosting |
| 3 | Clustering, PCA, feature engineering, pipelines |
| 4 | End-to-end projects — digital health conversion model, media engagement predictor |

---

## 🛠️ Technical Skills

| Category | Tools |
|----------|-------|
| **Languages** | Python, SQL (7+ years) |
| **Data manipulation** | pandas, NumPy |
| **Statistics** | SciPy, statsmodels |
| **Machine Learning** | scikit-learn *(in progress)* |
| **Workflow** | VS Code, Jupyter Notebooks, GitHub |
| **Coming soon** | dbt, XGBoost, LightGBM |

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

---

*Last updated: April 2026 | Currently completing SciPy bootcamp — scikit-learn starts next*
