# 📊 Data Science Portfolio

**Frankie Lam** | Data Analyst → Data Scientist  
📍 New York, NY & Bay Area, CA | [LinkedIn](https://www.linkedin.com/in/frankielgk/) | [Email](mailto:frankie_lam@outlook.com)

---

> ⚠️ **Important Disclaimer**
>
> All datasets, scenarios, company names, metrics, and business contexts used in this portfolio are **entirely synthetic and independently constructed**. They were simulated from scratch using publicly available research, open-source datasets (Kaggle, UCI), and domain knowledge derived from publicly documented industry practices.
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

End-to-end pandas mastery using real-world public datasets (Titanic, restaurant tips, SF salaries, country demographics).

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

Applied statistics with a focus on real business problems in digital health and media. All examples use publicly available datasets or fully synthetic data designed to reflect realistic industry scenarios. No employer data of any kind was used — see disclaimer at the top of this page.

---

#### Week 1 — Theory & Application

**Day 1 — Probability Distributions**
`SciPy_W1_D1_Probability_Distributions.ipynb`

Covers the full toolkit of distributions used in A/B testing and data science modelling.

| Topic | Key Functions |
|-------|--------------|
| Normal distribution — CLT foundation | `stats.norm` |
| Binomial — conversion rate modelling | `stats.binom` |
| Poisson — event rate modelling | `stats.poisson` |
| Beta — Bayesian priors for proportions | `stats.beta` |
| Log-normal — session duration, revenue | `stats.lognorm` |

**Skills demonstrated:** `scipy.stats` · PDF/CDF/PPF · distribution fitting · visualization of distribution shape

---

**Day 2 — Sampling, CLT & Confidence Intervals**
`SciPy_W1_D2_Sampling_CLT_and_Confidence_Intervals.ipynb`

| Topic | Key Concepts |
|-------|-------------|
| Sampling distributions | Why sample means follow Normal regardless of raw data |
| Central Limit Theorem simulation | Visualising convergence with increasing n |
| Confidence intervals | `proportion_confint` · `norm.interval` |
| Bootstrapping | Non-parametric CI estimation via resampling |

**Skills demonstrated:** `scipy.stats` · `statsmodels` · bootstrapping · CI construction · sampling simulation

---

**Day 3 — Hypothesis Testing, A/B Testing & t-tests**
`SciPy_W1_D3A_Foundations_and_The_Two-Proportion_Test.ipynb`
`SciPy_W1_D3B_Continuous_Metrics_and_t-tests.ipynb`

Split across two notebooks covering binary and continuous metrics end-to-end.

*Day 3A — Two-Proportion Test & p-value Foundations*

| Question | Topic |
|----------|-------|
| Q1 | Full two-proportion z-test pipeline — conversion rates, 95% CI error bars, z-stat, p-value, lift |
| Q2 | Cohen's h effect size — classification, dot plot, statistical vs practical significance |
| Q3 | p-value deep dive — 2,000-trial simulation under H0 and H1, power overlay, false positive rate |
| Q4 | Absolute vs relative lift — grouped bar chart, stakeholder communication (executives vs engineers) |
| Q5 | Revenue impact & sensitivity analysis — monthly/annual projections, $1M threshold, lift × price heatmap |

*Day 3B — Continuous Metrics & t-tests*

| Question | Topic |
|----------|-------|
| Q1 | Welch's t-test on lognormal session duration — histogram overlay, Cohen's d, Welch's vs Student's justification |
| Q2 | Variance kills power — low vs high variance scenarios, empirical power simulation (vectorised 1,000-trial), sample size calculation |
| Q3 | Paired t-test — before/after scatter plot, difference distribution, paired vs independent p-value comparison |
| Q4 | CUPED variance reduction — theta calculation, adjusted metrics, 40% variance reduction, p-value improvement |
| Q5 | Test selection challenge — four datasets requiring different tests: proportion z-test, Mann-Whitney U, paired t-test, Welch's with winsorization |

**Key concepts covered:** Type I / Type II errors · statistical power · effect size · one-tailed vs two-tailed tests · non-parametric tests · CUPED · variance reduction

**Skills demonstrated:** `proportions_ztest` · `ttest_ind` · `ttest_rel` · `mannwhitneyu` · `proportion_effectsize` · `TTestIndPower` · vectorised simulation · winsorization

---

**Day 4 — ANOVA, Post-hoc Tests & ANCOVA**
`SciPy_W1_D4_ANOVA_and_ANCOVA.ipynb`

| Topic | Key Concepts |
|-------|-------------|
| One-way ANOVA | F-statistic, p-value, when to use vs t-test |
| Post-hoc testing | Tukey HSD — pairwise comparisons after significant ANOVA |
| Two-way ANOVA | Interaction effects between two factors |
| ANCOVA as regression | Controlling for a covariate — the unified linear model view |

**Skills demonstrated:** `scipy.stats.f_oneway` · `statsmodels.stats.multicomp` · `smf.ols` · interaction terms · covariate adjustment

---

**Day 5 — Correlation, Linear & Logistic Regression**
`SciPy_W1_D5_Correlation_and_Regression.ipynb`

| Topic | Key Concepts |
|-------|-------------|
| Pearson & Spearman correlation | Assumptions, when each applies |
| Simple linear regression | OLS, R², residual analysis |
| Multiple regression | Feature selection, multicollinearity, VIF |
| Logistic regression | Log-odds, odds ratios, classification threshold |
| ⭐ Portfolio project extension (Q6) | Digital health conversion model — see Featured Project above |

**Skills demonstrated:** `scipy.stats.pearsonr` · `scipy.stats.spearmanr` · `statsmodels.formula.api` · `smf.ols` · `smf.logit` · odds ratios · confusion matrix · AUC-ROC

---

**Day 6 — Time Series Analysis**
`SciPy_W1_D6_Time_Series.ipynb`

| Topic | Key Concepts |
|-------|-------------|
| Decomposition | Trend, seasonality, residual — additive vs multiplicative |
| Stationarity | ADF test, differencing |
| Autocorrelation | ACF and PACF plots |
| Seasonality detection | Period identification, seasonal adjustment |

**Skills demonstrated:** `statsmodels.tsa` · `seasonal_decompose` · `adfuller` · ACF/PACF · time series visualisation

---

**Day 7 — Monte Carlo Simulation & Power Analysis**
`SciPy_W1_D7_Monte_Carlo_and_Power_Analysis.ipynb`

| Topic | Key Concepts |
|-------|-------------|
| Monte Carlo fundamentals | Uncertainty quantification via repeated simulation |
| Revenue scenario modelling | Distribution of outcomes across 10,000 simulations |
| Power analysis | Sample size planning before an experiment |
| Type I / Type II error trade-offs | α, β, and the cost of underpowered tests |

**Skills demonstrated:** `np.random` · simulation loops · `NormalIndPower` · `solve_power` · confidence intervals on simulation outputs

---

#### Week 2 — Practice on Real Data

| Day | Topic | Dataset |
|-----|-------|---------|
| W2D1 | Distributions practice | Heart Disease UCI (Kaggle) |
| W2D2 | Sampling & confidence intervals | News Category Dataset (Kaggle) |
| W2D3 | A/B testing full pipeline | A/B Testing Dataset (Kaggle) |
| W2D4 | ANOVA & ANCOVA | Student Performance (Kaggle) |
| W2D5 | Regression end-to-end | News articles dataset |
| W2D6 | Time series — traffic & seasonality | Web Traffic Dataset (Kaggle) |
| W2D7 | Monte Carlo capstone ⭐ | Simulated digital health & media business scenarios |

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
