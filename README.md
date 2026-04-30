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
├── pandas/          # 10-Day Pandas Bootcamp ✅
├── numpy/           # 5-Day NumPy Bootcamp ✅
├── scipy/           # 14-Day SciPy Statistics Bootcamp (Week 1 complete ✅)
└── scikit-learn/    # 30-Day scikit-learn Bootcamp (starting soon)
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

**Key findings:** Drug information article consumption (OR = 10.1) and symptom article consumption (OR = 5.0) are the strongest predictors of conversion — consistent with published research on health information-seeking behaviour.

**Coming in scikit-learn Days 22–24:**
- Proper train/test split and k-fold cross-validation
- Full AUC-ROC and precision-recall curves
- Comparison against Random Forest and XGBoost
- Hyperparameter tuning with GridSearchCV

**Skills demonstrated:** `scipy.stats` · `statsmodels` · `numpy` · `pandas` · logistic regression · odds ratios · synthetic data simulation · domain-driven feature engineering

---

## 📚 Completed Bootcamps

### 🐼 Pandas — 10-Day Bootcamp ✅
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

### 🔢 NumPy — 5-Day Bootcamp ✅
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

#### ✅ Week 1 — Theory & Application (Complete)

**Day 1 — Probability Distributions**
`SciPy_W1_D1_Probability_Foundations_and_Distributions.ipynb`

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

**Day 3 — Hypothesis Testing, A/B Testing & Statistical Tests**

Split across five notebooks covering the full spectrum of hypothesis testing from binary proportions to non-parametric methods and advanced pitfalls.

*Day 3A — Two-Proportion Test & p-value Foundations*
`SciPy_W1_D3A_Foundations_and_The_Two-Proportion_Test.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Full two-proportion z-test pipeline — conversion rates, 95% CI error bars, z-stat, p-value, lift |
| Q2 | Cohen's h effect size — classification, dot plot, statistical vs practical significance |
| Q3 | p-value deep dive — 2,000-trial simulation under H0 and H1, power overlay, false positive rate |
| Q4 | Absolute vs relative lift — grouped bar chart, stakeholder communication |
| Q5 | Revenue impact & sensitivity analysis — monthly/annual projections, $1M threshold, lift × price heatmap |

*Day 3B — Continuous Metrics & t-tests*
`SciPy_W1_D3B_Continuous_Metrics_and_t-tests.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Welch's t-test on lognormal session duration — histogram overlay, Cohen's d, Welch's vs Student's justification |
| Q2 | Variance kills power — low vs high variance scenarios, vectorised 1,000-trial simulation, sample size calculation |
| Q3 | Paired t-test — before/after scatter plot, difference distribution, paired vs independent comparison |
| Q4 | CUPED variance reduction — theta calculation, adjusted metrics, 40% variance reduction |
| Q5 | Test selection challenge — proportion z-test, Mann-Whitney U, paired t-test, Welch's with winsorization |

*Day 3C — Sample Size, Power & Experiment Design*
`SciPy_W1_D3C_Sample_Size__Power_and_Experiment_Design.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Sample size calculator — proportions and continuous metrics, power curves |
| Q2 | MDE sensitivity — n vs MDE table, annotated chart, PM interpretation |
| Q3 | The peeking problem — 1,000-experiment simulation, cumulative false positive rate, business impact |
| Q4 | Runtime estimation — weekly traffic variation, cumulative sample size plot, minimum 7-day rule |
| Q5 | Full experiment design brief — hypotheses, metrics, sample size, runtime, invalidation risks |

*Day 3D — Non-Parametric Tests*
`SciPy_W1_D3D_Non-Parametric_Tests.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Mann-Whitney U on revenue data — log-scaled boxplot, rank-biserial effect size |
| Q2 | Wilcoxon signed-rank on satisfaction scores — score transition heatmap, paired vs independent comparison |
| Q3 | Permutation test from scratch — 10,000 shuffles, null distribution, two-tailed p-value, axvspan shading |
| Q4 | Dunn's post-hoc bridge — Kruskal-Wallis confirmation, Bonferroni correction, p-value heatmap |
| Q5 | Parametric vs non-parametric decision framework — 5 real-world scenarios |

*Day 3E — Multiple Testing, Advanced Topics & Pitfalls*
`SciPy_W1_D3E_Multiple_Testing__Advanced_Topics_and_Pitfalls.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Multiple testing correction — Bonferroni vs Benjamini-Hochberg FDR, comparison chart |
| Q2 | Simpson's Paradox — synthetic A/B dataset, segment composition stacked bar, plain-English explanation |
| Q3 | Novelty effect detection — 21-day simulation, week-1 vs week-3 t-tests, duration decision |
| Q4 | Bayesian A/B testing — Beta posteriors, P(treatment > control) via Monte Carlo, credible interval shading |
| Q5 | Full A/B testing audit — peeking simulation, SRM chi-squared, multiple testing correction, structured audit report |
| Q6 | Chi-squared test — 3×2 contingency table, Cramér's V effect size, observed vs expected heatmap, multi-segment Bonferroni |

**Key concepts covered:** Type I/II errors · statistical power · effect size · CUPED · variance reduction · non-parametric tests · Bayesian A/B · multiple testing · Simpson's Paradox · chi-squared · peeking · SRM · experiment design

**Skills demonstrated:** `proportions_ztest` · `ttest_ind` · `ttest_rel` · `mannwhitneyu` · `chi2_contingency` · `proportion_effectsize` · `TTestIndPower` · `multipletests` · `stats.beta` · permutation testing · Bonferroni · BH FDR · vectorised simulation · winsorization

---

**Day 4 — ANOVA & Beyond: Parametric and Non-Parametric 3+ Group Tests**

Split across three notebooks progressing from one-way ANOVA to repeated measures and non-parametric alternatives.

*Day 4A — One-Way ANOVA, Post-Hoc Tests & Assumptions*
`SciPy_W1_D4A_ANOVA_PostHoc_Assumptions.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | One-way ANOVA — F-statistic, p-value, group visualization |
| Q2 | Tukey HSD post-hoc — all pairwise comparisons, confidence intervals |
| Q3 | ANOVA assumptions — Shapiro-Wilk per group, Levene's test, violation consequences |
| Q4 | ANCOVA — covariate control, p-value comparison before and after adjustment |
| Q5 | ANCOVA as regression — OLS equivalence proof with `np.isclose()`, GLM explanation |
| Q6 | Two-way ANOVA — interaction effects, Type II table, interaction plot, mobile vs desktop business interpretation |

*Day 4B — ANCOVA, Two-Way ANOVA & Repeated Measures*
`SciPy_W1_D4B_ANCOVA_TwoWay_RepeatedMeasures.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | ANCOVA with correlated covariate — "Fairness Filter" framing, residuals vs fitted diagnostics |
| Q2 | Two-way ANOVA — layout × device interaction, pointplot, mobile/desktop gap quantification |
| Q3 | Repeated measures ANOVA — AnovaRM, pairwise Bonferroni post-hoc, within-subject power explanation |
| Q4 | Unified GLM — ANOVA = ANCOVA = OLS regression demonstrated, coefficient interpretation |
| Q5 | Reusable `assumption_check()` function — Shapiro-Wilk + Levene, Day 4A/4B vs 4C recommendation |

*Day 4C — Non-Parametric Tests for 3+ Groups*
`SciPy_W1_D4C_NonParametric_3Plus_Groups.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Kruskal-Wallis — KDE overlay with median lines, H-statistic, effect size |
| Q2 | Dunn's post-hoc — Bonferroni correction, adjusted p-value heatmap |
| Q3 | Friedman test — pairwise Wilcoxon with Bonferroni, median line chart with IQR bars |
| Q4 | Parametric vs non-parametric side by side — outlier contamination experiment, robustness nuance |
| Q5 | Full decision framework — 4 datasets, assumption checks, test selection, post-hoc, `pingouin` |

**Skills demonstrated:** `f_oneway` · `pairwise_tukeyhsd` · `AnovaRM` · `anova_lm` · `smf.ols` · `stats.kruskal` · `stats.friedmanchisquare` · `posthoc_dunn` · `pingouin` · interaction terms · covariate adjustment · GLM unification

---

**Day 5 — Correlation, Linear & Logistic Regression**
`SciPy_W1_D5_Correlation_and_Regression.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Pearson vs Spearman correlation — full matrix, strongest/weakest identification |
| Q2 | Simple linear regression — slope interpretation, R², predictions |
| Q3 | Multiple regression — OLS with 3 predictors, significance, Adjusted R² |
| Q4 | Regression diagnostics — Shapiro-Wilk on residuals, residuals vs fitted, Cook's Distance |
| Q5 | Logistic regression preview — binary outcome, odds ratios, pseudo R², business interpretation |
| Q6 ⭐ | Portfolio extension — DX/RX patient conversion model (see Featured Project above) |

**Skills demonstrated:** `stats.pearsonr` · `stats.spearmanr` · `stats.linregress` · `smf.ols` · `smf.logit` · odds ratios · Cook's Distance · confusion matrix · AUC-ROC · point-biserial correlation

---

**Day 6 — Time Series Analysis**
`SciPy_W1_D6_Time_Series_Analysis.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Build and decompose — Gaussian flu spike, seasonal_decompose, component magnitude summary |
| Q2 | Stationarity testing — ADF test, first-order differencing, before/after stationarity plot |
| Q3 | Autocorrelation analysis — ACF/PACF on stationary series, lag threshold, forecasting horizon |
| Q4 | Seasonality deep dive — daily series, weekly + annual patterns, dual bar chart |
| Q5 | Rolling statistics and anomaly detection — 7-day rolling band, 2σ threshold, highlighted anomaly plot |

**Skills demonstrated:** `seasonal_decompose` · `adfuller` · `acf` · `pacf` · `plot_acf` · `plot_pacf` · rolling statistics · anomaly detection · time series visualisation

---

**Day 7 — Monte Carlo Simulation**
`SciPy_W1_D7_Monte_Carlo_Simulation.ipynb`

| Question | Topic |
|----------|-------|
| Q1 | Estimate π — 1M point simulation, convergence table at 1k/10k/100k/1M, 1/√n convergence commentary |
| Q2 | Revenue projection — 50,000 simulations, subscriber compounding loop, downside/upside percentiles, distribution histogram |
| Q3 | A/B test power simulation — empirical power at 6 sample sizes using `chi2_contingency`, reusable function |
| Q4 | Bootstrap confidence intervals — mean, median, 90th percentile CIs, tail instability explanation |
| Q5 | Risk simulation — 3D array simulation `(N, days, articles)`, daily views and RPM revenue, bad-day 10th percentile |

**Skills demonstrated:** `np.random` · vectorised 3D simulation · bootstrap resampling · `chi2_contingency` · `stats.beta` · Monte Carlo convergence · revenue modelling · risk quantification · uncertainty quantification

---

#### Week 2 — Practice on Real Data *(starting next)*

| Day | Topic | Dataset |
|-----|-------|---------|
| W2D1 | Distributions practice | Heart Disease UCI (Kaggle) |
| W2D2 | Sampling & confidence intervals | News Category Dataset (Kaggle) |
| W2D3 | A/B testing full pipeline | A/B Testing Dataset (Kaggle) |
| W2D4 | ANOVA & ANCOVA | Student Performance (Kaggle) |
| W2D5 | Regression end-to-end | News articles dataset |
| W2D6 | Time series — traffic & seasonality | Web Traffic Dataset (Kaggle) |
| W2D7 | Monte Carlo capstone ⭐ | Simulated digital health & media business scenarios |

---

### 🤖 scikit-learn — 30-Day Bootcamp *(starting after SciPy Week 2)*
**`scikit-learn/`**

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
| **Statistics** | SciPy, statsmodels, pingouin |
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

*Last updated: April 2026 | SciPy Week 1 complete — starting Week 2 practice notebooks*
