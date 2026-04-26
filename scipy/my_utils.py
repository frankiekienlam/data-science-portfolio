# my_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas as pd            # <--- Added for your second function
from scipy import stats        # <--- Added for your second function

def check_assumptions_regression(model):
    # ... keep the code from the first function here ...
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=axes[0], 
                  line_kws={'color': 'red', 'lw': 1})
    axes[0].set_title('Residuals vs Fitted')
    sm.qqplot(model.resid, line='s', ax=axes[1])
    axes[1].set_title('Normal Q-Q Plot')
    plt.tight_layout()
    plt.show()

def check_assumptions_anova(*groups, names=None, alpha=0.05):
    # ... paste your new function here ...
    if names is None:
        names = [f"Group {i+1}" for i in range(len(groups))]
    
    results = []
    normality_met = True
    
    for name, data in zip(names, groups):
        stat, p_val = stats.shapiro(data)
        is_normal = p_val > alpha
        if not is_normal:
            normality_met = False
        results.append({
            'Group': name,
            'n': len(data),
            'Shapiro p-value': round(p_val, 4),
            'Normal?': "Yes" if is_normal else "No"
        })
    
    lev_stat, lev_p = stats.levene(*groups)
    variance_met = lev_p > alpha
    
    df_results = pd.DataFrame(results)
    print("\n--- Normality Check (Shapiro-Wilk) ---")
    print(df_results.to_string(index=False))
    print(f"\n--- Variance Check (Levene's) ---")
    print(f"Levene p-value: {round(lev_p, 4)} | Equal Variance? {'Yes' if variance_met else 'No'}")
    
    if not normality_met:
        return 'Recommendation: Use Kruskal-Wallis — normality violated'
    if not variance_met:
        return 'Recommendation: Use Kruskal-Wallis — equal variance violated'
    return 'Recommendation: Use ANOVA — assumptions met'
