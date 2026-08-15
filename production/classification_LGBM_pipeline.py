# ======================================================================
# classification_LGBM_pipeline.py
# Generic Binary Classification Pipeline — LightGBM + PyOD + SHAP
# ======================================================================
# A reusable, dataset-agnostic production ML pipeline.
# Supports any binary classification problem — health, churn, fraud, etc.
#
# Usage:
#   1. Load your DataFrame as `df`
#   2. Edit the USER-CONFIGURABLE PARAMETERS block below
#   3. Drop any ID/date columns before running
#   4. Run top to bottom
#
# Architecture (8 Steps):
#   Steps 1–5 : Live inference pipeline
#   Steps 6–8 : Post-training evaluation & audit (offline)
#
#                    [ Raw Input Tabular Vector ]
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 1:         │  PyODOutlierEnsembleDeleter (Training Only)      │
# Anomalous       │  • IForest + CBLOF + KNN ensemble                │
# Defense Filter  │    (→ IForest + CBLOF + HBOS if > 100K rows)     │
#                 │  • Drops top 1% anomalous records during CV.     │
#                 │  • Automatically deactivates at inference time.  │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 2:         │  Metadata Engine & Train-Test Separation         │
# Data            │  • Binds TARGET_COLUMN dynamically.              │
# Preparation     │  • Auto-detects numeric vs categorical features. │
#                 │  • Stratified train/test split (TEST_SIZE=0.2).  │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 3:         │  ColumnTransformer (Ordinal Base)                │
# Preprocessor    │  • KNNImputer (→ MedianImputer if > 100K rows)   │
#                 │  • Scales continuous numeric fields.             │
#                 │  • Converts strings to clean dense integers.     │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 4: Core    │  CalibratedClassifierCV (LightGBM Core)          │
# Classifier &    │  • LightGBM corrects S-curve overconfidence bias │
# Calibration     │    via Isotonic Regression (cv=3, nested in      │
#                 │    outer CV — acceptable for datasets > 3K rows) │
#                 │  • Output probabilities map to real outcome      │
#                 │    frequencies, not raw boosting leaf scores.    │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 5: The     │  TunedThresholdClassifierCV                      │
# Operational     │  • Finds optimal decision boundary via OOF CV.   │
# Gatekeeper      │  • Enforces TARGET_RECALL_FLOOR (default 85%).   │
#                 │  • Maximizes precision above the recall floor.   │
#                 │  • Final model: single LightGBM + single         │
#                 │    isotonic map retrained on full X_train.       │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                         [ Actionable Production Alert Output ]
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   POST-TRAINING EVALUATION & AUDIT (Offline — not
#   part of the live inference path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 6:         │  Holdout Test Validation Report                  │
# Performance     │  • Evaluates final pipeline on untouched X_test. │
# Report          │  • Reports ROC-AUC, Recall, and full             │
#                 │    classification report at optimal threshold.   │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 7:         │  Calibration Curve Validation (Optional)         │
# Probability     │  • Plots predicted probabilities vs true         │
# Audit           │    real-world frequencies.                       │
#                 │  • Confirms isotonic calibration corrected       │
#                 │    LightGBM's S-curve overconfidence bias.       │
#                 │  • Expected output: curve tracks 45° diagonal.   │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                 ┌──────────────────────────────────────────────────┐
# Step 8:         │  SHAP Explainability Layer                       │
# Explainability  │  • Extracts LightGBM from inside pipeline.       │
#                 │  • Beeswarm — global feature impact.             │
#                 │  • Bar — mean absolute SHAP importance.          │
#                 │  • Waterfall — highest-risk deep-dive.           │
#                 │  • Dependency — top feature interaction plot.    │
#                 └──────────────────────────────────────────────────┘
#                                           │
#                                           ▼
#                         [ Audit-Ready Production Deployment ]
# ======================================================================


# ======================================================================
# ── USER-CONFIGURABLE PARAMETERS — Edit these before running ──────────
# ── When using %run -i, define these in your notebook first ───────────
# ── Pipeline uses your notebook values if set, otherwise defaults ─────
# ======================================================================
TARGET_COLUMN         = locals().get('TARGET_COLUMN',         'target')
TARGET_RECALL_FLOOR   = locals().get('TARGET_RECALL_FLOOR',   0.85)
OUTLIER_CONTAMINATION = locals().get('OUTLIER_CONTAMINATION', 0.01)
N_ESTIMATORS          = locals().get('N_ESTIMATORS',          200)
LEARNING_RATE         = locals().get('LEARNING_RATE',         0.03)
MAX_DEPTH             = locals().get('MAX_DEPTH',             5)
TEST_SIZE             = locals().get('TEST_SIZE',             0.2)
RANDOM_STATE          = locals().get('RANDOM_STATE',          420)
CV_FOLDS              = locals().get('CV_FOLDS',              5)
USE_KNN_IMPUTER       = locals().get('USE_KNN_IMPUTER',       True)
LARGE_DATASET_MODE    = locals().get('LARGE_DATASET_MODE',    False)
SCALE_POS_WEIGHT      = locals().get('SCALE_POS_WEIGHT',      None)
ENABLE_SMOTENC        = locals().get('ENABLE_SMOTENC',        False) # → True if > 50:1 imbalance
CALIBRATION_METHOD    = locals().get('CALIBRATION_METHOD',    'auto')  # 'auto'|'isotonic'|'sigmoid'|'none'
CALIBRATION_THRESHOLD = locals().get('CALIBRATION_THRESHOLD', 10.0)    # skip if ratio > threshold × 50
ENABLE_TUNING         = locals().get('ENABLE_TUNING',         False)   # → True to run Optuna tuning (+10-30 min)
N_TRIALS              = locals().get('N_TRIALS',              30)      # number of Optuna trials
TUNING_METRIC         = locals().get('TUNING_METRIC',         'auprc') # 'auprc' | 'roc_auc' | 'recall'

# ── Optuna search ranges (only used when ENABLE_TUNING=True) ──────────
# Format: [min, max] — Optuna searches within these bounds
OPTUNA_N_ESTIMATORS       = locals().get('OPTUNA_N_ESTIMATORS',       [100, 600])
OPTUNA_LEARNING_RATE      = locals().get('OPTUNA_LEARNING_RATE',      [0.01, 0.15])
OPTUNA_MAX_DEPTH          = locals().get('OPTUNA_MAX_DEPTH',          [3, 9])
OPTUNA_NUM_LEAVES         = locals().get('OPTUNA_NUM_LEAVES',         [15, 127])
OPTUNA_MIN_CHILD_SAMPLES  = locals().get('OPTUNA_MIN_CHILD_SAMPLES',  [10, 100])
OPTUNA_SUBSAMPLE          = locals().get('OPTUNA_SUBSAMPLE',          [0.5, 1.0])
OPTUNA_COLSAMPLE_BYTREE   = locals().get('OPTUNA_COLSAMPLE_BYTREE',   [0.5, 1.0])
OPTUNA_REG_ALPHA          = locals().get('OPTUNA_REG_ALPHA',          [0.0, 1.0])
OPTUNA_REG_LAMBDA         = locals().get('OPTUNA_REG_LAMBDA',         [0.0, 1.0])
# ======================================================================
# ⚠️  LARGE DATASET MODE (> 100K rows) — change the values above to:
#
#     N_ESTIMATORS           = 500
#     LEARNING_RATE          = 0.05
#     MAX_DEPTH              = 6
#     CV_FOLDS               = 3
#     USE_KNN_IMPUTER        = False   # switches to median imputation
#     LARGE_DATASET_MODE     = True    # swaps KNN detector → HBOS (O(n))
#
# ⚠️  HYPERPARAMETER TUNING (optional):
#
#     ENABLE_TUNING  = False    # default — uses fixed config params
#     ENABLE_TUNING  = True     # runs Optuna Bayesian search (+10-30 min)
#     N_TRIALS       = 30       # increase for better results, decrease for speed
#     TUNING_METRIC  = 'auprc'  # optimize for AUPRC (best for imbalanced)
#     TUNING_METRIC  = 'roc_auc'# optimize for ROC-AUC (balanced datasets)
#     TUNING_METRIC  = 'recall' # optimize for recall (clinical/fraud)
#
#     Optuna search ranges — [min, max] per parameter:
#     OPTUNA_N_ESTIMATORS      = [100, 600]   # number of trees
#     OPTUNA_LEARNING_RATE     = [0.01, 0.15] # shrinkage rate (log scale)
#     OPTUNA_MAX_DEPTH         = [3, 9]       # max tree depth
#     OPTUNA_NUM_LEAVES        = [15, 127]    # max leaves per tree
#     OPTUNA_MIN_CHILD_SAMPLES = [10, 100]    # min samples per leaf
#     OPTUNA_SUBSAMPLE         = [0.5, 1.0]   # row sampling fraction
#     OPTUNA_COLSAMPLE_BYTREE  = [0.5, 1.0]   # feature sampling fraction
#     OPTUNA_REG_ALPHA         = [0.0, 1.0]   # L1 regularization
#     OPTUNA_REG_LAMBDA        = [0.0, 1.0]   # L2 regularization
#
#     Recommended when:
#       - First time running on a new dataset
#       - AUC/AUPRC is below expectations
#       - You have time to invest (+10-30 min CPU, ~5-10 min GPU)
#
# ⚠️  SEVERE IMBALANCE (> 10:1 ratio) — set:
#
#     SCALE_POS_WEIGHT = None  # leave as None to auto-compute in Step 2
#                              # or manually set to (neg_count / pos_count)
#     Ratio guide:
#       < 10:1   → SCALE_POS_WEIGHT = None, ENABLE_SMOTENC = False
#       10–50:1  → SCALE_POS_WEIGHT = auto (printed in Step 2 diagnostic)
#       > 50:1   → SCALE_POS_WEIGHT = auto + ENABLE_SMOTENC = True
#
# ⚠️  CALIBRATION — controls isotonic/sigmoid/none:
#
#     CALIBRATION_METHOD = 'auto'      # pipeline decides based on class ratio
#     CALIBRATION_METHOD = 'isotonic'  # best for mild imbalance (< 50:1)
#     CALIBRATION_METHOD = 'sigmoid'   # best for small calibration sets
#     CALIBRATION_METHOD = 'none'      # skip — best for extreme imbalance (> 100:1)
#
#     CALIBRATION_THRESHOLD = 10.0     # multiplier — skip if ratio > threshold × 50
#                                      # default 10 = skip if ratio > 500:1
# ======================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Core Pipeline & Feature Processing Utilities
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# Imbalance & PyOD Anomaly Engines (Locked & Decoupled)
from sklearn.pipeline import Pipeline
# BaseSampler removed — PyODOutlierEnsembleDeleter now uses BaseEstimator
from imblearn.pipeline import Pipeline as ImbPipeline
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.utils.utility import standardizer

# Final Production Classifiers & Operational Wrappers
import lightgbm as lgb
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (make_scorer, recall_score, precision_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import TunedThresholdClassifierCV
import scipy.stats as stats


# ======================================================================
# PIPELINE TELEMETRY — Runtime & Resource Monitoring
# ======================================================================
import time
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not installed — memory tracking disabled. Run: pip install psutil")

class PipelineTimer:
    """
    Lightweight telemetry tracker for pipeline step timing and
    resource monitoring. Records wall time, CPU, and memory per step.
    Install psutil for full memory tracking: pip install psutil
    """
    def __init__(self):
        self.steps      = {}
        self.order      = []
        self.start_time = time.time()
        self._process   = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None

    def _mem_gb(self):
        if self._process:
            return self._process.memory_info().rss / 1024**3
        return 0.0

    def start(self, step_name):
        self.order.append(step_name)
        self.steps[step_name] = {
            'start'    : time.time(),
            'mem_start': self._mem_gb()
        }

    def end(self, step_name):
        if step_name not in self.steps:
            return
        s            = self.steps[step_name]
        s['elapsed'] = time.time() - s['start']
        s['mem_end'] = self._mem_gb()
        s['mem_delta'] = s['mem_end'] - s['mem_start']

    def report(self):
        total = time.time() - self.start_time
        print(f"\n{'='*65}")
        print(f"PIPELINE TELEMETRY REPORT")
        print(f"{'='*65}")
        print(f"{'Step':<35} {'Time':>8} {'RAM Δ':>8} {'RAM':>7}")
        print(f"{'-'*65}")
        for name in self.order:
            s = self.steps.get(name, {})
            if 'elapsed' not in s:
                continue
            elapsed  = s['elapsed']
            time_str = f"{elapsed/60:.1f}m" if elapsed >= 60 else f"{elapsed:.1f}s"
            delta    = s.get('mem_delta', 0)
            mem_end  = s.get('mem_end', 0)
            print(f"{name:<35} {time_str:>8} {delta:>+7.2f}G {mem_end:>6.2f}G")
        print(f"{'-'*65}")
        total_str = f"{total/60:.1f}m" if total >= 60 else f"{total:.1f}s"
        print(f"{'TOTAL WALL TIME':<35} {total_str:>8}")
        print(f"{'='*65}\n")

# Instantiate timer — runs throughout pipeline
timer = PipelineTimer()


# ── Load dataset ───────────────────────────────────────────────────────
# Supports both local Windows and Google Colab environments.
# On Colab: pre-define INPUT_DATA_PATH before exec() to override default.
# On local: runs as-is using the default Windows path below.
import pandas as pd

_default_path = r'C:\Users\user\Desktop\Data_Science_Project\MyProject\Production\input_data.csv'
_load_path    = locals().get('INPUT_DATA_PATH', _default_path)

df = pd.read_csv(_load_path, low_memory=False)
print(f"✅ Dataset loaded: {df.shape}")

# ======================================================================
# STEP 1: FAULT-TOLERANT PYOD OUTLIER ENSEMBLE DELETER (SAMPLER)
# ======================================================================
class PyODOutlierEnsembleDeleter(BaseEstimator):
    """
    Unsupervised protective shield. Combines anomaly detectors into an
    ensemble, standardizes and averages scores, then deletes the top 1%
    most severe anomalous records strictly during training folds.
    Deactivates automatically during production API calls to prevent crashes.

    Inherits from BaseEstimator (not BaseSampler) for full compatibility
    with newer sklearn/imblearn versions — avoids _sampling_type and
    _parameter_constraints requirements that BaseSampler enforces.

    Standard mode    (< 100K rows): IForest + KNN  + CBLOF
    Large dataset    (> 100K rows): IForest + HBOS + CBLOF

    Detector complexity:
    - IForest : O(n log n) — scale-safe, handles high-dimensional data well
    - KNN     : O(n²)     — accurate but slow at scale → swap for HBOS
    - HBOS    : O(n)      — histogram-based, scale-safe KNN replacement
    - CBLOF   : O(n)      — cluster-based, catches dense-region outliers
    """
    def __init__(self,
                 contamination=OUTLIER_CONTAMINATION,
                 random_state=RANDOM_STATE,
                 large_dataset_mode=False):    # ← set True if > 100K rows
        self.contamination      = contamination
        self.random_state       = random_state
        self.large_dataset_mode = large_dataset_mode
        # Note: detectors_ set during fit_resample — not __init__
        # Trailing underscore convention reserved for fitted attributes only

    def fit_resample(self, X, y):
        """
        Called by ImbPipeline during training folds only.
        Deactivates at inference time — ImbPipeline never calls
        fit_resample during predict/predict_proba.
        """

        # ── Select detector ensemble based on dataset scale ────────────
        if self.large_dataset_mode:
            from pyod.models.hbos import HBOS
            from pyod.models.lof import LOF
            self.detectors_ = [
                IForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                HBOS(
                    contamination=self.contamination
                    # O(n) histogram-based — no random_state, deterministic
                ),
                LOF(
                    contamination=self.contamination,
                    n_jobs=-1
                    # Local Outlier Factor — robust on large datasets
                    # Replaces CBLOF which fails on high-dimensional data
                    # when cluster separation cannot be formed
                )
            ]
        else:
            self.detectors_ = [
                IForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                KNN(
                    contamination=self.contamination,
                    n_jobs=-1
                    # O(n²) — deterministic, no random_state needed
                ),
                CBLOF(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            ]

        # ── Fit each detector and collect standardized anomaly scores ──
        all_scores = []
        for detector in self.detectors_:
            detector.fit(X)
            scores = detector.decision_function(X).reshape(-1, 1)
            all_scores.append(standardizer(scores))  # z-normalize before averaging

        # ── Average across all three detectors for robust ensemble score
        ensemble_score   = np.mean(np.hstack(all_scores), axis=1)

        # ── Keep only rows below the contamination percentile cutoff ───
        cutoff_threshold = np.percentile(ensemble_score, 100 * (1 - self.contamination))
        keep_mask        = (ensemble_score <= cutoff_threshold)

        return X[keep_mask], y[keep_mask]


# ======================================================================
# MCAR DIAGNOSTIC — KS Test for Structured Missingness
# ======================================================================
def mcar_diagnostic(X, threshold=0.01, sample_size=10000):
    """
    Detects columns where missingness is not random (NMAR/MAR) using
    two-sample Kolmogorov-Smirnov tests.

    If missingness in column A correlates with the distribution of
    column B, the absence of data is itself a predictive signal —
    a binary indicator flag should be added alongside imputation.

    Returns list of NMAR column names. Empty list = MCAR (safe to impute).

    Samples down to sample_size rows for speed on large datasets.
    KS threshold of 0.01 (strict) reduces false positives on large datasets
    where even tiny distributional differences become statistically significant.
    """
    df_audit     = X.sample(min(len(X), sample_size), random_state=420)
    missing_cols = [c for c in df_audit.columns if df_audit[c].isnull().any()]
    numeric_cols = df_audit.select_dtypes(include='number').columns.tolist()
    nmar_cols    = []

    for col in missing_cols:
        mask = df_audit[col].isnull()
        if mask.nunique() < 2:
            continue
        for num_col in numeric_cols:
            if num_col == col:
                continue
            g1 = df_audit.loc[~mask, num_col].dropna()
            g2 = df_audit.loc[mask,  num_col].dropna()
            if len(g1) > 10 and len(g2) > 10:
                _, p = stats.ks_2samp(g1, g2)
                if p < threshold:
                    nmar_cols.append(col)
                    break

    return list(set(nmar_cols))


# ======================================================================
# STEP 2: METADATA ENGINE & PRODUCTION TRAIN-TEST SEPARATION
# ======================================================================
timer.start('Step 2: Data Split')
# Bind target label column (Adapts dynamically to your specific datasets)
X = df.drop(columns=[TARGET_COLUMN], errors='ignore')
y = df[TARGET_COLUMN]

# Parse features automatically to keep the code entirely dataset-agnostic
numeric_features     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("--- [INITIALIZING CLEAN PRODUCTION PIPELINE ENGINE] ---")
print(f"Data Matrix Shape : {X.shape[0]:,} rows, {X.shape[1]} features")
print(f"Target rate       : {y.mean():.2%}")
print(f"Numeric features  : {numeric_features}")
print(f"Categorical feat  : {categorical_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain size : {len(X_train):,} rows")
print(f"Test size  : {len(X_test):,} rows")

# ── MCAR diagnostic on training features ──────────────────────────────
# Run on X_train only — never look at X_test during training
_has_missing = X_train.isnull().any().any()
if _has_missing:
    print(f"\nRunning MCAR diagnostic on training data...")
    nmar_cols = mcar_diagnostic(X_train)
    auto_add_indicator = len(nmar_cols) > 0
    if auto_add_indicator:
        print(f"⚠️  Structured missingness (NMAR) detected in: {nmar_cols}")
        print(f"    Binary indicator flags will be added alongside imputation")
    else:
        print(f"✅ MCAR confirmed — standard imputation is safe, no flags needed")
else:
    nmar_cols          = []
    auto_add_indicator = False
    print(f"\n✅ No missing values detected — MCAR diagnostic skipped")

# ── Imbalance diagnostic ───────────────────────────────────────────────
neg_count  = (y_train == 0).sum()
pos_count  = (y_train == 1).sum()
auto_scale = round(neg_count / pos_count, 2)
print(f"\nClass ratio (neg/pos): {auto_scale:.2f}:1")

if SCALE_POS_WEIGHT is None:
    if auto_scale > 50:
        # Auto-set scale_pos_weight for severe imbalance
        # Don't wait for user to set it — apply immediately
        SCALE_POS_WEIGHT = round(auto_scale, 2)
        print(f"⚠️  Severe imbalance ({auto_scale:.0f}:1) — auto-setting:")
        print(f"    SCALE_POS_WEIGHT = {SCALE_POS_WEIGHT} (auto)")
        if auto_scale > 500:
            print(f"    CALIBRATION_METHOD auto-override → 'none' (ratio > 500:1)")
            CALIBRATION_METHOD = 'none'
    elif auto_scale > 10:
        SCALE_POS_WEIGHT = round(auto_scale, 2)
        print(f"⚠️  Moderate imbalance ({auto_scale:.1f}:1) — auto-setting:")
        print(f"    SCALE_POS_WEIGHT = {SCALE_POS_WEIGHT} (auto)")
    else:
        print(f"✅ Mild imbalance ({auto_scale:.1f}:1) — "
              f"SCALE_POS_WEIGHT=None, ENABLE_SMOTENC=False — threshold tuning sufficient")
else:
    print(f"✅ SCALE_POS_WEIGHT = {SCALE_POS_WEIGHT} (manually set)")

timer.end('Step 2: Data Split')

# ======================================================================
# STEP 3: PREPROCESSOR BLOCK (The Ordinal Core)
# ======================================================================
timer.start('Step 3: Preprocessor Build')
# Ordinal encoding keeps your data matrix narrow and compact.
# This prevents high-dimensional sparse array issues and speeds up
# calculation times. LightGBM handles ordinal integers natively.
# ── FaultTolerantMissingIndicator ─────────────────────────────────────
# Custom transformer that safely handles the case where MCAR diagnostic
# finds no missing columns — returns zero-width array instead of crashing.
# Inherits BaseEstimator + TransformerMixin for full sklearn compatibility.
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion

class FaultTolerantMissingIndicator(BaseEstimator, TransformerMixin):
    """
    Wraps sklearn's MissingIndicator with a safe fallback.
    When active=False or no missing columns exist, returns a zero-width
    array (n_rows, 0) so FeatureUnion can merge without errors.
    Binary flags are kept unscaled (0.0/1.0) for LightGBM splitting efficiency.
    """
    def __init__(self, active=False):
        self.active    = active
        self.indicator = MissingIndicator(error_on_new=False)

    def fit(self, X, y=None):
        if self.active:
            self.indicator.fit(X)
        return self

    def transform(self, X):
        if not self.active or not hasattr(self.indicator, 'features_'):
            # Zero-width array — safe fallback for MCAR or empty datasets
            return np.zeros((X.shape[0], 0))
        # Return raw 0.0/1.0 flags — NOT scaled so LightGBM splits cleanly
        return self.indicator.transform(X).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        """Returns human-readable flag names for SHAP and feature inspection."""
        if not self.active or not hasattr(self.indicator, 'features_'):
            return np.array([])
        if input_features is not None:
            return np.array([f"{input_features[i]}_IS_MISSING"
                             for i in self.indicator.features_])
        return np.array([f"feature_{i}_IS_MISSING"
                         for i in self.indicator.features_])

# ── Parallel numeric processing — two tracks ──────────────────────────
# Track A: continuous values → imputed → StandardScaled
# Track B: binary missing flags → unscaled 0.0/1.0 (LightGBM-friendly)
# FeatureUnion combines both horizontally before ColumnTransformer
numeric_processing_union = FeatureUnion(transformer_list=[
    ('continuous_scaled', Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)
                    if USE_KNN_IMPUTER and not LARGE_DATASET_MODE
                    else SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])),
    ('structural_flags', Pipeline(steps=[
        ('indicator', FaultTolerantMissingIndicator(active=auto_add_indicator))
    ]))
])

if auto_add_indicator:
    print(f"\nMissing indicator flags active — unscaled binary track enabled")
    print(f"  NMAR columns: {nmar_cols}")
else:
    print(f"\nNo missing indicators — standard scaled numeric pipeline")

preprocessor_ordinal = ColumnTransformer(
    transformers=[
        ('num', numeric_processing_union, numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical_features)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
timer.end('Step 3: Preprocessor Build')

# ======================================================================
# STEP 4: CORE ESTIMATOR & PROBABILITY CALIBRATION LAYER
# ======================================================================
# LightGBM is a gradient boosting machine that sequentially minimizes
# residual errors, pushing predictions toward 0 and 1 (S-curve bias).
#
# Calibration strategy:
# CalibratedClassifierCV(cv=3) CANNOT sit inside TunedThresholdClassifierCV
# because PyOD may drop minority class samples from small CV folds, causing
# calibration to see only one class → (n,1) probability output → crash.
#
# Solution: LightGBM fits inside the pipeline. After TunedThresholdClassifierCV
# finds the optimal threshold (Step 5), isotonic calibration is applied ONCE
# on a held-out calibration set in Step 5 using cv='prefit'. This gives
# clean calibration with zero nested CV leakage.

lgb_core = lgb.LGBMClassifier(
    n_estimators=N_ESTIMATORS,          # ← config block
    learning_rate=LEARNING_RATE,        # ← config block
    max_depth=MAX_DEPTH,                # ← config block
    num_leaves=31,
    random_state=RANDOM_STATE,          # ← config block
    scale_pos_weight=SCALE_POS_WEIGHT,  # ← config block (None = no adjustment)
    n_jobs=-1,                          # ← LightGBM handles multi-threading internally
    verbose=-1
)

tuned_pipeline = ImbPipeline([
    ('preprocessor',  preprocessor_ordinal),
    ('outlier_shield', PyODOutlierEnsembleDeleter(
        contamination=OUTLIER_CONTAMINATION,
        random_state=RANDOM_STATE,
        large_dataset_mode=LARGE_DATASET_MODE    # ← config block
    )),
    ('classifier',    lgb_core)
])


# ======================================================================
# STEP 5: AUTOMATED TARGET-CONSTRAINED THRESHOLD-SHIFTING LAYER
# ======================================================================
print("\nOptimizing Operational Decision Boundaries via Cross-Validation Folds...")

def business_constraint_recall_scorer(y_true, y_pred):
    """
    Enforces a strict business recall floor (TARGET_RECALL_FLOOR).
    Maximizes precision to prevent false alarms, but applies a fatal
    penalty if the model drops beneath the mandated catch rate.

    Edge case handling:
    - recall == 0: no positives predicted at all — heavily penalized
    - recall < floor: disqualified with graduated penalty
    - precision < 5% at severe imbalance: rejects flag-everything solution

    The graduated penalty (-1.0 + current_recall) gives Optuna a
    meaningful gradient to learn from rather than a flat -1.0 signal.
    """
    current_recall    = recall_score(y_true, y_pred, zero_division=0)
    current_precision = precision_score(y_true, y_pred, zero_division=0)

    # Edge case: model predicted zero positives entirely
    if current_recall == 0:
        return -1.0

    # Recall floor — graduated penalty gives Optuna learning signal
    if current_recall < TARGET_RECALL_FLOOR:
        return -1.0 + current_recall   # range: -1.0 to (floor - 1.0)

    # Precision floor — rejects flag-everything solution at severe imbalance
    if auto_scale > 100 and current_precision < 0.05:
        return current_precision - 1.0

    return current_precision

custom_operational_scorer = make_scorer(business_constraint_recall_scorer)

# ── 5a. Split a calibration holdout from training data ────────────────
X_train_main, X_cal, y_train_main, y_cal = train_test_split(
    X_train, y_train,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y_train
)

# ── 5b. Fit preprocessor on X_train_main and transform ───────────────
# PyOD requires a numeric matrix — must preprocess before outlier removal
from sklearn.pipeline import Pipeline as SklearnPipeline
import copy

timer.start('Step 5b: Preprocess Transform')
preprocessor_for_pyod = copy.deepcopy(preprocessor_ordinal)
X_train_main_transformed = preprocessor_for_pyod.fit_transform(X_train_main)
X_cal_transformed         = preprocessor_for_pyod.transform(X_cal)
timer.end('Step 5b: Preprocess Transform')

# ── 5c. Run PyOD outlier removal ONCE on transformed training data ────
# PyOD must run outside of TunedThresholdClassifierCV CV folds.
# Auto-bypassed for datasets > 500K rows — PyOD detectors are too slow
# at that scale. LightGBM's native robustness handles outliers implicitly.
PYOD_ROW_LIMIT = 500_000

timer.start('Step 5c: PyOD Outlier Removal')
if len(X_train_main_transformed) > PYOD_ROW_LIMIT:
    print(f"⚠️  PyOD bypassed — {len(X_train_main_transformed):,} rows exceeds "
          f"{PYOD_ROW_LIMIT:,} row limit.")
    print(f"    LightGBM handles outliers natively at this scale.")
    X_train_clean  = X_train_main_transformed
    y_train_clean  = y_train_main
    print(f"Rows passed through : {len(X_train_clean):,}")
else:
    print("Running PyOD outlier ensemble on training data...")
    outlier_shield = PyODOutlierEnsembleDeleter(
        contamination=OUTLIER_CONTAMINATION,
        random_state=RANDOM_STATE,
        large_dataset_mode=LARGE_DATASET_MODE
    )
    X_train_clean, y_train_clean = outlier_shield.fit_resample(
        X_train_main_transformed, y_train_main
    )
    print(f"Rows before outlier removal : {len(X_train_main):,}")
    print(f"Rows after outlier removal  : {len(X_train_clean):,}")
    print(f"Rows removed                : {len(X_train_main) - len(X_train_clean):,}")

timer.end('Step 5c: PyOD Outlier Removal')

# ── 5d. Build pipeline WITHOUT preprocessor and PyOD ─────────────────
# Preprocessor already fitted and applied above — pipeline only needs
# the classifier since input is already a clean numeric matrix.
# If ENABLE_SMOTENC=True, SMOTENC is injected before the classifier
# to handle severe class imbalance (> 50:1 ratio).
if ENABLE_SMOTENC:
    n_numeric_cols   = len(numeric_features)
    n_total_cols     = X_train_clean.shape[1]
    cat_indices_post = list(range(n_numeric_cols, n_total_cols))

    if len(cat_indices_post) == 0:
        # No categorical features — SMOTENC requires at least one
        # Fall back to regular SMOTE for numeric-only datasets
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipelineClean
        print(f"⚠️  SMOTENC requested but no categorical features found")
        print(f"    Falling back to SMOTE (numeric-only dataset)")
        clean_pipeline = ImbPipelineClean([
            ('smote',      SMOTE(random_state=RANDOM_STATE)),
            ('classifier', lgb_core)
        ])
    else:
        from imblearn.over_sampling import SMOTENC
        from imblearn.pipeline import Pipeline as ImbPipelineClean
        print(f"SMOTENC active — categorical indices: {cat_indices_post}")
        clean_pipeline = ImbPipelineClean([
            ('smotenc',   SMOTENC(
                categorical_features=cat_indices_post,
                random_state=RANDOM_STATE
            )),
            ('classifier', lgb_core)
        ])
else:
    clean_pipeline = SklearnPipeline([
        ('classifier', lgb_core)
    ])

# ── 5d. Optional Optuna hyperparameter tuning ────────────────────────
if ENABLE_TUNING:
    try:
        import optuna
    except ImportError:
        print("Installing optuna...")
        import subprocess
        subprocess.run(['pip', 'install', 'optuna', '-q'], capture_output=True)
        import optuna

    # Set verbosity inside the block — avoids polluting global logger
    # for the rest of the session when ENABLE_TUNING=False
    import logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger('optuna').setLevel(logging.WARNING)

    print(f"\n🔍 Optuna hyperparameter search — {N_TRIALS} trials...")
    print(f"   This adds ~10-30 minutes depending on dataset size")

    from sklearn.model_selection import cross_val_score

    # Map TUNING_METRIC to sklearn scoring string
    _metric_map = {
        'auprc'  : 'average_precision',  # best for imbalanced datasets
        'roc_auc': 'roc_auc',            # good for balanced datasets
        'recall' : 'recall',             # clinical/fraud — catch everything
    }
    _scoring = _metric_map.get(TUNING_METRIC, 'average_precision')
    print(f"   Optimizing for: {TUNING_METRIC} ({_scoring})")

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int(
                                     'n_estimators',
                                     OPTUNA_N_ESTIMATORS[0],
                                     OPTUNA_N_ESTIMATORS[1]),
            'learning_rate'    : trial.suggest_float(
                                     'learning_rate',
                                     OPTUNA_LEARNING_RATE[0],
                                     OPTUNA_LEARNING_RATE[1],
                                     log=True),
            'max_depth'        : trial.suggest_int(
                                     'max_depth',
                                     OPTUNA_MAX_DEPTH[0],
                                     OPTUNA_MAX_DEPTH[1]),
            'num_leaves'       : trial.suggest_int(
                                     'num_leaves',
                                     OPTUNA_NUM_LEAVES[0],
                                     OPTUNA_NUM_LEAVES[1]),
            'min_child_samples': trial.suggest_int(
                                     'min_child_samples',
                                     OPTUNA_MIN_CHILD_SAMPLES[0],
                                     OPTUNA_MIN_CHILD_SAMPLES[1]),
            'subsample'        : trial.suggest_float(
                                     'subsample',
                                     OPTUNA_SUBSAMPLE[0],
                                     OPTUNA_SUBSAMPLE[1]),
            'colsample_bytree' : trial.suggest_float(
                                     'colsample_bytree',
                                     OPTUNA_COLSAMPLE_BYTREE[0],
                                     OPTUNA_COLSAMPLE_BYTREE[1]),
            'reg_alpha'        : trial.suggest_float(
                                     'reg_alpha',
                                     OPTUNA_REG_ALPHA[0],
                                     OPTUNA_REG_ALPHA[1]),
            'reg_lambda'       : trial.suggest_float(
                                     'reg_lambda',
                                     OPTUNA_REG_LAMBDA[0],
                                     OPTUNA_REG_LAMBDA[1]),
        }
        lgb_core.set_params(**params)

        scores = cross_val_score(
            clean_pipeline,
            X_train_clean, y_train_clean,
            cv=3,
            scoring=_scoring,
            n_jobs=-1
        )

        # Pruning — stop unpromising trials early
        trial.report(scores.mean(), step=1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return scores.mean()

    # Dynamic min_child_samples bounds based on training set size
    # Prevents underfitting on small datasets and overfitting on large ones
    # Lower: 0.1% of training rows (min 5)
    # Upper: 1.0% of training rows (min 100)
    _n_samples             = len(X_train_clean)
    _dynamic_child_lower   = max(5,   int(_n_samples * 0.001))
    _dynamic_child_upper   = max(100, int(_n_samples * 0.010))
    print(f"   Dynamic min_child_samples range: "
          f"[{_dynamic_child_lower}, {_dynamic_child_upper}] "
          f"({_n_samples:,} training rows)")

    # Override OPTUNA_MIN_CHILD_SAMPLES with dynamic bounds
    OPTUNA_MIN_CHILD_SAMPLES = [_dynamic_child_lower, _dynamic_child_upper]

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_auc    = study.best_value

    print(f"\n✅ Optuna search complete")
    print(f"   Metric      : {TUNING_METRIC}")
    print(f"   Best score  : {best_auc:.4f}")
    print(f"   Best params :")
    for k, v in best_params.items():
        print(f"     {k:<25} : {v}")

    # Apply best params to lgb_core for threshold tuning
    lgb_core.set_params(**best_params)
    print(f"\n✅ Best params applied — proceeding to threshold tuning")

else:
    print(f"\nHyperparameter tuning skipped (ENABLE_TUNING=False)")
    print(f"Using fixed config: n_estimators={N_ESTIMATORS}, "
          f"lr={LEARNING_RATE}, max_depth={MAX_DEPTH}")

# ── 5d. Find optimal threshold on clean training data ─────────────────
production_ready_meta_pipeline = TunedThresholdClassifierCV(
    estimator=clean_pipeline,
    scoring=custom_operational_scorer,
    cv=cv_strategy,
    n_jobs=1    # ← use 1 to avoid multiprocessing serialization issues
)
# Pass categorical feature indices directly to lgb_core before fitting.
# TunedThresholdClassifierCV does not support routing fit params without
# enabling sklearn metadata routing — simpler to set on the object directly.
# This unlocks LightGBM's native categorical optimization so OrdinalEncoder
# integers are treated as categories, not raw continuous numbers.
n_numeric_transformed = len(numeric_features)
cat_feature_indices   = list(range(n_numeric_transformed, X_train_clean.shape[1]))
lgb_core.set_params(categorical_feature=cat_feature_indices)

timer.start('Step 5d: Threshold Tuning (CV)')
production_ready_meta_pipeline.fit(X_train_clean, y_train_clean)
timer.end('Step 5d: Threshold Tuning (CV)')

# Get threshold — works for both calibrated and passthrough
if hasattr(production_ready_meta_pipeline, 'best_threshold_'):
    best_threshold_ = production_ready_meta_pipeline.best_threshold_
else:
    best_threshold_ = production_ready_meta_pipeline.estimator.best_threshold_
print(f"\n🎯 Threshold Optimisation Complete!")
print(f"Optimal Threshold: {best_threshold_:.4f}")

# ── 5e. Apply isotonic calibration on held-out calibration set ────────
# cv='prefit' — model is already fitted, just learn the calibration map
# Corrects LightGBM S-curve overconfidence bias with zero leakage
# ── Determine calibration method ─────────────────────────────────────
# auto mode: skip calibration if class ratio is extreme (> CALIBRATION_THRESHOLD × 50)
# because isotonic over-compresses probabilities toward base rate at extreme imbalance
if CALIBRATION_METHOD == 'auto':
    _skip_threshold = CALIBRATION_THRESHOLD * 50
    if auto_scale > _skip_threshold:
        _effective_method = 'none'
        print(f"\n⚠️  Auto-calibration: skipping — ratio {auto_scale:.0f}:1 "
              f"exceeds {_skip_threshold:.0f}:1 threshold")
        print(f"    Raw LightGBM probabilities used directly")
    elif auto_scale > 50:
        _effective_method = 'sigmoid'
        print(f"\n⚠️  Auto-calibration: using sigmoid — ratio {auto_scale:.0f}:1 "
              f"(moderate-severe imbalance)")
    else:
        _effective_method = 'isotonic'
        print(f"\n✅ Auto-calibration: using isotonic — ratio {auto_scale:.1f}:1 "
              f"(mild imbalance)")
else:
    _effective_method = CALIBRATION_METHOD
    print(f"\nCalibration method: {_effective_method} (manually set)")

# ── Apply calibration ─────────────────────────────────────────────────
timer.start('Step 5e: Calibration')
if _effective_method == 'none':
    print("Calibration skipped — using raw pipeline probabilities")
    # Wrap in a passthrough so predict/predict_proba interface stays consistent
    class _PassthroughCalibrator(BaseEstimator, ClassifierMixin):
        """
        Passthrough wrapper for when calibration is skipped.
        Inherits BaseEstimator + ClassifierMixin so sklearn duck-typing
        checks pass — .classes_, .best_threshold_, .predict(),
        .predict_proba() all exposed correctly.
        """
        def __init__(self, estimator):
            self.estimator       = estimator
            self.classes_        = getattr(estimator, 'classes_',
                                           np.array([0, 1]))
            self.best_threshold_ = getattr(estimator, 'best_threshold_',
                                           0.5)
        def predict(self, X):
            return self.estimator.predict(X)
        def predict_proba(self, X):
            return self.estimator.predict_proba(X)
    production_ready_meta_pipeline = _PassthroughCalibrator(
        production_ready_meta_pipeline
    )
else:
    production_ready_meta_pipeline = CalibratedClassifierCV(
        estimator=production_ready_meta_pipeline,
        method=_effective_method,
        cv='prefit'
    )
    production_ready_meta_pipeline.fit(X_cal_transformed, y_cal)
    print(f"✅ {_effective_method.capitalize()} calibration applied")

timer.end('Step 5e: Calibration')
print(f"✅ Calibration Complete — pipeline ready for evaluation")


# ======================================================================
# STEP 6: HOLDOUT TEST VALIDATION PERFORMANCE REPORT
# ======================================================================
print("\nEvaluating Pipeline Against Untouched Holdout Test Data...")
timer.start('Step 6: Holdout Evaluation')
# Transform X_test using the same preprocessor fitted in Step 5b
X_test_transformed_eval = preprocessor_for_pyod.transform(X_test)
y_pred_final  = production_ready_meta_pipeline.predict(X_test_transformed_eval)
y_proba_final = production_ready_meta_pipeline.predict_proba(X_test_transformed_eval)[:, 1]
timer.end('Step 6: Holdout Evaluation')

print(f"\n==========================================================")
print(f"UNSEEN PRODUCTION HOLDOUT TEST PERFORMANCE:")
print(f"==========================================================")
print(f"Calibrated Test ROC-AUC Score : {roc_auc_score(y_test, y_proba_final):.4f}")
print(f"Enforced Recall Rate (Catch)  : {recall_score(y_test, y_pred_final):.2%}")
print(f"==========================================================\n")

print(classification_report(y_test, y_pred_final,
                             target_names=['Negative', 'Positive']))


# ======================================================================
# STEP 7: CALIBRATION CURVE VALIDATION
# ======================================================================

# ── 1. Compute calibration curve ──────────────────────────────────────
# n_bins=10 gives 10 probability buckets across [0, 1]
# prob_true = actual fraction of positives in each bucket
# prob_pred = mean predicted probability in each bucket
prob_true, prob_pred = calibration_curve(
    y_test, y_proba_final,
    n_bins=10,
    strategy='uniform'   # equal-width bins — better for overconfidence diagnosis
)

# ── 2. Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left — Calibration curve
axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey',
             linewidth=1.5, label='Perfect calibration (45° diagonal)')
axes[0].plot(prob_pred, prob_true, marker='o', color='teal',
             linewidth=2, markersize=7, label='LightGBM (isotonic calibrated)')
axes[0].fill_between(prob_pred, prob_pred, prob_true,
                     alpha=0.1, color='teal', label='Calibration gap')
axes[0].set_xlabel('Mean Predicted Probability')
axes[0].set_ylabel('Fraction of Positives (Actual)')
axes[0].set_title('Calibration Curve — Post Isotonic Correction',
                  fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# Right — Predicted probability distribution
axes[1].hist(y_proba_final[y_test == 0], bins=40, alpha=0.6,
             color='steelblue', label='Negative (0)', edgecolor='white')
axes[1].hist(y_proba_final[y_test == 1], bins=40, alpha=0.6,
             color='crimson', label='Positive (1)', edgecolor='white')
axes[1].axvline(best_threshold_,
                color='black', linestyle='--', linewidth=1.5,
                label=f'Optimal threshold = '
                      f'{best_threshold_:.4f}')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Count')
axes[1].set_title('Predicted Probability Distribution by True Class',
                  fontweight='bold')
axes[1].legend()

plt.suptitle('Production Pipeline — Probability Calibration Audit',
             fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# ── 3. Calibration quality metrics ────────────────────────────────────
# Expected Calibration Error (ECE) — lower is better
# Measures average gap between predicted probability and actual frequency
ece       = np.mean(np.abs(prob_true - prob_pred))
max_error = np.max(np.abs(prob_true - prob_pred))

print(f"""
=== CALIBRATION AUDIT REPORT ===

Expected Calibration Error (ECE): {ece:.4f}
  < 0.05  → Excellent calibration
  < 0.10  → Acceptable calibration
  > 0.10  → Recalibration recommended

Max Single-Bin Error:              {max_error:.4f}
  (worst-case bucket deviation from perfect calibration)

Interpretation:
  {'✅ Isotonic calibration is working — probabilities reliably reflect'
   if ece < 0.05 else
   '⚠️  Mild calibration gap detected — consider increasing cv folds in'}
  {'real-world outcome frequencies.'
   if ece < 0.05 else
   'CalibratedClassifierCV or using a larger calibration dataset.'}

Optimal threshold applied : {best_threshold_:.4f}
Enforced recall floor     : {TARGET_RECALL_FLOOR:.0%}
""")


# ======================================================================
# STEP 8: SHAP EXPLAINABILITY LAYER
# ======================================================================

# ── 1. Extract the fitted LightGBM model from inside the pipeline ─────
# Navigation path:
# TunedThresholdClassifierCV
#   └── .estimator                        → ImbPipeline
#         └── .named_steps
#               ['calibrated_classifier'] → CalibratedClassifierCV
#                 └── .calibrated_classifiers_[0] → _CalibratedClassifier
#                       └── .estimator            → LGBMClassifier
# Navigate through:
# CalibratedClassifierCV (Step 5e)
#   └── .estimator                         → TunedThresholdClassifierCV (Step 5d)
#         └── .estimator_                  → SklearnPipeline
#               └── .named_steps
#                     ['classifier']       → LGBMClassifier
# Navigate through:
# CalibratedClassifierCV (Step 5e)
#   └── .estimator              → TunedThresholdClassifierCV (Step 5d)
#         └── .estimator_       → SklearnPipeline (classifier only)
#               └── .named_steps['classifier'] → LGBMClassifier
# Navigate to LightGBM — path depends on calibration method used
if _effective_method == 'none':
    # PassthroughCalibrator → TunedThresholdClassifierCV → SklearnPipeline
    lgb_model_fitted = (
        production_ready_meta_pipeline
        .estimator                                  # TunedThresholdClassifierCV
        .estimator_                                 # SklearnPipeline (fitted)
        .named_steps['classifier']                  # LGBMClassifier
    )
else:
    # CalibratedClassifierCV → TunedThresholdClassifierCV → SklearnPipeline
    lgb_model_fitted = (
        production_ready_meta_pipeline
        .estimator                                  # TunedThresholdClassifierCV
        .estimator_                                 # SklearnPipeline (fitted)
        .named_steps['classifier']                  # LGBMClassifier
    )

# ── 2. Transform X_test through preprocessor ──────────────────────────
# Preprocessor was fitted in Step 5b — use it directly
X_test_transformed = preprocessor_for_pyod.transform(X_test)

# ── 3. Recover feature names post-preprocessing ───────────────────────
# FeatureUnion inside ColumnTransformer can break get_feature_names_out()
# on older sklearn versions — use manual reconstruction as safe fallback
try:
    feature_names = preprocessor_for_pyod.get_feature_names_out().tolist()
    feature_names = [f.split('__')[-1] for f in feature_names]
    print(f"✅ Feature names recovered via get_feature_names_out()")
except Exception:
    # Manual reconstruction — guaranteed to work regardless of sklearn version
    # Follows exact ColumnTransformer output order:
    # [numeric_scaled | numeric_IS_MISSING flags | categorical encoded | passthrough]
    feature_names = list(numeric_features)

    if auto_add_indicator:
        try:
            fitted_union      = preprocessor_for_pyod.named_transformers_['num']
            fitted_indicator  = (fitted_union
                                 .transformer_list[1][1]
                                 .named_steps['indicator']
                                 .indicator)
            if hasattr(fitted_indicator, 'features_'):
                flag_names = [f"{numeric_features[i]}_IS_MISSING"
                              for i in fitted_indicator.features_]
                feature_names += flag_names
                print(f"✅ Missing flag names recovered: {flag_names}")
        except Exception as e:
            print(f"⚠️  Could not recover flag names: {e}")

    # Add categorical feature names (OrdinalEncoder keeps original column names)
    feature_names += list(categorical_features)
    print(f"✅ Feature names manually reconstructed ({len(feature_names)} features)")
feature_names = [f.split('__')[-1] for f in feature_names]

print(f"SHAP input matrix shape : {X_test_transformed.shape}")
print(f"Features                : {feature_names}")

# ── 4. Build TreeExplainer and compute SHAP values ────────────────────
timer.start('Step 8: SHAP Computation')
explainer   = shap.TreeExplainer(lgb_model_fitted)
shap_values_raw = explainer.shap_values(X_test_transformed)

# For binary classification LightGBM returns a list of two arrays:
#   shap_values_raw[0] → negative class (0) contributions
#   shap_values_raw[1] → positive class (1) contributions
# Always use index [1] — we care about what drives the positive outcome
shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw
timer.end('Step 8: SHAP Computation')

# ── 5. Beeswarm — global feature impact ───────────────────────────────
print("\n--- Global Feature Impact (Beeswarm) ---")
shap.summary_plot(
    shap_values, X_test_transformed,
    feature_names=feature_names,
    plot_type='dot', show=True
)

# ── 6. Bar — mean absolute SHAP per feature ────────────────────────────
print("\n--- Mean Absolute SHAP Importance (Bar) ---")
shap.summary_plot(
    shap_values, X_test_transformed,
    feature_names=feature_names,
    plot_type='bar', show=True
)

# ── 7. Waterfall — highest-risk case deep-dive ────────────────────────
highest_risk_idx = y_proba_final.argmax()
print(f"\n--- Highest Risk Case (index {highest_risk_idx}) ---")
print(f"Predicted probability : {y_proba_final[highest_risk_idx]:.4f}")
print(f"Actual outcome        : "
      f"{'Positive (1)' if y_test.iloc[highest_risk_idx] == 1 else 'Negative (0)'}")

# Extract positive class base value — LightGBM binary classification
# returns expected_value as list [base_neg, base_pos] in some SHAP versions
_base_val = (explainer.expected_value[1]
             if isinstance(explainer.expected_value, (list, np.ndarray))
             else explainer.expected_value)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[highest_risk_idx],
        base_values=_base_val,
        data=X_test_transformed[highest_risk_idx],
        feature_names=feature_names
    )
)

# ── 8. Waterfall — lowest-risk case deep-dive ────────────────────────
lowest_risk_idx = y_proba_final.argmin()
print(f"\n--- Lowest Risk Case (index {lowest_risk_idx}) ---")
print(f"Predicted probability : {y_proba_final[lowest_risk_idx]:.4f}")
print(f"Actual outcome        : "
      f"{'Positive (1)' if y_test.iloc[lowest_risk_idx] == 1 else 'Negative (0)'}")

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[lowest_risk_idx],
        base_values=_base_val,   # reuse positive class base value
        data=X_test_transformed[lowest_risk_idx],
        feature_names=feature_names
    )
)

# ── 9. Dependency plot — top feature interaction ───────────────────────
top_feature    = feature_names[np.abs(shap_values).mean(axis=0).argmax()]
second_feature = feature_names[np.abs(shap_values).mean(axis=0).argsort()[-2]]

print(f"\n--- Dependency Plot: {top_feature} × {second_feature} ---")
shap.dependence_plot(
    top_feature, shap_values, X_test_transformed,
    feature_names=feature_names,
    interaction_index=second_feature,
    show=True
)

# ── Final telemetry report ─────────────────────────────────────────────
timer.report()
