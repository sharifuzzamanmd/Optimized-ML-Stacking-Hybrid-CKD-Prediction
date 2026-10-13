"""
Reproducible pipeline for:

    Optimized Machine Learning Models and Stacking Hybrid Approach
    for Chronic Kidney Disease Prediction (IFIP AIAI 2025)

Stages
------
1. Load + clean the CKD dataset (400 records, 24 features + class).
2. Preprocessing step 1: missing-value imputation (random sampling for
   numeric features, mode for categorical), label encoding.
3. Baseline models (KNN, DT, LR, GB, NB) on an 80/20 split.
4. Preprocessing step 2: VarianceThreshold -> PowerTransformer -> SMOTE,
   stratified 80/20 split (random_state=42).
5. Optimized models: either the tuned hyperparameters reported in the
   paper (default) or a fresh GridSearchCV (--tune).
6. Stacking hybrid model: LR + NB + GB base learners, LR meta-learner.
7. Metrics (accuracy, precision, recall, F1, Jaccard, ROC-AUC) and
   ROC curves are written to ./results.

Usage
-----
    python src/ckd_pipeline.py                 # paper hyperparameters
    python src/ckd_pipeline.py --tune          # re-run GridSearchCV (slow)

Note: results can differ slightly from the paper because missing values are
imputed by random sampling and the paper's runs were done in a notebook.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.tree import DecisionTreeClassifier

# The paper keeps max_iter=100 for LR/saga; silence the resulting convergence notices.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "original" / "kidney_disease.csv"
RESULTS_DIR = ROOT / "results"
SEED = 42

COLUMN_NAMES = [
    "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
    "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria",
    "blood_glucose_random", "blood_urea", "serum_creatinine", "sodium",
    "potassium", "haemoglobin", "packed_cell_volume", "white_blood_cell_count",
    "red_blood_cell_count", "hypertension", "diabetes_mellitus",
    "coronary_artery_disease", "appetite", "pedal_edema", "anemia", "class",
]

# Best hyperparameters reported in Table 2 of the paper.
PAPER_PARAMS = {
    "KNN": dict(algorithm="auto", leaf_size=10, metric="manhattan", n_neighbors=3, p=1, weights="uniform"),
    "Decision Tree": dict(criterion="gini", max_depth=5, max_features="log2", min_samples_leaf=2,
                          min_samples_split=3, splitter="best", random_state=SEED),
    "Logistic Regression": dict(C=0.23357214690901212, solver="saga", max_iter=100),  # L2 penalty (default)
    "Gradient Boosting": dict(learning_rate=0.001, loss="exponential", max_depth=5, max_features="sqrt",
                              min_samples_leaf=2, min_samples_split=2, n_estimators=50, subsample=1.0,
                              random_state=SEED),
    "Naive Bayes": dict(priors=None, var_smoothing=0.1),
}

# Search spaces used for GridSearchCV (--tune). Kept close to the notebook,
# with options removed that recent scikit-learn versions no longer accept.
PARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],
        "leaf_size": [10, 20, 30, 40, 50],
    },
    "Decision Tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10],
        "splitter": ["best", "random"],
        "min_samples_leaf": [1, 2, 3, 5, 7],
        "min_samples_split": [2, 3, 5, 7],
        "max_features": ["sqrt", "log2"],
    },
    "Logistic Regression": {
        "C": np.logspace(-4, 4, 20),
                "solver": ["liblinear", "saga", "lbfgs"],
        "max_iter": [100, 500, 1000],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 1.0],
        "max_features": ["sqrt", "log2"],
        "loss": ["log_loss", "exponential"],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    },
}

BASE_ESTIMATORS = {
    "KNN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Logistic Regression": LogisticRegression,
    "Gradient Boosting": GradientBoostingClassifier,
    "Naive Bayes": GaussianNB,
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_and_clean(path: Path = DATA_PATH, seed: int = SEED) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path).drop(columns="id")
    df.columns = COLUMN_NAMES

    # Numeric columns stored as text (contain stray '\t?' entries)
    for col in ["packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace / tab artefacts in categorical values
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    cat_cols = [c for c in obj_cols if c != "class"]
    num_cols = [c for c in df.columns if c not in obj_cols]

    # Preprocessing step 1: imputation
    rng = np.random.RandomState(seed)
    for col in num_cols + ["red_blood_cells", "pus_cell"]:  # random-sample imputation
        missing = df[col].isna()
        if missing.any():
            df.loc[missing, col] = df[col].dropna().sample(missing.sum(), replace=True, random_state=rng).values
    for col in cat_cols:  # mode imputation for the remaining categoricals
        df[col] = df[col].fillna(df[col].mode()[0])

    # Label encoding
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Target encoded as in the paper/notebook (LabelEncoder order):
    # 0 = CKD, 1 = not CKD. Precision/recall are therefore reported for label 1.
    y = (df.pop("class") == "notckd").astype(int)
    return df, y


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "jaccard": jaccard_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def plot_roc(models: dict, X_test, y_test, title: str, out: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true", help="re-run GridSearchCV instead of using paper params")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="path to kidney_disease.csv")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    X, y = load_and_clean(args.data)
    print(f"Loaded {len(X)} records, {X.shape[1]} features | CKD={(y == 0).sum()}, not CKD={(y == 1).sum()}")

    # ---- Baseline models ---------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    baseline_models, baseline_rows = {}, []
    for name, Est in BASE_ESTIMATORS.items():
        kwargs = {"random_state": SEED} if "random_state" in Est().get_params() else {}
        if Est is LogisticRegression:
            kwargs["max_iter"] = 1000
        model = Est(**kwargs).fit(X_tr, y_tr)
        baseline_models[name] = model
        baseline_rows.append({"model": name, **evaluate(model, X_te, y_te)})
    baseline_df = pd.DataFrame(baseline_rows)
    plot_roc(baseline_models, X_te, y_te, "ROC - Baseline Models", RESULTS_DIR / "roc_baseline.png")

    # ---- Preprocessing step 2 ---------------------------------------------
    # As in the paper, transformation and SMOTE are applied to the full
    # feature matrix before the stratified split.
    X2 = VarianceThreshold(threshold=0.01).fit_transform(X)
    X2 = PowerTransformer().fit_transform(X2)
    X2, y2 = SMOTE(sampling_strategy=1.0, random_state=SEED).fit_resample(X2, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.2, random_state=SEED, stratify=y2)

    # ---- Optimized models -------------------------------------------------
    best_params, opt_models, opt_rows = {}, {}, []
    for name, Est in BASE_ESTIMATORS.items():
        if args.tune:
            base = Est(random_state=SEED) if "random_state" in Est().get_params() else Est()
            gs = GridSearchCV(base, PARAM_GRIDS[name], cv=5, scoring="accuracy", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            model, best_params[name] = gs.best_estimator_, gs.best_params_
            print(f"[GridSearchCV] {name}: CV acc={gs.best_score_:.4f} {gs.best_params_}")
        else:
            best_params[name] = PAPER_PARAMS[name]
            model = Est(**PAPER_PARAMS[name]).fit(X_tr, y_tr)
        opt_models[name] = model
        opt_rows.append({"model": f"{name} (optimized)", **evaluate(model, X_te, y_te)})

    # ---- Stacking hybrid --------------------------------------------------
    stack = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(**{**best_params["Logistic Regression"]})),
            ("nb", GaussianNB(**best_params["Naive Bayes"])),
            ("gb", GradientBoostingClassifier(**best_params["Gradient Boosting"])),
        ],
        final_estimator=LogisticRegression(**PAPER_PARAMS["Logistic Regression"]),
        cv=5,
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    opt_models["Stacking Hybrid (LR+NB+GB)"] = stack
    opt_rows.append({"model": "Stacking Hybrid (LR+NB+GB)", **evaluate(stack, X_te, y_te)})
    opt_df = pd.DataFrame(opt_rows)
    plot_roc(opt_models, X_te, y_te, "ROC - Optimized & Hybrid Models", RESULTS_DIR / "roc_optimized_hybrid.png")

    # ---- Save -------------------------------------------------------------
    baseline_df.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    opt_df.to_csv(RESULTS_DIR / "optimized_hybrid_metrics.csv", index=False)
    (RESULTS_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    pd.set_option("display.width", 140)
    fmt = {c: "{:.4f}".format for c in ["accuracy", "precision", "recall", "f1", "jaccard", "roc_auc"]}
    print("\nBaseline models (test set)\n" + baseline_df.to_string(index=False, formatters=fmt))
    print("\nOptimized + hybrid models (test set)\n" + opt_df.to_string(index=False, formatters=fmt))
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
