"""Phase 3: Training, evaluation, calibration, cost-based threshold, and model saving.

Trains LogisticRegression, RandomForest, LightGBM on both feature sets (A/B).
Uses StratifiedGroupKFold(5) with student_key as groups.
Reports PR-AUC, ROC-AUC, Recall(at-risk), F2.
Picks best per feature set by PR-AUC, calibrates, finds threshold for recall >= 0.85.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, recall_score, fbeta_score,
    precision_score,
)
import lightgbm as lgb

from src.features import prepare_data, build_preprocessor, get_feature_names

SEED = 42
N_FOLDS = 5
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Risk tier cutoffs
TIER_LOW = 0.3
TIER_HIGH = 0.6


def _ensure_dirs():
    for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def get_models():
    """Return dict of model name -> estimator."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, is_unbalance=True, random_state=SEED,
            verbose=-1, n_jobs=-1
        ),
    }


def cross_validate(X, y, groups, preprocessor, model, n_folds=N_FOLDS):
    """Run StratifiedGroupKFold CV, return OOF predictions and metrics."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y))
    metrics = {"pr_auc": [], "roc_auc": [], "recall": [], "f2": []}

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds

        metrics["pr_auc"].append(average_precision_score(y_val, probs))
        metrics["roc_auc"].append(roc_auc_score(y_val, probs))
        metrics["recall"].append(recall_score(y_val, preds))
        metrics["f2"].append(fbeta_score(y_val, preds, beta=2))

    return oof_probs, oof_preds, metrics


def find_threshold_for_recall(y, probs, target_recall=0.85):
    """Find the *highest* threshold that still achieves >= target_recall.
    Searching high-to-low maximises precision subject to the recall constraint."""
    thresholds = np.arange(0.99, 0.0, -0.01)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        r = recall_score(y, preds, zero_division=0)
        if r >= target_recall:
            p = precision_score(y, preds, zero_division=0)
            return float(t), float(r), float(p)
    # fallback: lowest threshold
    return 0.05, float(recall_score(y, (probs >= 0.05).astype(int))), \
           float(precision_score(y, (probs >= 0.05).astype(int)))


def risk_tier(prob):
    if prob < TIER_LOW:
        return "Low"
    elif prob <= TIER_HIGH:
        return "Medium"
    else:
        return "High"


def train_and_evaluate(df):
    """Full training pipeline for both Model A and B."""
    _ensure_dirs()
    results = {}

    for variant in ["A", "B"]:
        X, y, groups = prepare_data(df, variant)
        preprocessor = build_preprocessor(variant)
        models = get_models()
        variant_results = {}

        print(f"\n{'='*60}")
        print(f"Model {variant} ({'Early Warning' if variant == 'A' else 'Mid-Term'})")
        print(f"{'='*60}")

        best_name, best_pr_auc = None, -1
        best_oof_probs = None

        for name, model in models.items():
            oof_probs, oof_preds, metrics = cross_validate(
                X, y, groups, preprocessor, model
            )
            m = {k: f"{np.mean(v):.3f} ± {np.std(v):.3f}" for k, v in metrics.items()}
            m_raw = {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
            print(f"\n  {name}:")
            for k, v in m.items():
                print(f"    {k}: {v}")

            variant_results[name] = m_raw
            mean_pr = float(np.mean(metrics["pr_auc"]))
            if mean_pr > best_pr_auc:
                best_pr_auc = mean_pr
                best_name = name
                best_oof_probs = oof_probs

        print(f"\n  >> Best for Model {variant}: {best_name} (PR-AUC={best_pr_auc:.4f})")

        # --- Retrain best model on full training data, calibrate ---------------
        # Group-aware train/test split (80/20)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_idx, test_idx = next(sgkf.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        preprocessor_final = build_preprocessor(variant)
        best_model = get_models()[best_name]
        pipe = Pipeline([("prep", preprocessor_final), ("clf", best_model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        # Calibrate
        cal_model = CalibratedClassifierCV(pipe, cv=3, method="isotonic")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal_model.fit(X_train, y_train)

        test_probs = cal_model.predict_proba(X_test)[:, 1]

        # Cost-based threshold
        threshold, recall_at_t, precision_at_t = find_threshold_for_recall(y_test, test_probs)
        print(f"  Threshold for recall>=0.85: {threshold:.2f} "
              f"(recall={recall_at_t:.3f}, precision={precision_at_t:.3f})")

        # Test-set metrics at chosen threshold
        test_preds = (test_probs >= threshold).astype(int)
        test_metrics = {
            "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
            "recall": round(float(recall_score(y_test, test_preds)), 4),
            "precision_at_threshold": round(precision_at_t, 4),
            "f2": round(float(fbeta_score(y_test, test_preds, beta=2)), 4),
            "threshold": round(threshold, 2),
        }

        # Save model and test data
        model_path = os.path.join(MODELS_DIR, f"model_{variant}.joblib")
        joblib.dump(cal_model, model_path)
        print(f"  Saved calibrated model to {model_path}")

        # Save test set for later use (dashboard, notes)
        test_df = df.iloc[test_idx].copy()
        test_df["risk_score"] = test_probs
        test_df["risk_tier"] = [risk_tier(p) for p in test_probs]
        test_df["predicted_at_risk"] = test_preds
        test_path = os.path.join(MODELS_DIR, f"test_set_{variant}.csv")
        test_df.to_csv(test_path, index=False)

        # Save preprocessor separately for what-if panel
        pp_only = build_preprocessor(variant)
        pp_only.fit(X_train)
        joblib.dump(pp_only, os.path.join(MODELS_DIR, f"preprocessor_{variant}.joblib"))

        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, test_probs, n_bins=8
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Model {variant}")
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration Curve — Model {variant}")
        ax.legend()
        fig.savefig(os.path.join(FIGURES_DIR, f"calibration_{variant}.png"), dpi=150)
        plt.close(fig)

        results[f"model_{variant}"] = {
            "best_model": best_name,
            "cv_results": variant_results,
            "test_metrics": test_metrics,
            "tier_cutoffs": {"low": TIER_LOW, "high": TIER_HIGH},
        }

    # --- Comparison bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, metric in enumerate(["pr_auc", "recall"]):
        a_val = results["model_A"]["test_metrics"][metric]
        b_val = results["model_B"]["test_metrics"][metric]
        axes[i].bar(["Model A\n(Early Warning)", "Model B\n(Mid-Term)"],
                     [a_val, b_val], color=["#3498db", "#e74c3c"], alpha=0.85)
        axes[i].set_title(metric.upper().replace("_", "-"))
        axes[i].set_ylim(0, 1)
        for j, v in enumerate([a_val, b_val]):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    plt.suptitle("Model A vs B Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Save metrics.json ---
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    return results


def validate_uci_697():
    """Phase 7: Cross-dataset validation on UCI dataset 697.
    Target: Dropout vs not-Dropout (Enrolled + Graduate).
    Trains one LightGBM with same CV/metrics approach."""
    from ucimlrepo import fetch_ucirepo
    import shap

    _ensure_dirs()
    print("\n" + "=" * 60)
    print("Phase 7: UCI 697 Cross-Dataset Validation")
    print("=" * 60)

    dataset = fetch_ucirepo(id=697)
    df = dataset.data.original.copy()
    print(f"  Loaded UCI 697: {len(df)} rows, {len(df.columns)} cols")

    # Build target: Dropout=1, else=0 (Graduate + Enrolled = not dropout)
    target_col = df.columns[-1]  # 'Target' column
    df["dropout"] = (df[target_col] == "Dropout").astype(int)
    print(f"  Dropout rate: {df['dropout'].mean():.1%} ({df['dropout'].sum()} / {len(df)})")

    # Features: all columns except target and our label
    feature_cols = [c for c in df.columns if c not in [target_col, "dropout"]]
    # Encode any string columns
    from sklearn.preprocessing import LabelEncoder
    le_dict = {}
    for c in feature_cols:
        if df[c].dtype == "object":
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            le_dict[c] = le

    X = df[feature_cols].values
    y = df["dropout"].values
    # No natural group key; use row index as dummy groups for StratifiedGroupKFold
    groups = np.arange(len(df))

    model = lgb.LGBMClassifier(
        n_estimators=200, is_unbalance=True, random_state=SEED, verbose=-1, n_jobs=-1
    )
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y))
    metrics = {"pr_auc": [], "roc_auc": [], "recall": [], "f2": []}

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        oof_probs[val_idx] = probs

        metrics["pr_auc"].append(average_precision_score(y_val, probs))
        metrics["roc_auc"].append(roc_auc_score(y_val, probs))
        metrics["recall"].append(recall_score(y_val, preds))
        metrics["f2"].append(fbeta_score(y_val, preds, beta=2))

    cv_results = {k: f"{np.mean(v):.3f} ± {np.std(v):.3f}" for k, v in metrics.items()}
    cv_raw = {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
    print("  CV Results:")
    for k, v in cv_results.items():
        print(f"    {k}: {v}")

    # SHAP top features (on last fold model)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-10:][::-1]
    top_features = [(feature_cols[i], round(float(mean_abs_shap[i]), 4)) for i in top_idx]
    print("  Top SHAP features:")
    for name, importance in top_features:
        print(f"    {name}: {importance}")

    # Save global SHAP plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.summary_plot(shap_values, X_val, feature_names=feature_cols, show=False, max_display=15)
    plt.title("SHAP Summary — UCI 697 (Dropout Prediction)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "shap_uci697.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "cv_metrics": cv_raw,
        "top_shap_features": [{"feature": n, "mean_abs_shap": v} for n, v in top_features],
        "n_samples": len(df),
        "dropout_rate": round(float(df["dropout"].mean()), 4),
    }

    # Append to metrics.json
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    all_metrics["uci_697_validation"] = result
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Results appended to {metrics_path}")

    print("\n--- Phase 7 Status ---")
    print(f"✓ UCI 697: {len(df)} rows, dropout rate {df['dropout'].mean():.1%}")
    print(f"✓ LightGBM CV PR-AUC={cv_raw['pr_auc']}, Recall={cv_raw['recall']}")
    print(f"✓ Results saved to metrics.json under uci_697_validation")

    return result


def run():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "student_merged.csv")
    df = pd.read_csv(data_path)
    results = train_and_evaluate(df)
    print("\n--- Phase 3 Status ---")
    for v in ["A", "B"]:
        m = results[f"model_{v}"]["test_metrics"]
        print(f"✓ Model {v}: best={results[f'model_{v}']['best_model']}, "
              f"PR-AUC={m['pr_auc']:.3f}, Recall={m['recall']:.3f}, "
              f"threshold={m['threshold']}")
    print(f"✓ Models saved, metrics.json written, comparison chart saved")

    # Phase 7
    validate_uci_697()


if __name__ == "__main__":
    run()
