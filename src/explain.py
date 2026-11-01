"""Phase 4: SHAP explanations for Model A and B.

Uses TreeExplainer for tree-based models, LinearExplainer for LR.
Saves global summary plots and provides explain_student() for per-student explanations.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from src.features import (
    prepare_data, build_preprocessor, get_feature_names,
    engineer_features, GRADE_COLS, SENSITIVE_COLS,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

# Human-readable feature name mapping
FEATURE_DISPLAY = {
    "alc_total": "Total alcohol consumption",
    "parent_edu": "Parental education (avg)",
    "support_index": "Support index (school+family+paid)",
    "absence_bucket": "Absence level",
    "failures": "Past failures",
    "absences": "Absences",
    "studytime": "Weekly study time",
    "goout": "Going out frequency",
    "Dalc": "Workday alcohol",
    "Walc": "Weekend alcohol",
    "health": "Health status",
    "freetime": "Free time",
    "famrel": "Family relationship quality",
    "age": "Age",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "traveltime": "Travel time",
    "G1": "First period grade",
    "G2": "Second period grade",
    "grade_trend": "Grade trend (G2-G1)",
    "higher": "Wants higher education",
    "romantic": "In a relationship",
    "internet": "Internet access",
    "schoolsup": "School support",
    "famsup": "Family support",
    "paid": "Paid classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery",
}


def _get_background_data(variant, preprocessor):
    """Load and transform the test set as SHAP background for LinearExplainer.

    LinearExplainer needs a representative background dataset to compute
    the expected value (mean prediction). Using a single row produces
    degenerate all-zero SHAP values.
    """
    test_df = pd.read_csv(os.path.join(MODELS_DIR, f"test_set_{variant}.csv"))
    X_bg, _, _ = prepare_data(test_df, variant)
    return preprocessor.transform(X_bg)


def _get_explainer_and_shap_values(model, X_transformed, feature_names):
    """Create appropriate SHAP explainer based on model type."""
    # The calibrated model wraps estimators; get the base estimator
    base_estimators = model.calibrated_classifiers_
    # Use the first calibrated classifier's estimator
    base_pipe = base_estimators[0].estimator
    base_clf = base_pipe.named_steps["clf"]

    clf_type = type(base_clf).__name__
    if clf_type in ("LGBMClassifier", "RandomForestClassifier"):
        # For tree models, we need to transform the data first
        preprocessor = base_pipe.named_steps["prep"]
        explainer = shap.TreeExplainer(base_clf)
    else:
        # LinearExplainer for LogisticRegression
        preprocessor = base_pipe.named_steps["prep"]
        explainer = shap.LinearExplainer(base_clf, X_transformed)

    shap_values = explainer.shap_values(X_transformed)

    # For binary classification, shap_values might be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    return explainer, shap_values


def generate_shap_plots(variant="A"):
    """Generate and save global SHAP summary plot for a model variant."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    model = joblib.load(os.path.join(MODELS_DIR, f"model_{variant}.joblib"))
    test_df = pd.read_csv(os.path.join(MODELS_DIR, f"test_set_{variant}.csv"))

    # Prepare features from test set
    test_X, _, _ = prepare_data(test_df, variant)

    # Transform using the model's internal preprocessor
    base_pipe = model.calibrated_classifiers_[0].estimator
    preprocessor = base_pipe.named_steps["prep"]
    X_transformed = preprocessor.transform(test_X)

    feature_names = get_feature_names(preprocessor, variant)

    explainer, shap_values = _get_explainer_and_shap_values(
        model, X_transformed, feature_names
    )

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        show=False, max_display=15
    )
    plt.title(f"SHAP Summary — Model {variant} ({'Early Warning' if variant == 'A' else 'Mid-Term'})")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"shap_summary_{variant}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP summary plot saved: {path}")

    return explainer, feature_names


def explain_student(row_df, variant="A"):
    """Explain a single student's prediction.

    Args:
        row_df: single-row DataFrame (raw features, same schema as student_merged.csv)
        variant: "A" or "B"

    Returns:
        dict with keys:
            drivers: list of top 3 risk-increasing features [{name, value, direction, shap_value}]
            protective: list of top 2 risk-decreasing features [{name, value, direction, shap_value}]
    """
    model = joblib.load(os.path.join(MODELS_DIR, f"model_{variant}.joblib"))

    X, _, _ = prepare_data(row_df, variant)
    base_pipe = model.calibrated_classifiers_[0].estimator
    preprocessor = base_pipe.named_steps["prep"]
    base_clf = base_pipe.named_steps["clf"]

    X_transformed = preprocessor.transform(X)
    feature_names = get_feature_names(preprocessor, variant)

    clf_type = type(base_clf).__name__
    if clf_type in ("LGBMClassifier", "RandomForestClassifier"):
        explainer = shap.TreeExplainer(base_clf)
    else:
        background = _get_background_data(variant, preprocessor)
        explainer = shap.LinearExplainer(base_clf, background)

    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):
        sv = sv[1]

    sv = sv.flatten()
    feature_vals = X_transformed.flatten() if hasattr(X_transformed, "flatten") else np.array(X_transformed).flatten()

    # Sort by SHAP value
    sorted_idx = np.argsort(sv)

    # Top 3 risk drivers (highest positive SHAP)
    drivers = []
    for i in sorted_idx[::-1]:
        if len(drivers) >= 3:
            break
        name = feature_names[i]
        display = FEATURE_DISPLAY.get(name, name)
        drivers.append({
            "feature": name,
            "display_name": display,
            "value": float(feature_vals[i]),
            "direction": "increases risk",
            "shap_value": float(sv[i]),
        })

    # Top 2 protective (most negative SHAP)
    protective = []
    for i in sorted_idx:
        if len(protective) >= 2:
            break
        if sv[i] >= 0:
            break
        name = feature_names[i]
        display = FEATURE_DISPLAY.get(name, name)
        protective.append({
            "feature": name,
            "display_name": display,
            "value": float(feature_vals[i]),
            "direction": "decreases risk",
            "shap_value": float(sv[i]),
        })

    return {"drivers": drivers, "protective": protective}


def explain_student_chart(row_df, variant="A"):
    """Generate a SHAP bar chart for a single student. Returns the figure."""
    model = joblib.load(os.path.join(MODELS_DIR, f"model_{variant}.joblib"))
    X, _, _ = prepare_data(row_df, variant)
    base_pipe = model.calibrated_classifiers_[0].estimator
    preprocessor = base_pipe.named_steps["prep"]
    base_clf = base_pipe.named_steps["clf"]

    X_transformed = preprocessor.transform(X)
    feature_names = get_feature_names(preprocessor, variant)

    clf_type = type(base_clf).__name__
    if clf_type in ("LGBMClassifier", "RandomForestClassifier"):
        explainer = shap.TreeExplainer(base_clf)
    else:
        background = _get_background_data(variant, preprocessor)
        explainer = shap.LinearExplainer(base_clf, background)

    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv.flatten()

    # Top 8 by absolute SHAP
    top_idx = np.argsort(np.abs(sv))[-8:]
    top_names = [FEATURE_DISPLAY.get(feature_names[i], feature_names[i]) for i in top_idx]
    top_vals = [sv[i] for i in top_idx]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_names, top_vals, color=colors)
    ax.set_xlabel("SHAP value (impact on risk)")
    ax.set_title("Feature Contributions to Risk Score")
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    return fig


def run():
    """Generate SHAP plots for both variants and demo explain_student."""
    for variant in ["A", "B"]:
        print(f"\nModel {variant}:")
        generate_shap_plots(variant)

    # Demo explain_student on first test student
    test_df = pd.read_csv(os.path.join(MODELS_DIR, "test_set_A.csv"))
    sample = test_df.iloc[[0]]
    explanation = explain_student(sample, "A")
    print("\nSample explanation (first test student, Model A):")
    print("  Drivers:")
    for d in explanation["drivers"]:
        print(f"    {d['display_name']}: {d['direction']} (SHAP={d['shap_value']:.3f})")
    print("  Protective:")
    for p in explanation["protective"]:
        print(f"    {p['display_name']}: {p['direction']} (SHAP={p['shap_value']:.3f})")

    print("\n--- Phase 4 Status ---")
    print(f"✓ SHAP summary plots saved for Model A and B")
    print(f"✓ explain_student() returns top 3 drivers + top 2 protective factors")
    print(f"✓ explain_student_chart() generates per-student bar chart")


if __name__ == "__main__":
    run()
