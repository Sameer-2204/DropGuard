"""Phase 2: Feature engineering and two feature sets (A/B).

G1 and G2 are mid-term grades. Including them in an early-warning model would
be impossible (they don't exist yet) and creates temporal leakage. G3 is the
final grade *and* the basis of the target, so it's always excluded.

Model A ("early warning"): uses only demographics, family, behaviour, support —
   available at enrolment. No G1, G2, G3.
Model B ("mid-term"): adds G1, G2 and grade_trend = G2 - G1, simulating a
   mid-semester checkpoint. Still excludes G3.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

SEED = 42

# --- columns by type ---------------------------------------------------------
BINARY_COLS = ["schoolsup", "famsup", "paid", "activities", "nursery",
               "higher", "internet", "romantic"]

NOMINAL_COLS = ["school", "sex", "address", "famsize", "Pstatus",
                "Mjob", "Fjob", "reason", "guardian"]

NUMERIC_COLS = ["age", "Medu", "Fedu", "traveltime", "studytime",
                "failures", "famrel", "freetime", "goout",
                "Dalc", "Walc", "health", "absences"]

# Columns excluded from recommendations (sensitive attrs)
SENSITIVE_COLS = ["sex", "address", "Pstatus", "Mjob", "Fjob"]

# --- feature definitions ------------------------------------------------------
GRADE_COLS = ["G1", "G2"]        # added only in Model B
TARGET_COL = "at_risk"
GROUP_COL = "student_key"
EXCLUDE_ALWAYS = ["G3", "at_risk", "student_key", "subject"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe (in place)."""
    df = df.copy()
    df["alc_total"] = df["Dalc"] + df["Walc"]
    df["parent_edu"] = (df["Medu"] + df["Fedu"]) / 2.0
    # support_index: count of binary support flags that are 'yes'
    df["support_index"] = (
        (df["schoolsup"] == "yes").astype(int)
        + (df["famsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )
    bins = [-1, 5, 10, 20, 100]
    labels = [0, 1, 2, 3]  # ordinal encoding for bins
    df["absence_bucket"] = pd.cut(
        df["absences"], bins=bins, labels=labels, right=True
    ).astype(float).fillna(0).astype(int)
    return df


def _get_feature_cols(model_variant: str) -> list:
    """Return feature column names for Model A or B."""
    engineered = ["alc_total", "parent_edu", "support_index", "absence_bucket"]
    base = NUMERIC_COLS + engineered
    cat = BINARY_COLS + NOMINAL_COLS

    if model_variant == "B":
        base = base + GRADE_COLS + ["grade_trend"]
    return base, cat


def build_preprocessor(model_variant: str = "A") -> ColumnTransformer:
    """Build a ColumnTransformer for the given model variant."""
    num_cols, cat_cols = _get_feature_cols(model_variant)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("bin", OrdinalEncoder(), BINARY_COLS),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             NOMINAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, model_variant: str = "A") -> list:
    """Get feature names after transformation."""
    num_cols, cat_cols = _get_feature_cols(model_variant)
    num_names = num_cols
    bin_names = BINARY_COLS
    nom_names = list(preprocessor.named_transformers_["nom"].get_feature_names_out(NOMINAL_COLS))
    return num_names + bin_names + nom_names


def prepare_data(df: pd.DataFrame, model_variant: str = "A"):
    """Prepare X, y, groups for a given model variant.
    Returns (X_df, y, groups) where X_df has only the relevant columns."""
    df = engineer_features(df)
    if model_variant == "B":
        df["grade_trend"] = df["G2"] - df["G1"]

    num_cols, cat_cols = _get_feature_cols(model_variant)
    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values
    return X, y, groups


def run():
    """Quick smoke test: load data, build both feature sets, print shapes."""
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "student_merged.csv")
    df = pd.read_csv(data_path)
    for variant in ["A", "B"]:
        X, y, groups = prepare_data(df, variant)
        pp = build_preprocessor(variant)
        X_t = pp.fit_transform(X)
        names = get_feature_names(pp, variant)
        has_g1 = "G1" in X.columns
        has_g3 = "G3" in X.columns
        print(f"Model {variant}: X shape={X_t.shape}, features={len(names)}, "
              f"has_G1={has_g1}, has_G3={has_g3}")

    print("\n--- Phase 2 Status ---")
    print(f"✓ Feature engineering: alc_total, parent_edu, support_index, absence_bucket")
    print(f"✓ Model A (early warning): no grades | Model B (mid-term): G1, G2, grade_trend")
    print(f"✓ G3 excluded from both feature sets")


if __name__ == "__main__":
    run()
