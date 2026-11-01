"""Phase 1: Data ingestion, label creation, and EDA."""
import os
import urllib.request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

# 13 shared attributes used to identify the same student across math/por
STUDENT_KEY_COLS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet",
]

URLS = {
    "student-mat.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
    "student-por.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
}


def _ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def download_data():
    """Download and extract UCI Student Performance CSVs."""
    _ensure_dirs()
    mat_path = os.path.join(RAW_DIR, "student-mat.csv")
    por_path = os.path.join(RAW_DIR, "student-por.csv")
    if os.path.exists(mat_path) and os.path.exists(por_path):
        print("Raw data already present, skipping download.")
        return mat_path, por_path

    zip_path = os.path.join(RAW_DIR, "student.zip")
    print("Downloading UCI Student Performance dataset...")
    urllib.request.urlretrieve(URLS["student-mat.csv"], zip_path)

    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("student-mat.csv") or name.endswith("student-por.csv"):
                # Extract to raw dir with flat name
                target = os.path.join(RAW_DIR, os.path.basename(name))
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    os.remove(zip_path)
    print(f"Extracted to {RAW_DIR}")
    return mat_path, por_path


def load_and_merge():
    """Load both CSVs, add subject column, concatenate, build student key."""
    mat_path, por_path = download_data()

    df_mat = pd.read_csv(mat_path, sep=";")
    df_por = pd.read_csv(por_path, sep=";")

    df_mat["subject"] = "math"
    df_por["subject"] = "por"

    df = pd.concat([df_mat, df_por], ignore_index=True)
    print(f"Combined rows: {len(df)} (math={len(df_mat)}, por={len(df_por)})")

    # Build student_key for group-aware splitting
    df["student_key"] = df[STUDENT_KEY_COLS].astype(str).agg("|".join, axis=1)
    n_unique = df["student_key"].nunique()
    n_overlap = len(df) - n_unique  # approximate overlap count
    # Count keys appearing in both subjects
    keys_mat = set(df_mat.assign(student_key=df_mat[STUDENT_KEY_COLS].astype(str).agg("|".join, axis=1))["student_key"])
    keys_por = set(df_por.assign(student_key=df_por[STUDENT_KEY_COLS].astype(str).agg("|".join, axis=1))["student_key"])
    overlap = keys_mat & keys_por
    print(f"Unique student keys: {n_unique}, students in both subjects: {len(overlap)}")

    return df


def create_label(df):
    """Create the at_risk proxy label: 1 if G3 < 10, else 0.
    This represents academic failure risk (a dropout precursor)."""
    df["at_risk"] = (df["G3"] < 10).astype(int)
    balance = df["at_risk"].value_counts(normalize=True).sort_index()
    print(f"\nClass balance (at_risk):\n  0 (not at risk): {balance.get(0, 0):.1%}\n  1 (at risk):     {balance.get(1, 0):.1%}")
    print(f"  Counts: {df['at_risk'].value_counts().sort_index().to_dict()}")
    return df


def eda_figure(df):
    """Quick EDA: at_risk rate by failures and by absences bucket."""
    _ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: at_risk rate by failures
    risk_by_fail = df.groupby("failures")["at_risk"].mean()
    axes[0].bar(risk_by_fail.index.astype(str), risk_by_fail.values, color="#e74c3c", alpha=0.85)
    axes[0].set_xlabel("Past failures")
    axes[0].set_ylabel("At-risk rate")
    axes[0].set_title("Academic failure risk by past failures")
    for i, v in enumerate(risk_by_fail.values):
        axes[0].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

    # Right: at_risk rate by absences bucket
    bins = [0, 5, 10, 20, 100]
    labels = ["0-5", "6-10", "11-20", "21+"]
    df["absence_bucket"] = pd.cut(df["absences"], bins=bins, labels=labels, right=True)
    risk_by_abs = df.groupby("absence_bucket", observed=False)["at_risk"].mean()
    axes[1].bar(risk_by_abs.index.astype(str), risk_by_abs.values, color="#3498db", alpha=0.85)
    axes[1].set_xlabel("Absences bucket")
    axes[1].set_ylabel("At-risk rate")
    axes[1].set_title("Academic failure risk by absences")
    for i, v in enumerate(risk_by_abs.values):
        axes[1].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_risk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nEDA figure saved to {path}")
    # Drop temp column
    df.drop(columns=["absence_bucket"], inplace=True, errors="ignore")
    return path


def run():
    """Full Phase 1 pipeline."""
    df = load_and_merge()
    df = create_label(df)
    eda_figure(df)

    out_path = os.path.join(PROCESSED_DIR, "student_merged.csv")
    df.to_csv(out_path, index=False)
    print(f"Merged data saved to {out_path} ({len(df)} rows, {len(df.columns)} cols)")
    print("\n--- Phase 1 Status ---")
    print(f"✓ Data loaded & merged: {len(df)} rows")
    print(f"✓ at_risk label created, EDA figure saved")
    print(f"✓ Output: {out_path}")
    return df


if __name__ == "__main__":
    run()
