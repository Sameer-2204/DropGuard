"""DropGuard — Run the full pipeline.

Usage:
    python -m src          # Run all phases (1-5, 7)
    python -m src --train  # Phases 1-3 only (data + train)
    python -m src --explain # Phase 4 only (SHAP)
    python -m src --notes  # Phase 5 only (counseling notes)
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="DropGuard pipeline")
    parser.add_argument("--train", action="store_true", help="Run phases 1-3 (data, features, train)")
    parser.add_argument("--explain", action="store_true", help="Run phase 4 (SHAP explanations)")
    parser.add_argument("--notes", action="store_true", help="Run phase 5 (counseling notes)")
    args = parser.parse_args()

    run_all = not (args.train or args.explain or args.notes)

    if run_all or args.train:
        print("=" * 60)
        print("Phase 1: Data Ingestion & EDA")
        print("=" * 60)
        from src.data import run as run_data
        run_data()

        print("\n" + "=" * 60)
        print("Phase 2: Feature Engineering")
        print("=" * 60)
        from src.features import run as run_features
        run_features()

        print("\n" + "=" * 60)
        print("Phase 3: Training & Evaluation (+ Phase 7: UCI 697)")
        print("=" * 60)
        from src.train import run as run_train
        run_train()

    if run_all or args.explain:
        print("\n" + "=" * 60)
        print("Phase 4: SHAP Explanations")
        print("=" * 60)
        from src.explain import run as run_explain
        run_explain()

    if run_all or args.notes:
        print("\n" + "=" * 60)
        print("Phase 5: Counseling Notes")
        print("=" * 60)
        from src.notes import run as run_notes
        run_notes()

    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("Run the dashboard: streamlit run app/dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

