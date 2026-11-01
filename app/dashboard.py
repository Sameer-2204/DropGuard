"""DropGuard – Streamlit Dashboard.

Displays student risk table, SHAP explanations, counseling notes, and what-if panel.
"""
import os, sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features import engineer_features, prepare_data, build_preprocessor, get_feature_names
from src.explain import explain_student, explain_student_chart, FEATURE_DISPLAY
from src.notes import generate_note
from src.train import risk_tier

MODELS_DIR = os.path.join(ROOT, "models")

# --- Page config ---
st.set_page_config(page_title="DropGuard", page_icon="🛡️", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
.tier-high { background-color: #e74c3c; color: white; padding: 2px 10px; border-radius: 10px; font-weight: bold; }
.tier-medium { background-color: #f39c12; color: white; padding: 2px 10px; border-radius: 10px; font-weight: bold; }
.tier-low { background-color: #27ae60; color: white; padding: 2px 10px; border-radius: 10px; font-weight: bold; }
.footer { text-align: center; color: #888; font-size: 0.85em; padding: 20px; border-top: 1px solid #ddd; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)


def tier_badge(tier):
    css = f"tier-{tier.lower()}"
    return f'<span class="{css}">{tier}</span>'


@st.cache_resource
def load_model(variant):
    return joblib.load(os.path.join(MODELS_DIR, f"model_{variant}.joblib"))


@st.cache_data
def load_test_data(variant):
    return pd.read_csv(os.path.join(MODELS_DIR, f"test_set_{variant}.csv"))


def main():
    st.title("🛡️ DropGuard — Academic Failure Risk Dashboard")
    st.caption("Decision-support tool for identifying students at risk of academic failure (a dropout precursor)")

    # --- Sidebar ---
    st.sidebar.header("Settings")
    variant = st.sidebar.radio("Model", ["A (Early Warning)", "B (Mid-Term)"],
                                index=0)
    variant_key = "A" if "A" in variant else "B"

    model = load_model(variant_key)
    test_df = load_test_data(variant_key)

    # Tier filter
    tier_filter = st.sidebar.multiselect("Filter by Tier", ["High", "Medium", "Low"],
                                          default=["High", "Medium", "Low"])

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Model {variant_key}**: {'No grade features' if variant_key == 'A' else 'Includes G1, G2, grade_trend'}")

    # --- Filter and sort ---
    filtered = test_df[test_df["risk_tier"].isin(tier_filter)].sort_values("risk_score", ascending=False)

    # --- Layout: two columns ---
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📋 Students")
        st.caption(f"Showing {len(filtered)} of {len(test_df)} students")

        # Build display table
        display_df = filtered[["risk_score", "risk_tier", "G3", "failures", "absences"]].copy()
        display_df.index = range(1, len(display_df) + 1)
        display_df.columns = ["Risk Score", "Tier", "G3", "Failures", "Absences"]
        display_df["Risk Score"] = display_df["Risk Score"].round(3)

        # Select student
        if len(filtered) > 0:
            selected_idx = st.selectbox(
                "Select student (by row)",
                options=filtered.index.tolist(),
                format_func=lambda x: (
                    f"#{filtered.index.tolist().index(x)+1} — "
                    f"Risk: {filtered.loc[x, 'risk_score']:.3f} "
                    f"({filtered.loc[x, 'risk_tier']})"
                )
            )

            # Show table with tier badges
            for _, row in display_df.iterrows():
                pass  # streamlit dataframe is enough
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.warning("No students match the selected filters.")
            selected_idx = None

    with col_right:
        if selected_idx is not None:
            student = filtered.loc[[selected_idx]]
            score = float(student["risk_score"].values[0])
            tier = student["risk_tier"].values[0]

            st.subheader("📊 Student Detail")

            # Risk score and tier
            met1, met2, met3 = st.columns(3)
            met1.metric("Risk Score", f"{score:.3f}")
            met2.markdown(f"**Tier**: {tier_badge(tier)}", unsafe_allow_html=True)
            met3.metric("Actual G3", int(student["G3"].values[0]))

            # SHAP chart
            st.markdown("#### Feature Contributions (SHAP)")
            fig = explain_student_chart(student, variant_key)
            st.pyplot(fig)
            plt.close(fig)

            # Explanation details
            explanation = explain_student(student, variant_key)
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("**🔴 Risk Drivers**")
                for d in explanation["drivers"]:
                    st.markdown(f"- {d['display_name']}: SHAP={d['shap_value']:.3f}")
            with exp_col2:
                st.markdown("**🟢 Protective Factors**")
                for p in explanation["protective"]:
                    st.markdown(f"- {p['display_name']}: SHAP={p['shap_value']:.3f}")

            # Counseling note
            st.markdown("---")
            st.markdown("#### 📝 Counseling Note")
            payload = {
                "risk_score": score,
                "tier": tier,
                "drivers": explanation["drivers"],
                "protective": explanation["protective"],
            }
            note, source = generate_note(payload)
            st.info(f"Source: **{source.upper()}**")
            st.markdown(note)

            # --- What-If Panel ---
            st.markdown("---")
            st.markdown("#### 🔮 What-If Analysis")
            st.caption("Adjust values to see how risk score changes")

            # Use keys tied to selected student so sliders reset properly
            # when a different student is selected
            student_id = int(selected_idx)

            wif1, wif2, wif3 = st.columns(3)
            with wif1:
                new_absences = st.slider("Absences", 0, 75,
                                          int(student["absences"].values[0]),
                                          key=f"wif_abs_{student_id}")
            with wif2:
                new_studytime = st.slider("Study Time (hrs/week)", 1, 4,
                                           int(student["studytime"].values[0]),
                                           key=f"wif_study_{student_id}")
            with wif3:
                new_failures = st.slider("Past Failures", 0, 4,
                                          int(student["failures"].values[0]),
                                          key=f"wif_fail_{student_id}")

            # Re-predict with modified values
            modified = student.copy()
            modified.loc[:, "absences"] = new_absences
            modified.loc[:, "studytime"] = new_studytime
            modified.loc[:, "failures"] = new_failures

            X_mod, _, _ = prepare_data(modified, variant_key)
            new_prob = float(model.predict_proba(X_mod)[:, 1][0])
            new_tier = risk_tier(new_prob)
            delta = new_prob - score

            wm1, wm2, wm3 = st.columns(3)
            wm1.metric("New Risk Score", f"{new_prob:.3f}", f"{delta:+.3f}")
            wm2.markdown(f"**New Tier**: {tier_badge(new_tier)}", unsafe_allow_html=True)
            wm3.metric("Change", f"{delta:+.3f}",
                       "Higher risk" if delta > 0 else "Lower risk" if delta < 0 else "No change")

    # --- Footer ---
    st.markdown(
        '<div class="footer">⚖️ Decision-support tool; a counselor makes the final call. '
        'This model predicts academic failure risk (a dropout precursor), not dropout itself.</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
