"""Phase 5: Auto-generated counseling notes.

Primary: Gemini Flash API. Fallback: rule-based templates.
Both produce the same 3-part format: Summary, Actions, Follow-up.
Sensitive attributes (sex, address, Pstatus, Mjob, Fjob) are excluded from recommendations.
"""
import os, json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- Gemini config ---
GEMINI_MODEL = "gemini-2.5-flash"

# Sensitive attributes to exclude from recommendation text
EXCLUDED_ATTRS = {"sex", "address", "Pstatus", "Mjob", "Fjob"}

SYSTEM_PROMPT = """You are a supportive academic counselor assistant. Given a student risk assessment, 
generate a concise counseling note for a human counselor to review.

RULES:
- Use ONLY the supplied risk drivers and protective factors. Do NOT invent any information.
- Output exactly 3 parts:
  1. **Summary**: One sentence describing the student's risk level and key concern.
  2. **Actions**: 2-3 concrete, actionable steps the counselor can take.
  3. **Follow-up**: A specific timeline for the next check-in.
- NEVER make medical or psychological diagnoses.
- NEVER base advice on sex, address type, or parental background.
- Use a supportive, non-stigmatizing tone.
- This is advice for a human counselor, not a verdict on the student.
- Refer to the student as "this student" or "the student", never by identifiers.
"""


def _build_prompt(payload):
    """Build the user prompt from the payload dict."""
    lines = [
        f"Risk Score: {payload['risk_score']:.2f} ({payload['tier']} tier)",
        "",
        "Risk Drivers (features increasing risk):",
    ]
    for d in payload.get("drivers", []):
        if d.get("feature") not in EXCLUDED_ATTRS:
            lines.append(f"  - {d['display_name']}: {d['direction']} (value context: {d.get('value', 'n/a')})")

    lines.append("")
    lines.append("Protective Factors (features decreasing risk):")
    for p in payload.get("protective", []):
        if p.get("feature") not in EXCLUDED_ATTRS:
            lines.append(f"  - {p['display_name']}: {p['direction']} (value context: {p.get('value', 'n/a')})")

    if not payload.get("protective"):
        lines.append("  - None identified")

    lines.append("")
    lines.append("Generate the counseling note with exactly 3 parts: Summary, Actions, Follow-up.")
    return "\n".join(lines)


def generate_note_llm(payload):
    """Generate a note using Gemini API (google-genai SDK)."""
    from google import genai
    from google.genai import types
    # Re-read key every call so dashboard picks it up without restart
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(payload)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text.strip()


def generate_note_fallback(payload):
    """Rule-based fallback producing the same 3-part format."""
    tier = payload["tier"]
    score = payload["risk_score"]
    drivers = payload.get("drivers", [])
    protective = payload.get("protective", [])

    # Extract feature names for rule matching
    driver_features = {d.get("feature", ""): d for d in drivers}
    driver_values = {d.get("feature", ""): d.get("value", 0) for d in drivers}

    # --- Summary ---
    if tier == "High":
        summary = (f"**Summary**: This student shows a high academic failure risk "
                   f"(score: {score:.2f}). Immediate support is recommended.")
    elif tier == "Medium":
        summary = (f"**Summary**: This student shows a moderate academic failure risk "
                   f"(score: {score:.2f}). Proactive monitoring and targeted support advised.")
    else:
        summary = (f"**Summary**: This student shows a low academic failure risk "
                   f"(score: {score:.2f}). Continue current support and monitor periodically.")

    # --- Actions ---
    actions = ["**Actions**:"]
    action_items = []

    # Rule-based action mapping (checking raw feature values from drivers)
    for d in drivers:
        feat = d.get("feature", "")
        val = d.get("value", 0)
        if feat == "failures" and val >= 2:
            action_items.append("Refer to remedial tutoring programme for subjects with prior failures.")
        elif feat == "absences" and val > 10:
            action_items.append("Schedule an attendance check-in within the next 7 days to understand barriers.")
        elif feat == "studytime" and val <= 1:
            action_items.append("Co-create a weekly study plan with the student, targeting at least 5 hours/week.")
        elif feat == "alc_total" and val >= 7:
            action_items.append("Initiate a supportive wellbeing conversation about lifestyle and coping strategies.")
        elif feat == "goout" and val >= 4:
            action_items.append("Have a supportive conversation about time management and social-academic balance.")
        elif feat == "absence_bucket" and val >= 2:
            action_items.append("Review attendance patterns and identify any systemic barriers to attendance.")

    # Add protective-based actions
    for p in protective:
        feat = p.get("feature", "")
        if feat == "higher":
            action_items.append("Leverage the student's motivation for higher education in goal-setting sessions.")
        elif feat == "famrel":
            action_items.append("Consider involving family support networks in the intervention plan.")

    # Ensure at least 2 actions
    if len(action_items) < 2:
        if tier in ("High", "Medium"):
            action_items.append("Schedule a one-on-one meeting to discuss academic progress and goals.")
            action_items.append("Connect with subject teachers to identify specific areas needing support.")
        else:
            action_items.append("Continue monitoring academic progress in regular check-ins.")
            action_items.append("Encourage participation in available support programmes.")

    # Keep 2-3 actions
    action_items = action_items[:3]
    for i, a in enumerate(action_items, 1):
        actions.append(f"  {i}. {a}")

    # --- Follow-up ---
    if tier == "High":
        followup = "**Follow-up**: Schedule next check-in within 1 week. Reassess risk after mid-term results."
    elif tier == "Medium":
        followup = "**Follow-up**: Schedule next check-in within 2 weeks. Review progress on action items."
    else:
        followup = "**Follow-up**: Schedule next check-in within 4 weeks. Monitor any changes in engagement."

    return f"{summary}\n\n" + "\n".join(actions) + f"\n\n{followup}"


def generate_note(payload):
    """Generate a counseling note. Uses LLM if available, otherwise fallback.
    Returns (note_text, source) where source is 'llm' or 'fallback'."""
    # Re-read key dynamically so a running server picks it up without restart
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your-key-here":
        try:
            note = generate_note_llm(payload)
            return note, "llm"
        except Exception as e:
            print(f"  LLM error: {e}. Using fallback.")
    return generate_note_fallback(payload), "fallback"


def generate_sample_notes():
    """Generate 8-10 sample notes from the test set (Model A) and save to reports/sample_notes.md."""
    import pandas as pd
    from src.explain import explain_student
    from src.train import risk_tier

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")

    test_df = pd.read_csv(os.path.join(models_dir, "test_set_A.csv"))

    # Select a mix of tiers
    high = test_df[test_df["risk_tier"] == "High"]
    medium = test_df[test_df["risk_tier"] == "Medium"]
    low = test_df[test_df["risk_tier"] == "Low"]

    samples = []

    # 3 High
    if len(high) >= 3:
        samples.extend(high.nlargest(3, "risk_score").index.tolist())
    else:
        samples.extend(high.index.tolist())

    # 2-3 Medium (including borderline)
    if len(medium) >= 2:
        # One borderline (closest to 0.3 boundary)
        medium_sorted = medium.iloc[(medium["risk_score"] - 0.3).abs().argsort()]
        samples.append(medium_sorted.index[0])
        # One more medium
        remaining = medium.drop(medium_sorted.index[0])
        if len(remaining) > 0:
            samples.append(remaining.index[0])

    # 2 Low
    if len(low) >= 2:
        samples.extend(low.nsmallest(2, "risk_score").index.tolist())
    elif len(low) >= 1:
        samples.extend(low.index.tolist())

    # 1 false positive: predicted at_risk but actual G3 >= 10
    fp = test_df[(test_df["predicted_at_risk"] == 1) & (test_df["at_risk"] == 0)]
    if len(fp) > 0:
        samples.append(fp.index[0])

    # Deduplicate while preserving order
    seen = set()
    unique_samples = []
    for s in samples:
        if s not in seen:
            seen.add(s)
            unique_samples.append(s)
    samples = unique_samples[:10]

    # Generate notes
    lines = ["# Sample Counseling Notes\n"]
    lines.append(f"Generated from Model A (Early Warning) test set using "
                 f"{'Gemini LLM' if GEMINI_API_KEY and GEMINI_API_KEY != 'your-key-here' else 'rule-based fallback'}.\n")
    lines.append("---\n")

    for i, idx in enumerate(samples, 1):
        row = test_df.loc[[idx]]
        risk_score = float(row["risk_score"].values[0])
        tier = row["risk_tier"].values[0]
        actual_g3 = int(row["G3"].values[0])
        actual_risk = int(row["at_risk"].values[0])
        predicted = int(row["predicted_at_risk"].values[0])

        # SHAP explanation
        explanation = explain_student(row, "A")

        # Check if false positive
        is_fp = predicted == 1 and actual_risk == 0
        label = ""
        if is_fp:
            label = f" ⚠️ **FALSE POSITIVE** (actual G3={actual_g3}, not at risk)"

        # Check if borderline
        is_borderline = abs(risk_score - 0.3) < 0.05 or abs(risk_score - 0.6) < 0.05
        if is_borderline and not is_fp:
            label = f" 🔶 **BORDERLINE CASE** (actual G3={actual_g3})"

        payload = {
            "risk_score": risk_score,
            "tier": tier,
            "drivers": explanation["drivers"],
            "protective": explanation["protective"],
        }

        note, source = generate_note(payload)

        lines.append(f"## Student {i}{label}\n")
        lines.append(f"- **Risk Score**: {risk_score:.3f}")
        lines.append(f"- **Tier**: {tier}")
        lines.append(f"- **Actual G3**: {actual_g3} | **At Risk**: {'Yes' if actual_risk else 'No'}")
        lines.append(f"- **Note Source**: {source}\n")

        lines.append("**Top Risk Drivers:**\n")
        for d in explanation["drivers"]:
            lines.append(f"- {d['display_name']}: {d['direction']} (SHAP={d['shap_value']:.3f})")
        lines.append("")

        if explanation["protective"]:
            lines.append("**Protective Factors:**\n")
            for p in explanation["protective"]:
                lines.append(f"- {p['display_name']}: {p['direction']} (SHAP={p['shap_value']:.3f})")
            lines.append("")

        lines.append("### Counseling Note\n")
        lines.append(note)
        lines.append("\n---\n")

    out_path = os.path.join(reports_dir, "sample_notes.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Sample notes saved to {out_path} ({len(samples)} notes)")
    return out_path


def run():
    path = generate_sample_notes()
    print("\n--- Phase 5 Status ---")
    print(f"✓ Generated sample counseling notes")
    print(f"✓ Output: {path}")
    print(f"✓ Note source: {'Gemini LLM' if GEMINI_API_KEY and GEMINI_API_KEY != 'your-key-here' else 'Rule-based fallback'}")


if __name__ == "__main__":
    run()
