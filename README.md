# 🛡️ DropGuard — Student Academic Failure Risk Prediction

A machine learning pipeline that predicts **academic failure risk** (a dropout precursor) using the UCI Student Performance dataset. Generates SHAP-based explanations and auto-generated counseling notes for human counselors.

> **Important**: This model predicts academic failure risk using a proxy label (`G3 < 10`), not actual dropout. See [Limitations](#limitations).

## Dataset & Proxy Label

- **Primary**: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320) — `student-mat.csv` (395 rows, Mathematics) and `student-por.csv` (649 rows, Portuguese), combined into ~1,044 rows.
- **Cross-validation**: [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) (4,424 rows) for transfer validation.
- **Proxy label**: `at_risk = 1 if G3 < 10 else 0` — students scoring below 10 on the final grade. This indicates academic failure risk, a known precursor to dropout, but is **not** a direct dropout label.
- **Student grouping**: ~382 students appear in both subjects. A 13-attribute key (`school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, nursery, internet`) is used as a group ID for `StratifiedGroupKFold` to prevent data leakage.

## Pipeline

```
data/raw/ ──► src/data.py ──► data/processed/
                                    │
                              src/features.py
                              (Model A / Model B)
                                    │
                              src/train.py
                        (CV, calibration, threshold)
                                    │
                    ┌───────────────┼───────────────┐
              src/explain.py   src/notes.py    app/dashboard.py
              (SHAP values)    (LLM/fallback)  (Streamlit UI)
```

## Model A vs Model B

| | Model A (Early Warning) | Model B (Mid-Term) |
|---|---|---|
| **Features** | Demographics, family, behaviour, support (no grades) | + G1, G2, grade_trend |
| **Best Model** | Logistic Regression (balanced) | Logistic Regression (balanced) |
| **CV PR-AUC** | 0.479 ± 0.076 | 0.896 ± varies |
| **Test PR-AUC** | 0.476 | 0.919 |
| **Test ROC-AUC** | 0.710 | 0.972 |
| **Test Recall** | 0.954 (at threshold) | 0.861 (at threshold) |
| **Precision@threshold** | 0.212 | 0.755 |
| **Threshold** | 0.16 | 0.31 |

Model A achieves high recall at the cost of precision (many false positives) since it has no grade information. Model B adds mid-term grades (G1, G2) and dramatically improves precision — useful once mid-term data is available.

## Threshold Choice

The threshold is chosen to achieve **recall ≥ 0.85** on the test set, maximising precision subject to that constraint. This reflects the asymmetric cost: missing an at-risk student (false negative) is worse than flagging a student who is not at risk (false positive).

**Risk tiers**: Low (< 0.3), Medium (0.3–0.6), High (> 0.6).

## Cross-Dataset Validation (UCI 697)

A LightGBM model trained on the UCI 697 dataset (4,424 rows, actual Dropout labels) achieves:
- **PR-AUC**: 0.884, **ROC-AUC**: 0.918, **Recall**: 0.771

Top SHAP drivers: curricular units approved (2nd sem), tuition fees up to date, curricular units approved (1st sem), course type, and age at enrollment. The key insight is that **academic engagement** (units approved, grades) consistently dominates risk prediction across both datasets, confirming the relevance of our "failure-as-precursor" proxy.

## Sample Note

> **Summary**: This student shows a high academic failure risk (score: 0.85). Immediate support is recommended.
>
> **Actions**:
> 1. Refer to remedial tutoring programme for subjects with prior failures.
> 2. Schedule an attendance check-in within the next 7 days to understand barriers.
> 3. Co-create a weekly study plan with the student, targeting at least 5 hours/week.
>
> **Follow-up**: Schedule next check-in within 1 week. Reassess risk after mid-term results.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment (optional, for LLM notes)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Train models (downloads data, trains, evaluates, runs Phase 7 validation)
python -m src.train

# 4. Generate SHAP explanations
python -m src.explain

# 5. Generate sample counseling notes
python -m src.notes

# 6. Launch dashboard
streamlit run app/dashboard.py
```

## Limitations

- **Proxy label**: `G3 < 10` is a proxy for academic failure risk, not actual dropout. The relationship between low grades and dropout is well-documented but imperfect.
- **Small data**: 1,044 rows with ~382 overlapping students across subjects. Results may not generalise to larger populations.
- **Portuguese schools only**: Data comes from two Portuguese secondary schools. Cultural and systemic differences limit transferability.
- **Fairness not audited**: No formal fairness analysis across protected attributes (gender, socioeconomic status). The model excludes sensitive attributes from recommendations but may still encode disparities.
- **Temporal validity**: Data is from a specific time period; student characteristics and educational contexts evolve.

## Ethics

- **Human-in-the-loop**: This is a decision-support tool. A counselor reviews all outputs and makes the final call. The model never labels or categorises students directly.
- **No student labeling**: Risk scores are advisory and should not be shared with students or used for punitive purposes.
- **Supportive framing**: All counseling notes use non-stigmatising language and focus on actionable support, never diagnosis.
- **Sensitive attributes excluded**: Sex, address, parental status, and parental jobs are excluded from the recommendation layer to avoid reinforcing biases.
