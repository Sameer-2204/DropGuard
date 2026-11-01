# Sample Counseling Notes

Generated from Model A (Early Warning) test set using rule-based fallback.

---

## Student 1

- **Risk Score**: 0.799
- **Tier**: High
- **Actual G3**: 4 | **At Risk**: Yes
- **Note Source**: fallback

**Top Risk Drivers:**

- Past failures: increases risk (SHAP=1.902)
- Paid classes: increases risk (SHAP=1.193)
- Absence level: increases risk (SHAP=0.852)

**Protective Factors:**

- reason_reputation: decreases risk (SHAP=-0.310)
- Fjob_other: decreases risk (SHAP=-0.162)

### Counseling Note

**Summary**: This student shows a high academic failure risk (score: 0.80). Immediate support is recommended.

**Actions**:
  1. Refer to remedial tutoring programme for subjects with prior failures.
  2. Review attendance patterns and identify any systemic barriers to attendance.

**Follow-up**: Schedule next check-in within 1 week. Reassess risk after mid-term results.

---

## Student 2

- **Risk Score**: 0.752
- **Tier**: High
- **Actual G3**: 7 | **At Risk**: Yes
- **Note Source**: fallback

**Top Risk Drivers:**

- Past failures: increases risk (SHAP=2.914)
- Age: increases risk (SHAP=0.462)
- school_MS: increases risk (SHAP=0.459)

**Protective Factors:**

- Attended nursery: decreases risk (SHAP=-0.300)
- Paid classes: decreases risk (SHAP=-0.280)

### Counseling Note

**Summary**: This student shows a high academic failure risk (score: 0.75). Immediate support is recommended.

**Actions**:
  1. Refer to remedial tutoring programme for subjects with prior failures.
  2. Schedule a one-on-one meeting to discuss academic progress and goals.
  3. Connect with subject teachers to identify specific areas needing support.

**Follow-up**: Schedule next check-in within 1 week. Reassess risk after mid-term results.

---

## Student 3 ⚠️ **FALSE POSITIVE** (actual G3=10, not at risk)

- **Risk Score**: 0.707
- **Tier**: High
- **Actual G3**: 10 | **At Risk**: No
- **Note Source**: fallback

**Top Risk Drivers:**

- Past failures: increases risk (SHAP=2.914)
- Paid classes: increases risk (SHAP=1.193)
- Support index (school+family+paid): increases risk (SHAP=0.414)

**Protective Factors:**

- School support: decreases risk (SHAP=-0.675)
- Mjob_at_home: decreases risk (SHAP=-0.294)

### Counseling Note

**Summary**: This student shows a high academic failure risk (score: 0.71). Immediate support is recommended.

**Actions**:
  1. Refer to remedial tutoring programme for subjects with prior failures.
  2. Schedule a one-on-one meeting to discuss academic progress and goals.
  3. Connect with subject teachers to identify specific areas needing support.

**Follow-up**: Schedule next check-in within 1 week. Reassess risk after mid-term results.

---

## Student 4 🔶 **BORDERLINE CASE** (actual G3=0)

- **Risk Score**: 0.302
- **Tier**: Medium
- **Actual G3**: 0 | **At Risk**: Yes
- **Note Source**: fallback

**Top Risk Drivers:**

- Paid classes: increases risk (SHAP=1.193)
- Past failures: increases risk (SHAP=0.890)
- school_MS: increases risk (SHAP=0.459)

**Protective Factors:**

- Fjob_other: decreases risk (SHAP=-0.162)
- Absence level: decreases risk (SHAP=-0.150)

### Counseling Note

**Summary**: This student shows a moderate academic failure risk (score: 0.30). Proactive monitoring and targeted support advised.

**Actions**:
  1. Schedule a one-on-one meeting to discuss academic progress and goals.
  2. Connect with subject teachers to identify specific areas needing support.

**Follow-up**: Schedule next check-in within 2 weeks. Review progress on action items.

---

## Student 5 🔶 **BORDERLINE CASE** (actual G3=9)

- **Risk Score**: 0.313
- **Tier**: Medium
- **Actual G3**: 9 | **At Risk**: Yes
- **Note Source**: fallback

**Top Risk Drivers:**

- Past failures: increases risk (SHAP=0.890)
- Absence level: increases risk (SHAP=0.518)
- Family support: increases risk (SHAP=0.257)

**Protective Factors:**

- School support: decreases risk (SHAP=-0.675)
- Fjob_at_home: decreases risk (SHAP=-0.560)

### Counseling Note

**Summary**: This student shows a moderate academic failure risk (score: 0.31). Proactive monitoring and targeted support advised.

**Actions**:
  1. Review attendance patterns and identify any systemic barriers to attendance.
  2. Schedule a one-on-one meeting to discuss academic progress and goals.
  3. Connect with subject teachers to identify specific areas needing support.

**Follow-up**: Schedule next check-in within 2 weeks. Review progress on action items.

---

## Student 6

- **Risk Score**: 0.098
- **Tier**: Low
- **Actual G3**: 10 | **At Risk**: No
- **Note Source**: fallback

**Top Risk Drivers:**

- Support index (school+family+paid): increases risk (SHAP=0.414)
- Absence level: increases risk (SHAP=0.184)
- guardian_other: increases risk (SHAP=0.160)

**Protective Factors:**

- School support: decreases risk (SHAP=-0.675)
- Travel time: decreases risk (SHAP=-0.426)

### Counseling Note

**Summary**: This student shows a low academic failure risk (score: 0.10). Continue current support and monitor periodically.

**Actions**:
  1. Continue monitoring academic progress in regular check-ins.
  2. Encourage participation in available support programmes.

**Follow-up**: Schedule next check-in within 4 weeks. Monitor any changes in engagement.

---

## Student 7

- **Risk Score**: 0.098
- **Tier**: Low
- **Actual G3**: 14 | **At Risk**: No
- **Note Source**: fallback

**Top Risk Drivers:**

- Family support: increases risk (SHAP=0.257)
- Family relationship quality: increases risk (SHAP=0.076)
- Mjob_at_home: increases risk (SHAP=0.069)

**Protective Factors:**

- reason_reputation: decreases risk (SHAP=-0.310)
- Attended nursery: decreases risk (SHAP=-0.300)

### Counseling Note

**Summary**: This student shows a low academic failure risk (score: 0.10). Continue current support and monitor periodically.

**Actions**:
  1. Continue monitoring academic progress in regular check-ins.
  2. Encourage participation in available support programmes.

**Follow-up**: Schedule next check-in within 4 weeks. Monitor any changes in engagement.

---
