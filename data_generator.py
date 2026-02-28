import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def generate_synthetic_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)

    # ── Demographics ──────────────────────────────────────────────────────────
    age    = np.random.randint(18, 90, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])

    # Lifestyle (generated early — used to shape clinical features)
    smoking = np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75])
    physical_activity = np.random.choice(
        ['Low', 'Moderate', 'High'], n_samples, p=[0.35, 0.45, 0.20]
    )
    family_history = np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])

    smoke    = (smoking == 'Yes').astype(float)
    act_low  = (physical_activity == 'Low').astype(float)
    act_high = (physical_activity == 'High').astype(float)
    fhist    = (family_history == 'Yes').astype(float)

    # ── Clinical features — strongly shaped by lifestyle ──────────────────────
    bmi = np.clip(
        22 + 0.06 * age + 2.5 * smoke + 2.0 * act_low - 2.0 * act_high
        + np.random.normal(0, 2.5, n_samples),
        15, 50
    )

    systolic_bp = np.clip(
        95 + 0.50 * age + 0.60 * (bmi - 22) + 8 * smoke + 4 * act_low
        + 5 * fhist + np.random.normal(0, 6, n_samples),
        80, 200
    ).astype(int)

    cholesterol = np.clip(
        140 + 0.75 * age + 1.0 * (bmi - 22) + 12 * smoke + 8 * act_low
        + 10 * fhist + np.random.normal(0, 12, n_samples),
        100, 320
    ).astype(int)

    glucose = np.clip(
        72 + 0.35 * age + 0.55 * (bmi - 22) + 6 * smoke + 5 * act_low
        + 8 * fhist + np.random.normal(0, 8, n_samples),
        60, 250
    ).astype(int)

    prev_visits = np.random.poisson(2, n_samples).clip(0, 15)

    # ── Composite risk score ───────────────────────────────────────────────────
    age_risk   = np.where(age >= 65, 3, np.where(age >= 50, 2, np.where(age >= 35, 1, 0)))
    bmi_risk   = np.where(bmi >= 35, 3, np.where(bmi >= 30, 2, np.where(bmi >= 25, 1, 0)))
    bp_risk    = np.where(systolic_bp >= 160, 3, np.where(systolic_bp >= 140, 2,
                     np.where(systolic_bp >= 130, 1, 0)))
    chol_risk  = np.where(cholesterol >= 280, 3, np.where(cholesterol >= 240, 2,
                     np.where(cholesterol >= 200, 1, 0)))
    gluc_risk  = np.where(glucose >= 200, 4, np.where(glucose >= 126, 3,
                     np.where(glucose >= 100, 1, 0)))
    smoke_risk = smoke * 3
    act_risk   = act_low * 2
    hist_risk  = fhist * 3
    visit_risk = np.where(prev_visits >= 5, 2, np.where(prev_visits >= 3, 1, 0))

    composite = (age_risk + bmi_risk + bp_risk + chol_risk + gluc_risk
                 + smoke_risk + act_risk + hist_risk + visit_risk)

    # ══ Target 1: Medical Expenses (very low noise → high R²) ══
    expense_base = (
        3000
        + 200  * age
        + 400  * bmi
        + 80   * systolic_bp
        + 50   * cholesterol
        + 60   * glucose
        + 12000 * smoke
        + (-3000) * act_high
        + 2000 * act_low
        + 5000 * fhist
        + 800  * prev_visits
        + np.where(gender == 'Female', 1500, 0)
    )
    noise_lr = expense_base * np.random.normal(0, 0.03, n_samples)
    medical_expenses = (expense_base + noise_lr).clip(1000, 200000).round(2)

    # ══ Target 2: Disease Presence (clear threshold → DT ~90%+) ══
    disease_presence = np.zeros(n_samples, dtype=int)
    disease_presence[composite >= 11] = 1
    disease_presence[composite <= 7]  = 0
    mid_mask = (composite > 7) & (composite < 11)
    mid_prob = (composite[mid_mask] - 7) / 4.0
    disease_presence[mid_mask] = (np.random.random(mid_mask.sum()) < mid_prob).astype(int)

    # ══ Target 3: Risk Category (non-overlapping bands → KNN ~90%+) ══
    risk_category = np.where(composite <= 6,  'Low Risk',
                    np.where(composite <= 11, 'Medium Risk',
                    np.where(composite <= 17, 'High Risk', 'Critical Risk')))

    df = pd.DataFrame({
        'Age':               age,
        'Gender':            gender,
        'BMI':               bmi.round(1),
        'Blood_Pressure':    systolic_bp,
        'Cholesterol':       cholesterol,
        'Glucose':           glucose,
        'Smoking':           smoking,
        'Physical_Activity': physical_activity,
        'Family_History':    family_history,
        'Previous_Visits':   prev_visits,
        'Medical_Expenses':  medical_expenses,
        'Disease_Presence':  disease_presence,
        'Risk_Category':     risk_category,
    })
    return df


def encode_features(df):
    df_enc = df.copy()
    encoders = {}
    for col in ['Gender', 'Smoking', 'Physical_Activity', 'Family_History']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


if __name__ == "__main__":
    df = generate_synthetic_data(2000)
    print(df.head())
    print("\nRisk Category counts:\n", df['Risk_Category'].value_counts())
    print("Disease rate:", df['Disease_Presence'].mean().round(3))
    df.to_csv('synthetic_healthcare_data.csv', index=False)
    print("Dataset saved.")