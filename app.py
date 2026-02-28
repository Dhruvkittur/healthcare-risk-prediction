import os, subprocess, sys

# Auto-train models on first run (important for deployment)
if not os.path.exists("models/linear_regression.pkl"):
    with open("train_log.txt", "w") as f:
        result = subprocess.run([sys.executable, "train_models.py"],
                                capture_output=True, text=True)
        f.write(result.stdout + result.stderr)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import generate_synthetic_data, encode_features

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e8f0fe;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #091529 100%);
    border-right: 1px solid rgba(64,196,255,0.15);
}

[data-testid="stSidebar"] .css-1d391kg { padding: 1rem; }

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,40,80,0.9), rgba(10,30,60,0.9));
    border: 1px solid rgba(64,196,255,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(64,196,255,0.5);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #40c4ff, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem;
    color: #90a4c8;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, rgba(15,40,80,0.95) 0%, rgba(5,20,50,0.95) 100%);
    border: 1px solid rgba(64,196,255,0.25);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(64,196,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #40c4ff 0%, #00e5ff 50%, #80d8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    color: #90c4e8;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto 1.5rem;
}

/* Team card */
.team-card {
    background: rgba(15,40,80,0.6);
    border: 1px solid rgba(64,196,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.team-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #40c4ff;
}
.team-role { font-size: 0.8rem; color: #78909c; }

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, rgba(0,100,60,0.3), rgba(0,60,40,0.3));
    border: 1.5px solid rgba(0,230,118,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #00e676;
}

/* Info box */
.info-box {
    background: rgba(64,196,255,0.08);
    border-left: 3px solid #40c4ff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

/* Streamlit overrides */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: rgba(15,40,80,0.8) !important;
    border-color: rgba(64,196,255,0.3) !important;
    color: #e8f0fe !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0077b6, #0096c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0096c7, #00b4d8) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,150,199,0.4) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,25,50,0.8) !important;
    border-radius: 10px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #90a4c8 !important;
    font-family: 'Syne', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(64,196,255,0.15) !important;
    color: #40c4ff !important;
}
hr { border-color: rgba(64,196,255,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    models = {}
    if os.path.exists(MODEL_DIR):
        for name in ['linear_regression', 'decision_tree', 'knn', 'kmeans']:
            p = f"{MODEL_DIR}/{name}.pkl"
            if os.path.exists(p):
                models[name] = joblib.load(p)
        for name in ['scaler', 'full_scaler', 'feat_cols', 'cluster_scaler',
                      'feature_encoders', 'risk_category_le', 'metrics']:
            p = f"{MODEL_DIR}/{name}.pkl"
            if os.path.exists(p):
                models[name] = joblib.load(p)
    return models


@st.cache_data
def load_dataset():
    p = f"{MODEL_DIR}/dataset_with_clusters.csv"
    if os.path.exists(p):
        return pd.read_csv(p)
    return generate_synthetic_data(1500)


def models_ready(models):
    return all(k in models for k in ['linear_regression', 'decision_tree', 'knn', 'kmeans', 'scaler'])


FEATURES = ['Age', 'Gender', 'BMI', 'Blood_Pressure', 'Cholesterol',
            'Glucose', 'Smoking', 'Physical_Activity', 'Family_History', 'Previous_Visits']

plotly_template = dict(
    layout=go.Layout(
        paper_bgcolor='rgba(5,20,50,0.0)',
        plot_bgcolor='rgba(5,20,50,0.4)',
        font=dict(color='#90c4e8', family='DM Sans'),
        xaxis=dict(gridcolor='rgba(64,196,255,0.1)', linecolor='rgba(64,196,255,0.2)'),
        yaxis=dict(gridcolor='rgba(64,196,255,0.1)', linecolor='rgba(64,196,255,0.2)'),
    )
)


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2.5rem;">🏥</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800;
                    background:linear-gradient(135deg,#40c4ff,#00e5ff);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            HealthPredict AI
        </div>
        <div style="color:#78909c; font-size:0.75rem;">Smart Risk Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "⚙️ Data Generator", "🔮 Predictions", "📊 Visualizations", "📈 Model Metrics"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("<div style='color:#78909c;font-size:0.75rem;text-align:center;'>v1.0.0 · Healthcare AI</div>",
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Smart Healthcare<br>Risk Prediction</div>
        <div class="hero-sub">
            AI-powered patient segmentation & predictive analytics for
            proactive clinical decision-making
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">ML Algorithms</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">1500+</div>
            <div class="metric-label">Patient Records</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Feature Variables</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Risk Segments</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    <div class="info-box">
    This system leverages machine learning to transform raw patient data into actionable clinical insights. 
    Four distinct algorithms tackle different prediction tasks — from estimating individual medical costs 
    to clustering entire populations into risk segments for targeted intervention.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🤖 Algorithms Used")
        alg_data = {
            "Algorithm": ["Linear Regression", "Decision Tree", "K-Nearest Neighbors", "K-Means Clustering"],
            "Task": ["Predict Medical Expenses", "Classify Disease Presence", "Predict Risk Category", "Segment Patient Populations"],
            "Output": ["Continuous (₹)", "Binary (0/1)", "4 Risk Classes", "4 Patient Clusters"]
        }
        st.dataframe(pd.DataFrame(alg_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 🧬 Input Features")
        feat_data = {
            "Feature": ["Age", "Gender", "BMI", "Blood Pressure", "Cholesterol",
                        "Glucose Level", "Smoking Habit", "Physical Activity",
                        "Family History", "Previous Visits"],
            "Type": ["Numeric", "Categorical", "Numeric", "Numeric", "Numeric",
                     "Numeric", "Binary", "Categorical", "Binary", "Numeric"]
        }
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 👥 Team Details")
    team = [
        ("Dhruv K", "ML Engineer"),
        ("Pramodini G", "Data Scientist"),
        ("Rahul S", "Backend Developer"),
        ("Ranjita M", "UI/UX & Deployment"),
    ]
    cols = st.columns(4)
    for col, (name, role) in zip(cols, team):
        with col:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size:2rem;">👤</div>
                <div class="team-name">{name}</div>
                <div class="team-role">{role}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA GENERATOR
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Data Generator":
    st.markdown("## ⚙️ Synthetic Data Generator")
    st.markdown("""
    <div class="info-box">
    Generate realistic synthetic patient records with statistically correlated features.
    Data is created using domain-informed distributions that mimic real clinical populations.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        n = st.slider("Number of records to generate", 100, 5000, 1500, 100)
    with col2:
        seed = st.number_input("Random seed", value=42, step=1)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button("🔄 Generate Dataset", use_container_width=True)

    if gen_btn or True:
        with st.spinner("Generating synthetic patient data…"):
            df = generate_synthetic_data(n, int(seed))

        st.success(f"✅ Generated **{len(df):,}** patient records with **{len(df.columns)}** features")
        st.markdown("#### 👁️ Sample Records")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="metric-card">
                <div class="metric-value">{:.0f}</div>
                <div class="metric-label">Avg Age</div>
            </div>""".format(df['Age'].mean()), unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card">
                <div class="metric-value">{:.1f}</div>
                <div class="metric-label">Avg BMI</div>
            </div>""".format(df['BMI'].mean()), unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card">
                <div class="metric-value">{:.1%}</div>
                <div class="metric-label">Disease Rate</div>
            </div>""".format(df['Disease_Presence'].mean()), unsafe_allow_html=True)

        st.markdown("#### 📊 Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Age', color='Gender',
                               title='Age Distribution by Gender',
                               template='plotly_dark',
                               color_discrete_map={'Male': '#40c4ff', 'Female': '#f48fb1'})
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            rc_counts = df['Risk_Category'].value_counts()
            fig = px.pie(values=rc_counts.values, names=rc_counts.index,
                         title='Risk Category Distribution',
                         template='plotly_dark',
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "synthetic_healthcare_data.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predictions":
    st.markdown("## 🔮 Patient Prediction Interface")

    models = load_models()
    if not models_ready(models):
        st.warning("⚠️ Models not trained yet. Run `python train_models.py` first, then refresh.")
        st.stop()

    st.markdown("""<div class="info-box">
    Enter patient clinical data below. All four models will run simultaneously
    to produce a comprehensive risk assessment.
    </div>""", unsafe_allow_html=True)

    with st.form("patient_form"):
        st.markdown("### 🧑‍⚕️ Patient Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age     = st.number_input("Age", 18, 90, 45)
            gender  = st.selectbox("Gender", ["Male", "Female"])
            bmi     = st.number_input("BMI", 15.0, 50.0, 26.5, 0.1)
        with c2:
            bp      = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
            chol    = st.number_input("Cholesterol (mg/dL)", 100, 320, 190)
            glucose = st.number_input("Glucose Level (mg/dL)", 60, 250, 95)
        with c3:
            smoking  = st.selectbox("Smoking Habit", ["No", "Yes"])
            activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
            fhist    = st.selectbox("Family Medical History", ["No", "Yes"])
            prev_v   = st.number_input("Previous Hospital Visits", 0, 15, 2)

        submitted = st.form_submit_button("🚀 Run Predictions", use_container_width=True)

    if submitted:
        enc = models['feature_encoders']
        gender_enc   = enc['Gender'].transform([gender])[0]
        smoking_enc  = enc['Smoking'].transform([smoking])[0]
        activity_enc = enc['Physical_Activity'].transform([activity])[0]
        fhist_enc    = enc['Family_History'].transform([fhist])[0]

        base_row = np.array([[age, gender_enc, bmi, bp, chol, glucose,
                              smoking_enc, activity_enc, fhist_enc, prev_v]])
        # Build engineered feature row matching training pipeline
        row_df = pd.DataFrame([{
            'Age': age, 'Gender': gender_enc, 'BMI': bmi,
            'Blood_Pressure': bp, 'Cholesterol': chol, 'Glucose': glucose,
            'Smoking': smoking_enc, 'Physical_Activity': activity_enc,
            'Family_History': fhist_enc, 'Previous_Visits': prev_v
        }])
        row_df['Age_BMI']          = row_df['Age'] * row_df['BMI']
        row_df['BP_Chol']          = row_df['Blood_Pressure'] * row_df['Cholesterol']
        row_df['Glucose_BMI']      = row_df['Glucose'] * row_df['BMI']
        row_df['Age_Glucose']      = row_df['Age'] * row_df['Glucose']
        row_df['Smoke_Age']        = row_df['Smoking'] * row_df['Age']
        row_df['Smoke_BMI']        = row_df['Smoking'] * row_df['BMI']
        row_df['Activity_BMI']     = row_df['Physical_Activity'] * row_df['BMI']
        row_df['History_Age']      = row_df['Family_History'] * row_df['Age']
        row_df['History_Chol']     = row_df['Family_History'] * row_df['Cholesterol']
        row_df['Visits_Expenses']  = row_df['Previous_Visits'] * row_df['Age']
        row_df['BP_Age_ratio']     = row_df['Blood_Pressure'] / (row_df['Age'] + 1)
        row_df['Chol_BMI_ratio']   = row_df['Cholesterol'] / (row_df['BMI'] + 1)
        row_df['Glucose_Age_ratio']= row_df['Glucose'] / (row_df['Age'] + 1)
        feat_cols = models.get('feat_cols', list(row_df.columns))
        row_feat = row_df[feat_cols].values
        row_sc = models['full_scaler'].transform(row_feat)
        # Cluster uses its own scaler on clinical subset
        cluster_cols = ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose',
                        'Previous_Visits', 'Smoking', 'Physical_Activity']
        clust_df = pd.DataFrame([[age, bmi, bp, chol, glucose, prev_v, smoking_enc, activity_enc]],
                                  columns=cluster_cols)
        row_clust = models['cluster_scaler'].transform(clust_df.values)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            expense = models['linear_regression'].predict(row_sc)[0]
            st.markdown(f"""<div class="pred-box">
                <div style="color:#90c4e8;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;">
                    💰 Medical Expenses</div>
                <div class="pred-value">₹{expense:,.0f}</div>
                <div style="color:#78909c;font-size:0.8rem;margin-top:0.5rem;">Annual Estimate</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            disease = models['decision_tree'].predict(row_sc)[0]
            disease_prob = models['decision_tree'].predict_proba(row_sc)[0]
            status = "POSITIVE" if disease == 1 else "NEGATIVE"
            color  = "#ff5252" if disease == 1 else "#00e676"
            st.markdown(f"""<div class="pred-box" style="
                    background:{'linear-gradient(135deg,rgba(100,0,0,0.3),rgba(60,0,0,0.3))' if disease else 'linear-gradient(135deg,rgba(0,100,60,0.3),rgba(0,60,40,0.3))'};
                    border-color:{color}40;">
                <div style="color:#90c4e8;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;">
                    🦠 Disease Presence</div>
                <div class="pred-value" style="color:{color};">{status}</div>
                <div style="color:#78909c;font-size:0.8rem;margin-top:0.5rem;">
                    Confidence: {max(disease_prob):.1%}</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            risk_enc = models['knn'].predict(row_sc)[0]
            risk_proba = models['knn'].predict_proba(row_sc)[0]
            risk_label = models['risk_category_le'].inverse_transform([risk_enc])[0]
            risk_colors = {'Low Risk': '#00e676', 'Medium Risk': '#ffd740',
                           'High Risk': '#ff6d00', 'Critical Risk': '#ff1744'}
            rcolor = risk_colors.get(risk_label, '#40c4ff')
            st.markdown(f"""<div class="pred-box" style="border-color:{rcolor}40;
                    background:linear-gradient(135deg,rgba(30,30,60,0.4),rgba(10,20,50,0.4));">
                <div style="color:#90c4e8;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;">
                    ⚠️ Risk Category</div>
                <div class="pred-value" style="color:{rcolor};font-size:1.6rem;">{risk_label}</div>
                <div style="color:#78909c;font-size:0.8rem;margin-top:0.5rem;">
                    Confidence: {max(risk_proba):.1%}</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            cluster = models['kmeans'].predict(row_clust)[0]
            cluster_names = {0: "Segment A", 1: "Segment B", 2: "Segment C", 3: "Segment D"}
            st.markdown(f"""<div class="pred-box" style="border-color:#40c4ff40;
                    background:linear-gradient(135deg,rgba(0,50,100,0.3),rgba(0,30,80,0.3));">
                <div style="color:#90c4e8;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;">
                    🔵 Patient Cluster</div>
                <div class="pred-value">{cluster_names[cluster]}</div>
                <div style="color:#78909c;font-size:0.8rem;margin-top:0.5rem;">Cluster #{cluster}</div>
            </div>""", unsafe_allow_html=True)

        # Radar chart of patient profile
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📡 Patient Risk Profile")
        categories = ['Age Risk', 'BMI Risk', 'BP Risk', 'Cholesterol Risk', 'Glucose Risk']
        values = [
            min(age / 90, 1),
            min((bmi - 15) / 35, 1),
            min((bp - 80) / 120, 1),
            min((chol - 100) / 220, 1),
            min((glucose - 60) / 190, 1)
        ]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(64,196,255,0.15)',
            line=dict(color='#40c4ff', width=2),
            name='Patient'
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(5,20,50,0.5)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(64,196,255,0.2)'),
                angularaxis=dict(gridcolor='rgba(64,196,255,0.2)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#90c4e8'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualizations":
    st.markdown("## 📊 Exploratory Data Analysis")
    df = load_dataset()

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Distributions", "🔗 Correlations", "🏷️ Clinical Insights", "🔵 Cluster Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='BMI', color='Gender', nbins=40,
                               title='BMI Distribution by Gender',
                               template='plotly_dark',
                               color_discrete_map={'Male':'#40c4ff','Female':'#f48fb1'})
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x='Medical_Expenses', nbins=50,
                               title='Medical Expenses Distribution',
                               template='plotly_dark', color_discrete_sequence=['#00e5ff'])
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='Physical_Activity', y='BMI', color='Smoking',
                         title='BMI by Physical Activity and Smoking',
                         template='plotly_dark',
                         color_discrete_map={'Yes':'#ff5252','No':'#40c4ff'})
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.violin(df, x='Risk_Category', y='Age',
                            title='Age Distribution by Risk Category',
                            template='plotly_dark', color='Risk_Category',
                            color_discrete_sequence=['#00e676','#ffd740','#ff6d00','#ff1744'])
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol',
                    'Glucose', 'Previous_Visits', 'Medical_Expenses']
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', title='Feature Correlation Heatmap',
                        template='plotly_dark',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(**plotly_template['layout'].to_plotly_json(), height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x='Age', y='Medical_Expenses', color='Smoking',
                             title='Age vs Medical Expenses (by Smoking)',
                             template='plotly_dark',
                             color_discrete_map={'Yes':'#ff5252','No':'#40c4ff'})
            z = np.polyfit(df['Age'], df['Medical_Expenses'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['Age'].min(), df['Age'].max(), 100)
            fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines',
                                     line=dict(color='#ffffff', width=2, dash='dash'),
                                     name='Trend'))
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df, x='BMI', y='Cholesterol', color='Physical_Activity',
                             title='BMI vs Cholesterol (by Activity Level)',
                             template='plotly_dark',
                             color_discrete_sequence=['#ff6d00','#40c4ff','#00e676'])
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            dp = df.groupby(['Smoking', 'Family_History'])['Disease_Presence'].mean().reset_index()
            fig = px.bar(dp, x='Smoking', y='Disease_Presence', color='Family_History',
                         barmode='group', title='Disease Rate by Smoking & Family History',
                         template='plotly_dark',
                         color_discrete_map={'Yes':'#ff5252','No':'#40c4ff'})
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            avg = df.groupby('Risk_Category')[['BMI', 'Blood_Pressure', 'Glucose', 'Cholesterol']].mean().reset_index()
            fig = px.bar(avg.melt(id_vars='Risk_Category'), x='Risk_Category', y='value',
                         color='variable', barmode='group',
                         title='Average Clinical Markers by Risk Category',
                         template='plotly_dark')
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)

        # Sunburst
        fig = px.sunburst(df.head(500), path=['Physical_Activity','Smoking','Risk_Category'],
                          title='Risk Pathway: Activity → Smoking → Risk Category',
                          template='plotly_dark',
                          color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(**plotly_template['layout'].to_plotly_json(), height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if 'Cluster' not in df.columns:
            st.warning("Run training first to see cluster analysis.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(df, x='Age', y='BMI', color='Cluster',
                                 title='Patient Clusters: Age vs BMI',
                                 template='plotly_dark',
                                 color_continuous_scale='Viridis',
                                 hover_data=['Blood_Pressure', 'Risk_Category'])
                fig.update_layout(**plotly_template['layout'].to_plotly_json())
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(df, x='Cholesterol', y='Glucose', color='Cluster',
                                 title='Patient Clusters: Cholesterol vs Glucose',
                                 template='plotly_dark', color_continuous_scale='Plasma',
                                 hover_data=['Age','Risk_Category'])
                fig.update_layout(**plotly_template['layout'].to_plotly_json())
                st.plotly_chart(fig, use_container_width=True)

            cluster_summary = df.groupby('Cluster')[
                ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose', 'Medical_Expenses']
            ].mean().round(1).reset_index()
            cluster_summary.columns = ['Cluster', 'Avg Age', 'Avg BMI', 'Avg BP',
                                        'Avg Cholesterol', 'Avg Glucose', 'Avg Expenses']
            st.markdown("#### Cluster Profiles")
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

            fig = px.scatter_3d(df.sample(min(500, len(df))),
                                x='Age', y='BMI', z='Medical_Expenses',
                                color='Cluster', title='3D Cluster Visualization',
                                color_continuous_scale='Viridis',
                                template='plotly_dark')
            fig.update_layout(**plotly_template['layout'].to_plotly_json(), height=500)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL METRICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Metrics":
    st.markdown("## 📈 Model Performance Metrics")
    models = load_models()
    if 'metrics' not in models:
        st.warning("⚠️ No metrics found. Run `python train_models.py` first.")
        st.stop()

    m = models['metrics']

    # Summary scorecard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{m['linear_regression']['R2']:.3f}</div>
            <div class="metric-label">Linear Reg R²</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{m['decision_tree']['Accuracy']:.1%}</div>
            <div class="metric-label">Decision Tree Acc.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{m['knn']['Accuracy']:.1%}</div>
            <div class="metric-label">KNN Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{m['kmeans']['Silhouette']:.3f}</div>
            <div class="metric-label">KMeans Silhouette</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📉 Linear Regression", "🌳 Decision Tree", "🔵 KNN", "⭕ K-Means"])

    with tab1:
        lr = m['linear_regression']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score",  f"{lr['R2']:.4f}")
        c2.metric("MAE",  f"₹{lr['MAE']:,.0f}")
        c3.metric("MSE",  f"{lr['MSE']:,.0f}")
        c4.metric("RMSE", f"₹{lr['RMSE']:,.0f}")

        y_test = np.array(lr['y_test'])
        y_pred = np.array(lr['y_pred'])
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                     marker=dict(color='#40c4ff', opacity=0.5, size=4),
                                     name='Predictions'))
            mn, mx = y_test.min(), y_test.max()
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                     line=dict(color='#ff5252', dash='dash'), name='Perfect'))
            fig.update_layout(title='Actual vs Predicted Expenses',
                              xaxis_title='Actual', yaxis_title='Predicted',
                              template='plotly_dark', **plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            residuals = y_test - y_pred
            fig = px.histogram(x=residuals, nbins=40, title='Residual Distribution',
                               template='plotly_dark', color_discrete_sequence=['#00e5ff'])
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dt = m['decision_tree']
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{dt['Accuracy']:.1%}")
        c2.metric("Precision (macro)", f"{dt['Report']['macro avg']['precision']:.1%}")

        col1, col2 = st.columns(2)
        with col1:
            cm = np.array(dt['Confusion'])
            fig = px.imshow(cm, text_auto=True,
                            title='Confusion Matrix',
                            labels=dict(x='Predicted', y='Actual'),
                            x=['No Disease', 'Disease'],
                            y=['No Disease', 'Disease'],
                            color_continuous_scale='Blues', template='plotly_dark')
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            rep = dt['Report']
            classes = ['0', '1']
            metrics_shown = ['precision', 'recall', 'f1-score']
            rows = {cls: [rep[cls][m_] for m_ in metrics_shown] for cls in classes}
            rep_df = pd.DataFrame(rows, index=metrics_shown).T
            rep_df.index = ['No Disease', 'Disease']
            fig = px.bar(rep_df.reset_index().melt(id_vars='index'),
                         x='index', y='value', color='variable', barmode='group',
                         title='Classification Report',
                         template='plotly_dark',
                         color_discrete_sequence=['#40c4ff', '#00e5ff', '#80d8ff'])
            fig.update_layout(**plotly_template['layout'].to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        knn = m['knn']
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{knn['Accuracy']:.1%}")
        c2.metric("Classes", len(knn['Classes']))

        cm = np.array(knn['Confusion'])
        fig = px.imshow(cm, text_auto=True, title='KNN Confusion Matrix',
                        x=knn['Classes'], y=knn['Classes'],
                        color_continuous_scale='Blues', template='plotly_dark')
        fig.update_layout(**plotly_template['layout'].to_plotly_json(), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        km = m['kmeans']
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{km['Silhouette']:.4f}")
        c2.metric("Inertia",          f"{km['Inertia']:,.0f}")
        c3.metric("# Clusters",       km['n_clusters'])

        st.markdown("""<div class="info-box">
        <b>Silhouette Score</b> ranges from -1 to 1.
        A score closer to 1 indicates well-separated, dense clusters.
        Values above 0.3 indicate reasonable cluster structure.
        </div>""", unsafe_allow_html=True)