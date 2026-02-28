import numpy as np
import pandas as pd
import joblib, os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score
)
from data_generator import generate_synthetic_data, encode_features

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_FEATURES = [
    'Age', 'Gender', 'BMI', 'Blood_Pressure', 'Cholesterol',
    'Glucose', 'Smoking', 'Physical_Activity', 'Family_History', 'Previous_Visits'
]


def add_engineered_features(df_enc):
    """Add interaction & ratio features that dramatically boost all models."""
    d = df_enc.copy()
    # Interaction terms
    d['Age_BMI']          = d['Age'] * d['BMI']
    d['BP_Chol']          = d['Blood_Pressure'] * d['Cholesterol']
    d['Glucose_BMI']      = d['Glucose'] * d['BMI']
    d['Age_Glucose']      = d['Age'] * d['Glucose']
    d['Smoke_Age']        = d['Smoking'] * d['Age']
    d['Smoke_BMI']        = d['Smoking'] * d['BMI']
    d['Activity_BMI']     = d['Physical_Activity'] * d['BMI']
    d['History_Age']      = d['Family_History'] * d['Age']
    d['History_Chol']     = d['Family_History'] * d['Cholesterol']
    d['Visits_Expenses']  = d['Previous_Visits'] * d['Age']
    # Ratio features
    d['BP_Age_ratio']     = d['Blood_Pressure'] / (d['Age'] + 1)
    d['Chol_BMI_ratio']   = d['Cholesterol'] / (d['BMI'] + 1)
    d['Glucose_Age_ratio']= d['Glucose'] / (d['Age'] + 1)
    return d


def get_feature_cols(df_enc):
    return [c for c in df_enc.columns if c not in
            ['Medical_Expenses', 'Disease_Presence', 'Risk_Category',
             'Risk_Category_enc', 'Cluster']]


def train_all(n_samples=2000):
    print("Generating dataset …")
    df = generate_synthetic_data(n_samples)
    df_enc, encoders = encode_features(df)

    # ── Encode risk category ──
    rc_le = LabelEncoder()
    df_enc['Risk_Category_enc'] = rc_le.fit_transform(df['Risk_Category'].astype(str))
    joblib.dump(rc_le,      f"{MODEL_DIR}/risk_category_le.pkl")
    joblib.dump(encoders,   f"{MODEL_DIR}/feature_encoders.pkl")
    df.to_csv(f"{MODEL_DIR}/dataset.csv", index=False)

    # ── Feature engineering ──
    df_feat = add_engineered_features(df_enc)
    FEAT_COLS = get_feature_cols(df_feat)

    X_raw = df_feat[FEAT_COLS].values

    # ── Scaler (fit on base features only — used by prediction UI) ──
    base_scaler = StandardScaler()
    X_base_scaled = base_scaler.fit_transform(df_enc[BASE_FEATURES].values)
    joblib.dump(base_scaler, f"{MODEL_DIR}/scaler.pkl")

    # ── Full scaler for engineered features ──
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X_raw)
    joblib.dump(full_scaler, f"{MODEL_DIR}/full_scaler.pkl")
    joblib.dump(FEAT_COLS,   f"{MODEL_DIR}/feat_cols.pkl")

    metrics = {}

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LINEAR REGRESSION — Medical Expenses
    #    Dataset noise = 3% → Linear model should achieve R² ≈ 0.97+
    # ══════════════════════════════════════════════════════════════════════════
    y_lr = df_enc['Medical_Expenses'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_lr, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    metrics['linear_regression'] = {
        'MAE':  round(mean_absolute_error(y_te, y_pred), 2),
        'MSE':  round(mean_squared_error(y_te, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_te, y_pred)), 2),
        'R2':   round(r2_score(y_te, y_pred), 4),
        'y_test': y_te.tolist(),
        'y_pred': y_pred.tolist()
    }
    joblib.dump(lr, f"{MODEL_DIR}/linear_regression.pkl")
    print(f"✅ Linear Regression  R²    = {metrics['linear_regression']['R2']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. DECISION TREE — Disease Presence
    #    Clear threshold rules → should hit ~90%+
    # ══════════════════════════════════════════════════════════════════════════
    y_dt = df_enc['Disease_Presence'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_dt, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        criterion='gini',
        random_state=42
    )
    dt.fit(X_tr, y_tr)
    y_pred = dt.predict(X_te)
    metrics['decision_tree'] = {
        'Accuracy':  round(accuracy_score(y_te, y_pred), 4),
        'Report':    classification_report(y_te, y_pred, output_dict=True, zero_division=0),
        'Confusion': confusion_matrix(y_te, y_pred).tolist(),
        'y_test':    y_te.tolist(),
        'y_pred':    y_pred.tolist()
    }
    joblib.dump(dt, f"{MODEL_DIR}/decision_tree.pkl")
    print(f"✅ Decision Tree      Acc   = {metrics['decision_tree']['Accuracy']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. KNN — Risk Category
    #    Non-overlapping bands + scaled features → should hit ~90%+
    # ══════════════════════════════════════════════════════════════════════════
    y_knn = df_enc['Risk_Category_enc'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_knn, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    metrics['knn'] = {
        'Accuracy':  round(accuracy_score(y_te, y_pred), 4),
        'Report':    classification_report(y_te, y_pred, output_dict=True, zero_division=0),
        'Confusion': confusion_matrix(y_te, y_pred).tolist(),
        'Classes':   rc_le.classes_.tolist()
    }
    joblib.dump(knn, f"{MODEL_DIR}/knn.pkl")
    print(f"✅ KNN                Acc   = {metrics['knn']['Accuracy']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. K-MEANS — Patient Segments
    #    Use clinical-only features (normalized) for tight clusters
    # ══════════════════════════════════════════════════════════════════════════
    # Clinical subset for clustering: numerics most clinically meaningful
    cluster_cols = ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose',
                    'Previous_Visits', 'Smoking', 'Physical_Activity']
    X_clust_raw = df_enc[cluster_cols].values
    clust_scaler = StandardScaler()
    X_clust = clust_scaler.fit_transform(X_clust_raw)
    joblib.dump(clust_scaler, f"{MODEL_DIR}/cluster_scaler.pkl")

    # Find best k via silhouette (we fix k=4 per requirement but optimise init)
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=30,
                    max_iter=500, random_state=42)
    cluster_labels = kmeans.fit_predict(X_clust)
    sil = round(silhouette_score(X_clust, cluster_labels), 4)
    df['Cluster'] = cluster_labels
    metrics['kmeans'] = {
        'Silhouette': sil,
        'Inertia':    round(kmeans.inertia_, 2),
        'n_clusters': 4
    }
    joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.pkl")
    df.to_csv(f"{MODEL_DIR}/dataset_with_clusters.csv", index=False)
    print(f"✅ K-Means            Sil   = {sil:.4f}")

    joblib.dump(metrics, f"{MODEL_DIR}/metrics.pkl")
    print("\n📦 All models saved to", MODEL_DIR)
    return metrics


if __name__ == "__main__":
    train_all()