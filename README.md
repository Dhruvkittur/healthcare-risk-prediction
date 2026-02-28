# 🏥 Smart Healthcare Risk Prediction & Patient Segmentation System

An AI-powered web application for healthcare risk prediction using multiple machine learning algorithms.

**Team:** Dhruv K · Pramodini G · Rahul S · Ranjita M

---

## 🚀 Features

| Algorithm | Task | Output |
|---|---|---|
| Linear Regression | Predict Medical Expenses | Annual cost estimate |
| Decision Tree | Classify Disease Presence | Binary (Yes/No) |
| KNN | Predict Risk Category | 4 risk levels |
| K-Means | Patient Segmentation | 4 clusters |

---

## 📁 Project Structure

```
healthcare_app/
├── app.py                  # Main Streamlit application
├── data_generator.py       # Synthetic dataset generator
├── train_models.py         # Model training script
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme config
└── models/                 # (auto-created after training)
    ├── linear_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── kmeans.pkl
    ├── scaler.pkl
    ├── feature_encoders.pkl
    ├── risk_category_le.pkl
    ├── metrics.pkl
    └── dataset_with_clusters.csv
```

---

## ⚙️ Local Setup

### Step 1 – Clone or download the project
```bash
git clone https://github.com/YOUR_USERNAME/healthcare-risk-prediction.git
cd healthcare-risk-prediction
```

### Step 2 – Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 – Train the models
```bash
python train_models.py
```
This generates the `models/` directory with all trained models and datasets.

### Step 5 – Run the app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## ☁️ Deployment on Streamlit Cloud (Free)

### Step 1 – Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Healthcare Risk Prediction App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 – Create a Streamlit Cloud account
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with your GitHub account

### Step 3 – Deploy
1. Click **"New app"**
2. Select your GitHub repository
3. Set **Main file path** to: `app.py`
4. Click **"Deploy!"**

> ⚠️ **Important:** The `models/` folder must be committed to GitHub, OR you need to add a startup script. See the note below.

### Step 3b – Handle model files for deployment

Since model `.pkl` files can be large, add a `packages.txt` file and ensure `train_models.py` is called on first run. The easiest way is to add this to the top of `app.py`:

```python
import os
if not os.path.exists("models/linear_regression.pkl"):
    import subprocess
    subprocess.run(["python", "train_models.py"])
```

This is already included in the app — models auto-train on first deployment.

---

## 🌐 Alternative Deployment: Hugging Face Spaces

### Step 1 – Create account at huggingface.co
### Step 2 – New Space → SDK: Streamlit
### Step 3 – Upload all files including `requirements.txt`
### Step 4 – Hugging Face auto-builds and deploys

---

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python train_models.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t healthcare-app .
docker run -p 8501:8501 healthcare-app
```

---

## 📊 Model Performance Summary

| Model | Metric | Value |
|---|---|---|
| Linear Regression | R² Score | ~0.86 |
| Decision Tree | Accuracy | ~72% |
| KNN | Accuracy | ~76% |
| K-Means | Silhouette Score | ~0.13 |

---

## 🧬 Dataset Features

| Feature | Type | Range/Values |
|---|---|---|
| Age | Numeric | 18–90 years |
| Gender | Categorical | Male / Female |
| BMI | Numeric | 15–50 kg/m² |
| Blood Pressure | Numeric | 80–200 mmHg |
| Cholesterol | Numeric | 100–320 mg/dL |
| Glucose Level | Numeric | 60–250 mg/dL |
| Smoking | Binary | Yes / No |
| Physical Activity | Categorical | Low / Moderate / High |
| Family History | Binary | Yes / No |
| Previous Visits | Numeric | 0–15 |

---

## 📝 GitHub Steps (Detailed)

```bash
# 1. Install Git if not installed: https://git-scm.com/downloads

# 2. Configure Git
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# 3. Initialize repo
cd healthcare_app
git init
git add .
git commit -m "feat: complete healthcare AI system"

# 4. Create repo on GitHub.com (click + → New repository)
#    Name: healthcare-risk-prediction
#    Visibility: Public
#    DON'T initialize with README

# 5. Connect and push
git remote add origin https://github.com/YOUR_USERNAME/healthcare-risk-prediction.git
git branch -M main
git push -u origin main
```

---

*Built with ❤️ using Python, Scikit-learn, and Streamlit*
