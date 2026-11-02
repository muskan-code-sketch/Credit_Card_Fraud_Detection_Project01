# 💳 ML-Based Credit Card Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalance-SMOTE%20Resampling-green.svg)](https://imbalanced-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, machine-learning-driven financial transaction screening platform designed to detect fraudulent credit card activity in highly imbalanced transaction streams (~0.17% fraud rate) using advanced classification algorithms, SMOTE resampling, and financial risk threshold calibration.

---

## 📌 Problem Statement & Synopsis Alignment

Digital payment systems process hundreds of millions of transactions daily. However, fraudulent transactions represent a minute fraction of the total volume (typically **~0.17%** or 1 in every 578 transactions). Detecting fraud effectively presents three critical technical and business challenges:

1. **Extreme Class Imbalance:** A naive classifier predicting 100% legitimate transactions achieves **99.828% accuracy**, yet fails completely to catch fraudulent attacks.
2. **Asymmetric Business Costs:** A **False Negative (missed fraud)** causes immediate chargeback loss and customer churn, while a **False Positive (false alarm)** creates customer friction and investigator review costs.
3. **Data Leakage in Resampling:** Resampling techniques such as **SMOTE** must be strictly applied **only** to training partitions to prevent data leakage and artificially optimistic test results.

This project addresses these challenges by comparing multiple classification algorithms (**Logistic Regression**, **Random Forest**, and **XGBoost**) under both **Cost-Sensitive Class Weighting** and **SMOTE Resampling**, optimizing decision thresholds to minimize financial loss, and exposing the models via an intuitive **Streamlit** user interface.

---

## 🚀 Key Features

* **Real-Time Single Transaction Screener:** Instant risk scoring with pre-configured test presets (*Genuine Coffee & Grocery* vs *Suspicious Overnight Account Drain*) and dynamic risk gauge meters.
* **Batch CSV Transaction Screening:** Drag-and-drop CSV batch upload, instant high-risk triaging, volume-at-risk analysis, and downloadable triage reports.
* **Leakage-Free Preprocessing:** `RobustScaler` on `Time` and `Amount` (fitted exclusively on training folds) to handle heavy financial outliers without contaminating test evaluations.
* **Class Imbalance Mitigation:**
  * **Cost-Sensitive Weighting:** `scale_pos_weight` in XGBoost and `class_weight='balanced_subsample'` in Random Forest.
  * **SMOTE Resampling:** Synthesizes minority fraud cases strictly within training splits.
* **Metric Alignment (Beyond Accuracy):** Evaluates models across **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, **Recall**, **Precision**, **F1-Score**, and **F2-Score**.
* **Financial Cost-Calibrated Decision Threshold ($\theta$):** Dynamically optimizes threshold $\theta \in [0.01, 0.99]$ to minimize total operational banking cost:
  $$\text{Total Cost} = \sum (\text{FN} \times \text{Direct Fraud Loss}) + \sum (\text{FP} \times \text{Manual Review Fee})$$
* **Interactive Jupyter Notebook:** Story-driven research notebook following strict ML best practices with markdown analysis for every step.

---

## 🛠️ Technology Stack

| Category | Tools & Libraries | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core implementation |
| **Data Processing** | Pandas, NumPy | Vectorized data manipulation & transformations |
| **Machine Learning** | Scikit-learn, XGBoost | Preprocessing pipelines, classification & metrics |
| **Imbalance Handling** | Imbalanced-learn (`imblearn`) | Synthetic Minority Over-sampling Technique (SMOTE) |
| **Data Visualization** | Plotly, Matplotlib, Seaborn | Interactive web charts and statistical plots |
| **User Interface** | Streamlit | Real-time web dashboard & batch triage interface |
| **Serialization** | Joblib | Model & preprocessor pipeline persistence |

---

## 📂 Project Architecture

```
credit_card_fraud_detection/
├── .gitignore                         # Excludes virtual env & raw 60MB data
├── .streamlit/
│   └── config.toml                    # Theme & server configuration
├── data/
│   ├── creditcard.csv                 # Full benchmark dataset (100k+ rows)
│   ├── creditcard_sample.csv          # Sample dataset for fast app loading (10k rows)
│   ├── test_batch_sample.csv          # 15-transaction batch test file (unlabeled)
│   └── test_batch_reference_labeled.csv # 15-transaction reference with ground truth
├── models/
│   ├── best_model.joblib              # Primary champion classifier (XGBoost)
│   ├── preprocessor.joblib            # Fitted RobustScaler pipeline
│   ├── logistic_regression.joblib     # Balanced Logistic Regression
│   ├── logistic_regression_smote.joblib # LR trained with SMOTE
│   ├── random_forest.joblib           # Balanced Random Forest
│   ├── xgboost.joblib                 # Cost-Sensitive XGBoost
│   ├── xgboost_smote.joblib           # XGBoost trained with SMOTE
│   ├── model_metrics.json             # Benchmark metrics dictionary
│   └── test_data.joblib               # Test set partition for simulator
├── notebooks/
│   ├── build_notebook.py              # Reproducible notebook generator
│   └── fraud_detection_analysis.ipynb # End-to-end research notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loader & benchmark generator
│   ├── preprocessor.py                # Leakage-free split & scaler
│   ├── train.py                       # Multi-model training orchestrator
│   └── evaluate.py                    # PR-AUC, ROC-AUC, threshold optimizer
├── app.py                             # Main Streamlit web application
├── requirements.txt                   # Production dependencies
└── README.md                          # Full documentation & guide
```

---

## 📊 Benchmark Results

| Model | PR-AUC (Primary) | ROC-AUC | Recall | Precision | F1-Score | F2-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Cost-Weighted)** 🏆 | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Random Forest (Balanced)** | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Logistic Regression (Balanced)** | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **XGBoost + SMOTE** | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Logistic Regression + SMOTE** | **1.0000** | **1.0000** | **100.0%** | **97.14%** | **0.9855** | **0.9942** |

> **Key Finding:** In financial fraud detection, **Cost-Sensitive Loss Weighting** (`scale_pos_weight ≈ 578.71` in XGBoost and `class_weight='balanced_subsample'` in Random Forest) consistently matches or exceeds SMOTE performance while eliminating synthetic noise in high-dimensional PCA spaces.

---

## ⚙️ Local Installation & Execution

### 1. Clone Repository & Create Virtual Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Initialize virtual environment
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\activate

# Activate on Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-Train Models

All models are already pre-trained and serialized in `models/`. To retrain from scratch:

```bash
python -m src.train
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Fork or push this repository to your **GitHub** account.
2. Sign in to **[share.streamlit.io](https://share.streamlit.io)** using your GitHub account.
3. Click **"New app"** and select:
   * **Repository:** `YOUR_USERNAME/credit-card-fraud-detection`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **"Deploy!"**
5. Within 1–2 minutes, your application will be live at a public HTTPS URL (e.g., `https://your-app-name.streamlit.app`).

---

## 📋 Methodology Flowchart

```
Dataset Collection (ULB European Cardholders)
                 │
                 ▼
Data Preprocessing (RobustScaler on Amount & Time)
                 │
                 ▼
Stratified Train-Test Split (80% Train / 20% Test)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
Cost Weighting           SMOTE (Train Only)
      │                     │
      └──────────┬──────────┘
                 ▼
Model Training (Logistic Regression, Random Forest, XGBoost)
                 │
                 ▼
Model Evaluation (PR-AUC, Recall, ROC-AUC, Confusion Matrix)
                 │
                 ▼
Cost-Calibrated Threshold Optimization (Minimizing Loss)
                 │
                 ▼
Streamlit Interactive Deployment
```

---

## ⚖️ Applications & Future Scope

### Applications:
* **Banking Transaction Monitoring:** Real-time scoring of card transactions at authorization time.
* **E-Commerce Fraud Screening:** Automated risk flagging during checkout to prevent chargebacks.
* **Risk Desk Triaging:** Routing high-risk payments to fraud investigation queues for 2FA/SMS challenge.

### Future Scope:
* Integration with real-time streaming engines (Apache Kafka / Flink).
* Incorporating geolocation, IP velocity, and device fingerprinting features.
* Implementing Explainable AI (SHAP / LIME waterfall plots) for cardholder dispute auditing.

---

## 📄 License & Acknowledgments

* **Dataset:** Modeled after the European Cardholders Credit Card Fraud Detection benchmark by Université Libre de Bruxelles (ULB) Machine Learning Group and Worldline.
* **License:** Released under the [MIT License](LICENSE).
