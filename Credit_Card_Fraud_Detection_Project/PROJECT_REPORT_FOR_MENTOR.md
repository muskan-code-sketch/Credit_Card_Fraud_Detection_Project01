# 📘 Academic Project Report: ML-Based Credit Card Fraud Detection System

**Author / Candidate:** Muskan  
**Domain:** Financial Machine Learning & Fraud Risk Analytics  
**Architecture:** Multi-Model Benchmark (Logistic Regression, Random Forest, XGBoost) with SMOTE & Cost-Sensitive Loss Weighting  

---

## 1. Executive Summary & Problem Formulation
Online digital transactions process hundreds of millions of card payments per day. However, fraudulent transactions account for an extreme minority—typically fewer than **0.2%** (~1 fraud per 578 legitimate transactions). 

A standard machine learning classifier trained on this data faces three severe challenges:
1. **The Accuracy Paradox:** Predicting 100% genuine transactions yields **99.828% accuracy**, but intercepts zero fraud cases, resulting in total operational failure.
2. **Asymmetric Misclassification Costs:** The cost of a **False Negative (missed fraud)** is significantly higher than a **False Positive (temporary 2FA friction / analyst review)**.
3. **Data Leakage Vulnerability:** Synthesizing data via SMOTE or fitting scalers before train-test splitting leaks test distribution information, inflating benchmark metrics that fail in production.

This project implements a rigorous, leakage-free pipeline that compares multiple machine learning algorithms under both **Cost-Sensitive Class Weighting** and **SMOTE Resampling**, determines cost-calibrated decision thresholds, and deploys an interactive **Streamlit** dashboard for real-time and batch screening.

---

## 2. Mathematical Formulations & Techniques

### A. Preprocessing: Robust Scaling
Financial transactions contain extreme monetary outliers (e.g., normal shopping €10–€50 vs high-dollar drains up to €25,000). Standard Z-score scaling ($Z = \frac{x - \mu}{\sigma}$) gets distorted by these outliers.
We use **RobustScaler**:
$$X_{\text{scaled}} = \frac{X - Q_2(X)}{Q_3(X) - Q_1(X)} = \frac{X - \text{Median}(X)}{\text{IQR}(X)}$$
* Fitted **strictly on $X_{\text{train}}$** to prevent data leakage.
* PCA features ($V_1$ to $V_{28}$) are already centered at 0 and preserved without double-scaling.

### B. Imbalance Handling: Cost-Sensitive Loss Weighting
Instead of modifying data distributions, cost-sensitive learning penalizes false negatives directly in the objective loss function:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{79,862}{138} \approx 578.71$$
In XGBoost, this ensures the gradient update places $\approx 578\times$ more importance on correctly classifying minority fraud instances.

### C. Imbalance Handling: SMOTE (Synthetic Minority Over-sampling Technique)
For a minority sample $\mathbf{x}_i \in \text{Class } 1$ and its $k$-nearest minority neighbor $\mathbf{x}_{zi}$:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim \text{Uniform}(0, 1)$$
* **Strict Leakage Prevention:** SMOTE is applied **only to the training fold** ($X_{\text{train}}, y_{\text{train}}$). The test set ($X_{\text{test}}, y_{\text{test}}$) remains untouched to measure true real-world generalization.

### D. Primary Evaluation Metric: PR-AUC (Precision-Recall Area Under Curve)
Because true negatives ($TN \approx 19,966$) dwarf false positives ($FP$), ROC-AUC can remain artificially high ($>0.98$) even when precision is poor.  
**PR-AUC (Average Precision)** evaluates the trade-off directly between Precision ($TP / (TP + FP)$) and Recall ($TP / (TP + FN)$):
$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

### E. Financial Cost Optimization Formula
Rather than an arbitrary $\theta = 0.50$ threshold, the system evaluates $\theta \in [0.01, 0.99]$ to minimize total operational cost:
$$\text{Total Cost}(\theta) = \sum \Big(\text{False Negatives}(\theta) \times C_{\text{fraud}}\Big) + \sum \Big(\text{False Positives}(\theta) \times C_{\text{review}}\Big)$$
* $C_{\text{fraud}}$: Average direct chargeback loss (default: €250).
* $C_{\text{review}}$: Operational investigator fee (default: €20).

---

## 3. Benchmark Experimental Results

Evaluated on an untouched test partition of 20,000 transactions (with 34 ground-truth fraud cases):

| Algorithm | Imbalance Strategy | PR-AUC (Primary) | ROC-AUC | Recall | Precision | F1-Score | F2-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** 🏆 | **Cost-Sensitive Weighting** | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Random Forest** | **Balanced Subsample** | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Logistic Regression** | **Balanced Class Weight** | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **XGBoost** | **SMOTE Resampling** | **1.0000** | **1.0000** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| **Logistic Regression** | **SMOTE Resampling** | **1.0000** | **1.0000** | **100.0%** | **97.14%** | **0.9855** | **0.9942** |

> **Key Research Finding:** Algorithmic class weighting (`scale_pos_weight`) performs equal to or better than SMOTE while requiring substantially less computation time and avoiding synthetic noise in high-dimensional PCA spaces.

---

## 4. Codebase Module Map for Code Review

| Module | File | Key Functions / Classes | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | `src/data_loader.py` | `generate_benchmark_credit_card_data()`, `load_credit_card_data()` | Loads Kaggle/OpenML data or generates high-fidelity synthetic benchmark with ~0.17% fraud rate. |
| **Preprocessing** | `src/preprocessor.py` | `FraudDataPreprocessor`, `prepare_train_test_data()` | Stratified 80/20 train-test split; fits `RobustScaler` strictly on training data. |
| **Model Training** | `src/train.py` | `train_and_compare_models()` | Trains 5 distinct model configurations, performs SMOTE resampling, and serializes models to `.joblib`. |
| **Evaluation** | `src/evaluate.py` | `evaluate_model_performance()`, `find_optimal_threshold()` | Computes PR-AUC, confusion matrices, and calculates cost-optimal threshold $\theta$. |
| **Web Dashboard** | `app.py` | Multi-tab Streamlit Application | 5 interactive tabs for EDA, Model Benchmarks, Single Inference, Batch CSV triage, and Cost Simulation. |
| **Research Notebook** | `notebooks/fraud_detection_analysis.ipynb` | Step-by-step cells | Comprehensive Jupyter notebook following story-driven data science best practices. |

---

## 5. Mentor Viva / Q&A Defense Cheatsheet

**Q1: Why did you not use Accuracy as the primary evaluation metric?**  
*Answer:* In a dataset with 99.83% genuine transactions, a dummy model predicting all transactions as genuine achieves 99.83% accuracy while failing to detect any fraud. PR-AUC and Recall evaluate how well minority positive cases are identified without being masked by the large majority class.

**Q2: What is the risk of applying SMOTE before train-test splitting?**  
*Answer:* Applying SMOTE across the whole dataset creates synthetic points that interpolate between training and test samples. This causes severe data leakage, giving unrealistically high scores during testing that fail completely in real production data.

**Q3: Why use RobustScaler instead of StandardScaler for financial amounts?**  
*Answer:* Transaction amounts contain severe right-skewed outliers (e.g. median €22 vs maximum €25,000). StandardScaler relies on mean and standard deviation, which are skewed by outliers. RobustScaler uses median and IQR ($Q_3 - Q_1$), preventing outlier distortion.

**Q4: How does Cost-Sensitive Learning compare with SMOTE?**  
*Answer:* SMOTE synthesizes artificial data points in 28-dimensional PCA space, which can introduce boundary noise. Cost-Sensitive weighting adjusts the algorithm's loss penalty ($scale\_pos\_weight \approx 578$) without modifying the original data distribution, training faster and achieving superior precision.
