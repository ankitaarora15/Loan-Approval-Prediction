# 🏦 Loan Approval Prediction using Supervised ML

## 📌 Project Overview
An end-to-end **Supervised Learning (Classification)** pipeline designed to automate credit-risk assessment and predict loan applicant approvals.

---

## 🛠️ Data Pipeline & Workflow Sequence
The project follows a standard 10-step data science lifecycle:
1. Data Inspection: Evaluated structure, null counts, and summary statistics.
2. Data Cleaning: Imputed missing numerical and categorical variables.
3. Exploratory Data Analysis (EDA): Visualized asset variations and anomalies.
4. Feature Engineering: Removed non-predictive reference identifiers.
5. Data Preprocessing: Used `LabelEncoder` and `OneHotEncoder` for text and categorical arrays.
6. Correlation Analysis: Evaluated linear feature links using a Seaborn heatmap.
7. Dataset Splitting: Separated inputs (`X`) and targets (`y`) with a 20% validation pool.
8. Feature Scaling: Normalized parameters via `StandardScaler`.
9. Model Execution & Evaluation: Trained classifiers (Logistic Regression, KNN, Gaussian Naive Bayes) and evaluated via Precision, Recall, F1-Score, and Accuracy.
10. Feature Optimization: Applied non-linear combinations to optimize accuracy.

---

## 💻 Tech Stack & Libraries Used
* Language: Python
* Data Manipulation: Pandas, NumPy
* Data Visualization: Seaborn, Matplotlib
* Machine Learning:  Scikit-Learn

