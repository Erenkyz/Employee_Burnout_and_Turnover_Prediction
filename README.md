# 🧑‍💼 Employee Churn Prediction

## 📌 Project Overview
Employee attrition (churn) is a major challenge for companies, as losing skilled employees increases recruitment costs and affects productivity.  
The goal of this project is to **predict whether an employee is likely to leave the company** based on HR data.  
We use **machine learning models** to identify key risk factors and provide insights that HR teams can use for proactive retention strategies.

---

## 🧠 Model and Approach
We experimented with several models including **Random Forest** and **XGBoost**.  
Although these models provided solid performance, they struggled to balance **recall (catching churn cases)** with **overall accuracy**.

The final chosen model is:
- **Logistic Regression** combined with **SMOTEENN (Synthetic Minority Oversampling + Edited Nearest Neighbors)**  
  - SMOTEENN was applied to handle the strong **class imbalance** (only ~16% of employees in the dataset left the company).
  - Logistic Regression was chosen because it gave the **best trade-off between precision and recall**, ensuring churn cases are not overlooked.

---

## 📊 Dataset
- Dataset: **HR-Employee-Attrition.csv**  
- Contains demographic, job satisfaction, workload, and career-related features.  
- Target variable: **Attrition (Yes/No)**

---

## 📈 Results
Final Model: **Logistic Regression + SMOTEENN** with custom decision threshold (0.66)  

**Classification Report:**
precision    recall  f1-score   support

           0       0.93      0.85      0.89       247
           1       0.45      0.64      0.53        47

    accuracy                           0.82       294

  
- **Recall for churned employees (class 1): 0.64** → The model correctly identifies **64% of employees who are likely to leave**.  
- **Accuracy: 82%** → Balanced performance across both classes.  
- Compared to Random Forest and XGBoost, this model provided **higher recall** without sacrificing much accuracy, making it more reliable for HR use cases.

---

## 🛠️ Tech Stack
- **Python Libraries**  
  - `streamlit` – Web application for interactive predictions  
  - `pandas`, `numpy` – Data processing  
  - `scikit-learn` – Machine learning models and evaluation  
  - `imbalanced-learn` – SMOTEENN for class balancing  
  - `matplotlib`, `seaborn` – Visualizations  

---

## 🎯 Project Purpose
This project aims to:
- Help companies identify employees at risk of leaving.  
- Provide HR managers with actionable insights for retention strategies.  
- Demonstrate how imbalanced classification problems can be solved using resampling techniques and proper threshold tuning.  

The final outcome is a **Streamlit application** where users can input employee details and instantly receive a prediction about churn risk.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/ErenKyz/Employee_Burnout_and_Turnover_Prediction.git
cd Employee_Burnout_and_Turnover_Prediction

pip install -r requirements.txt

streamlit run app.py
