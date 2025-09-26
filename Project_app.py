import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("Employee Churn Risk Prediction App")
st.write("This application predicts the risk of employee churn (turnover). Please enter the information below.")


# Load model, threshold, feature names, and risk table
try:
    with open("smoteenn_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("best_threshold.pkl", "rb") as f:
        best_threshold = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    employee_risk = pd.read_csv("employee_risk_smoteenn.csv")

    st.success(f"✅ Model, threshold, and risk table loaded successfully. Used Threshold: {best_threshold:.2f}")

except FileNotFoundError as e:
    st.error(f"File loading error: {str(e)}")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"Dependency error: {str(e)}. Please install required libraries (e.g., 'pip install imbalanced-learn').")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    st.stop()

# Display risk table
st.subheader("Churn Risk Table")
st.write("This table shows the churn risk for employees in the validation set, including key features, true attrition status, risk score, and prediction.")
st.dataframe(employee_risk)

# Default values (focused on top effective features + Age, MonthlyIncome, TotalWorkingYears, NumCompaniesWorked)
default_values = {
    "Age": 35,
    "MonthlyIncome": 5000,
    "TotalWorkingYears": 10,
    "NumCompaniesWorked": 2,
    "YearsAtCompany": 5,
    "JobSatisfaction": 3,
    "WorkLifeBalance": 3,
    "YearsSinceLastPromotion": 2,
    "DistanceFromHome": 5,
    "PercentSalaryHike": 15,
    "TrainingTimesLastYear": 3,
    "YearsInCurrentRole": 3,
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "JobInvolvement": 3,
    "RelationshipSatisfaction": 3,
    "YearsWithCurrManager": 3,
    "StockOptionLevel": 1,
    "JobLevel": 2,
    "avg_weekly_hours": 40,
    "annual_leave_days": 20,
    "performance_score": 3,
    "burnout_risk": 0.5,
    "OverTime": "No",
    "BusinessTravel": "Travel_Rarely",
    "MaritalStatus": "Married",
    "JobRole": "Research Scientist",
    "Department": "Research & Development",
    "EducationField": "Life Sciences",
    "Gender": "Male"
}

# Expected columns (model requires these)
expected_columns = [
    "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked",
    "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
    "YearsAtCompany", "YearsInCurrentRole", "JobSatisfaction",
    "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "RelationshipSatisfaction", "WorkLifeBalance", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "StockOptionLevel", "JobLevel",
    "avg_weekly_hours", "annual_leave_days", "performance_score",
    "burnout_risk", "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

# User input form
with st.form("churn_form"):
    st.subheader("Enter Employee Information")
    age = st.number_input("Age (18-65)", min_value=18, max_value=65, value=35, step=1)
    salary = st.number_input("Monthly Salary (1000-20000)", min_value=1000, max_value=20000, value=5000, step=100)
    total_working_years = st.number_input("Total Working Years (0-40)", min_value=0, max_value=40, value=10, step=1)
    num_companies_worked = st.number_input("Number of Companies Worked (0-10)", min_value=0, max_value=10, value=2, step=1)
    years_at_company = st.number_input("Years at Company (0-40)", min_value=0, max_value=40, value=5, step=1)
    overtime = st.selectbox("Overtime (Yes/No)", ["No", "Yes"])
    burnout_risk = st.slider("Burnout Risk (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    years_since_last_promotion = st.number_input("Years Since Last Promotion (0-20)", min_value=0, max_value=20, value=2, step=1)
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    job_role_options = [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative"
    ]
    job_role = st.selectbox("Job Role", job_role_options)
    job_satisfaction = st.number_input("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3, step=1)
    work_life_balance = st.number_input("Work-Life Balance (1-4)", min_value=1, max_value=4, value=3, step=1)

    # Submit form
    submitted = st.form_submit_button("Predict")

# Prediction process
if submitted:
    try:
        # Update user inputs
        input_data = default_values.copy()
        input_data.update({
            "Age": age,
            "MonthlyIncome": salary,
            "TotalWorkingYears": total_working_years,
            "NumCompaniesWorked": num_companies_worked,
            "YearsAtCompany": years_at_company,
            "OverTime": overtime,
            "burnout_risk": burnout_risk,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "BusinessTravel": business_travel,
            "MaritalStatus": marital_status,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance
        })

        # Debug: Check input_data keys
        if set(input_data.keys()) != set(expected_columns):
            st.error("❌ Error: input_data keys do not match expected columns!")
            st.write("Missing keys:", set(expected_columns) - set(input_data.keys()))
            st.write("Extra keys:", set(input_data.keys()) - set(expected_columns))
            st.stop()

        # Create DataFrame (with original columns)
        new_data = pd.DataFrame([input_data])[expected_columns]
        st.success(f"✅ DataFrame created successfully. Shape: {new_data.shape}")

        # Make prediction
        prob_yes = pipeline.predict_proba(new_data)[0][1]
        prediction = 1 if prob_yes >= best_threshold else 0

        # Display results
        st.subheader("Prediction Results")
        st.write(f"- **Staying (No Churn)**: {1 - prob_yes:.2f}")
        st.write(f"- **Leaving (Churn)**: {prob_yes:.2f}")
        st.write("---")
        st.write(f"**Final Risk Score (Churn)**: {prob_yes:.2f}")
        st.write(f"**Used Threshold**: {best_threshold:.2f}")
        if prediction == 1:
            st.error("⚠️ High churn risk")
            # Explain high churn risk
            st.subheader("Why High Churn Risk?")
            if overtime == "Yes":
                st.write("- **Overtime (Yes)**: Strongly increases churn risk (importance: ~0.79)")
            if years_since_last_promotion > 2:
                st.write("- **Long time since last promotion**: Increases churn risk (importance: ~0.59)")
            if business_travel == "Travel_Frequently":
                st.write("- **Frequent business travel**: Increases churn risk (importance: ~0.63)")
            if marital_status == "Single":
                st.write("- **Single marital status**: Increases churn risk (importance: ~0.49)")
            if job_role in ["Sales Representative", "Laboratory Technician"]:
                st.write(f"- **Job Role ({job_role})**: Increases churn risk (importance: ~0.51-0.68)")
            if job_satisfaction <= 2:
                st.write("- **Low job satisfaction**: Increases churn risk (importance: ~0.62)")
            if work_life_balance <= 2:
                st.write("- **Poor work-life balance**: Increases churn risk (importance: ~0.47)")
        else:
            st.success("✅ Low churn risk")

        # Feature Importance
        classifier = pipeline.named_steps['classifier']
        if hasattr(classifier, 'coef_'):
            coefs = classifier.coef_[0]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs,
                'Importance': np.abs(coefs)
            }).sort_values('Importance', ascending=False)
            st.subheader("Feature Importance")
            st.dataframe(importance_df.head(10))
            

            # Plot Feature Importance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'].head(10), importance_df['Importance'].head(10))
            ax.set_xlabel("Importance (Absolute Coefficient)")
            ax.set_title("Top 10 Feature Importance (Logistic Regression + SMOTEENN)")
            ax.invert_yaxis()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")