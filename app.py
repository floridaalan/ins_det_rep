import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Streamlit App Header
st.title("\U0001F6E1️ Government Health Insurance Fraud Detection System")
st.write("AI-powered system to detect fraudulent insurance claims while protecting frequent innocent patients.")

# Upload CSV File
uploaded_file = st.file_uploader("\U0001F4C1 Upload Health Insurance Claims CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### \U0001F50D Data Preview")
    st.write(df.head())

    # Predefined dictionaries for cost calculation
    gov_rates = {'P1695': 30000, 'P1237': 150000, 'P1881': 50000, 'P1290': 20000}
    hospital_types = {'HOS666': 'Government', 'HOS325': 'Private', 'HOS531': 'Specialized'}
    geo_variation = {'HOS666': 1.0, 'HOS325': 1.2, 'HOS531': 1.5}
    additional_expenses = {'TXN000001': 5000, 'TXN000002': 10000}
    chronic_conditions = {12345: 'Diabetes', 67890: 'Cancer', 11223: 'Dialysis'}  # Example Patient IDs


    # Calculate Actual Cost
    def determine_actual_cost(row):
        base_cost = gov_rates.get(row['Procedure Code'], 0)
        hospital_factor = 0.8 if hospital_types.get(row['Hospital ID'], 'Private') == 'Government' else 1.2
        complexity_factor = 1.5 if row['Diagnosis Code'] in ['D123', 'D456'] else 1.0
        geo_factor = geo_variation.get(row['Hospital ID'], 1.0)
        additional_cost = additional_expenses.get(row['Transaction ID'], 0)
        return base_cost * hospital_factor * complexity_factor * geo_factor + additional_cost


    df['Computed_Actual_Cost'] = df.apply(determine_actual_cost, axis=1)
    df['Claim_Treatment_Diff'] = df['Claim_Amount'] - df['Computed_Actual_Cost']
    df['Chronic_Condition_Flag'] = df['Patient ID'].apply(lambda x: 1 if x in chronic_conditions else 0)

    # Features for ML
    features = ['Claim_Amount', 'Claim_Treatment_Diff', 'Days_Between_Claims']
    X = df[features]

    has_label = 'Final Status' in df.columns
    y = (df['Final Status'] == 'Fraud').astype(int) if has_label else None

    # Autoencoder for Anomaly Detection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    autoencoder = Sequential([
        Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(X_scaled.shape[1], activation='linear')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=4, verbose=0)

    mse = np.mean(np.power(X_scaled - autoencoder.predict(X_scaled), 2), axis=1)
    threshold = np.percentile(mse, 95)
    df['Fraud_Anomaly_Score'] = mse
    df['Fraud_Flag_Autoencoder'] = (mse > threshold).astype(int)

    # XGBoost for Supervised Learning
    if has_label:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        df['Fraud_Flag_XGBoost'] = xgb_model.predict(X)
        accuracy = xgb_model.score(X_test, y_test)
        st.info(f"✅ XGBoost Model Accuracy: {accuracy:.2%}")
    else:
        df['Fraud_Flag_XGBoost'] = 0
        st.warning("⚠️ XGBoost model skipped due to missing labels.")

    # Final Fraud Decision (Reducing False Positives for Chronic Patients)
    df['Fraud_Flag_Final'] = ((df['Fraud_Flag_Autoencoder'] + df['Fraud_Flag_XGBoost']) > 0).astype(int)
    df.loc[df['Chronic_Condition_Flag'] == 1, 'Fraud_Flag_Final'] = 0


    # Fraud Reasoning
    def get_fraud_reason(row):
        reasons = []
        if row['Claim_Treatment_Diff'] > 50000:
            reasons.append("Excess Claim Amount")
        if row['Days_Between_Claims'] < 30:
            reasons.append("Frequent Claims")
        if row['Fraud_Flag_Autoencoder'] == 1:
            reasons.append("Anomaly Detected")
        if row['Fraud_Flag_XGBoost'] == 1:
            reasons.append("XGBoost Prediction")
        if row['Chronic_Condition_Flag'] == 1:
            reasons.append("Chronic Patient (Protected)")
        return " | ".join(reasons) if reasons else "Normal"


    df['Fraud_Reasons'] = df.apply(get_fraud_reason, axis=1)

    # Display Fraud Cases
    st.subheader("🚨 Fraud Detection Results")
    fraud_cases = df[df['Fraud_Flag_Final'] == 1]
    if not fraud_cases.empty:
        st.error(f"🚨 **{len(fraud_cases)} Fraudulent Claims Detected!** 🚨")
        st.write(fraud_cases[['Transaction ID', 'Claim_Amount', 'Hospital ID', 'Fraud_Reasons']])
    else:
        st.success("✅ No fraudulent claims detected. Government funds are safe.")

    # Visualization
    st.subheader("Fraud Anomaly Score vs Claim Amount")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Claim_Amount', y='Fraud_Anomaly_Score', hue='Fraud_Flag_Final', palette='coolwarm')
    plt.xlabel("Claim Amount")
    plt.ylabel("Fraud Anomaly Score")
    plt.title("Fraud Detection Visualization")
    st.pyplot(plt)

    st.success("✅ Fraud detection process completed successfully!")
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
