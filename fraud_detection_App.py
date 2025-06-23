import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE

# --- Page Setup ---
st.set_page_config(page_title='Fraud Detection Pro', layout='wide')
st.markdown("<h1 style='text-align: center;'>🚨 Fraud Detection System</h1>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/595/595067.png", width=100)
st.sidebar.markdown("## 🔎 About This App")
st.sidebar.markdown("Predict whether a transaction is **fraudulent or legitimate** using machine learning.")
st.sidebar.markdown("Built by **Deepti Prasad** 💻")

# --- Load and Train ---
@st.cache_data
def load_data_and_amount_scaler():
    df = pd.read_csv('Fraud.csv')
    df['nameDest'] = df['nameDest'].fillna('Unknown')
    df['oldbalanceDest'] = df['oldbalanceDest'].fillna(0)
    df['newbalanceDest'] = df['newbalanceDest'].fillna(0)
    
    amount_scaler = RobustScaler()
    df['amount_scaled'] = amount_scaler.fit_transform(df['amount'].values.reshape(-1, 1))
    
    df['balanceChangeOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceChangeDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['type'] = df['type'].astype('category').cat.codes
    df = df.drop(['nameOrig', 'nameDest', 'amount'], axis=1)
    return df, amount_scaler

@st.cache_resource
def train_model():
    df, amount_scaler = load_data_and_amount_scaler()
    df_sampled = df.sample(frac=0.05, random_state=42)  # Reduced to 5% for speed
    X = df_sampled.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = df_sampled['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
    
    with st.spinner('Training Model...'):
        time.sleep(1)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)

    return model, scaler, amount_scaler, X_test_scaled, y_test, X.columns, df_sampled

model, scaler, amount_scaler, X_test_scaled, y_test, feature_names, full_df = train_model()

# --- Dataset Preview ---
st.markdown("## 📁 Dataset Preview")
st.dataframe(full_df.head(10))

# --- Fraud Stats ---
st.markdown("## 📊 Dataset Summary")
col1, col2 = st.columns(2)
with col1:
    fraud_count = full_df['isFraud'].sum()
    total_count = full_df.shape[0]
    st.metric("🚩 Fraudulent Transactions", f"{fraud_count:,}")
with col2:
    st.metric("📉 Fraud Rate", f"{(fraud_count / total_count):.2%}")

# --- Sample Transactions ---
st.markdown("## 🧾 Sample Predictions")
sample_data = pd.DataFrame({
    'step': [1, 100],
    'type': ['TRANSFER', 'CASH_OUT'],
    'oldbalanceOrg': [10000, 0],
    'newbalanceOrig': [0, 0],
    'oldbalanceDest': [0, 5000],
    'newbalanceDest': [10000, 10000]
})
sample_data['balanceChangeOrig'] = sample_data['oldbalanceOrg'] - sample_data['newbalanceOrig']
sample_data['balanceChangeDest'] = sample_data['newbalanceDest'] - sample_data['oldbalanceDest']
sample_data['type'] = sample_data['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})
sample_data['amount_scaled'] = amount_scaler.transform(sample_data[['balanceChangeOrig']])
sample_data = sample_data[feature_names]
sample_scaled = scaler.transform(sample_data)
sample_probs = model.predict_proba(sample_scaled)[:, 1]
sample_table = pd.DataFrame({
    **sample_data,
    "Fraud Probability": [f"{p:.2%}" for p in sample_probs]
})
st.dataframe(sample_table)

# --- Custom Prediction ---
st.markdown("## ✍️ Enter Custom Transaction")
type_map = {'TRANSFER': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'CASH_IN': 4}
user_input = {}
with st.form("form_custom"):
    col1, col2 = st.columns(2)
    with col1:
        user_input['step'] = st.number_input("Step", value=1.0)
        tx_type = st.selectbox("Type", list(type_map.keys()))
        user_input['type'] = type_map[tx_type]
        user_input['oldbalanceOrg'] = st.number_input("Old Balance (Sender)", value=0.0)
        user_input['newbalanceOrig'] = st.number_input("New Balance (Sender)", value=0.0)
    with col2:
        user_input['oldbalanceDest'] = st.number_input("Old Balance (Receiver)", value=0.0)
        user_input['newbalanceDest'] = st.number_input("New Balance (Receiver)", value=0.0)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    user_input['balanceChangeOrig'] = user_input['oldbalanceOrg'] - user_input['newbalanceOrig']
    user_input['balanceChangeDest'] = user_input['newbalanceDest'] - user_input['oldbalanceDest']
    user_input['amount_scaled'] = amount_scaler.transform([[abs(user_input['balanceChangeOrig'])]])[0][0]
    input_df = pd.DataFrame([user_input])
    for feat in feature_names:
        if feat not in input_df.columns:
            input_df[feat] = 0.0
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

    st.success("✅ Prediction Completed!")
    st.metric("📊 Fraud Probability", f"{prob:.2%}")
    st.dataframe(input_df.assign(**{"Predicted Fraud Probability": [f"{prob:.2%}"]}))

# --- Evaluation Graphs ---
with st.expander("📈 Show Model Evaluation"):
    st.markdown("### Confusion Matrix")
    y_pred = model.predict(X_test_scaled)
    fig1, ax1 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="YlGnBu",
                xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"], ax=ax1)
    st.pyplot(fig1)

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision, color="green")
    ax3.set_title("Precision-Recall Curve")
    st.pyplot(fig3)

    st.markdown("### 🔍 Feature Importance")
    fig4, ax4 = plt.subplots()
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances.nlargest(10).plot(kind='barh', ax=ax4)
    st.pyplot(fig4)

    st.markdown("### 🧮 Class Distribution")
    fig5, ax5 = plt.subplots()
    sns.countplot(data=full_df, x='isFraud', palette='Set2', ax=ax5)
    ax5.set_xticklabels(['Legit', 'Fraud'])
    ax5.set_title("Class Distribution in Sampled Data")
    st.pyplot(fig5)

st.caption("✨ Made by Deepti Prasad")
