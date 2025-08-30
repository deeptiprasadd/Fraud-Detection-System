# app.py
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("Fraud Detection System")

# ---------------- Load Pre-Trained Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/fraud_scaler.pkl")
    features = joblib.load("models/fraud_features.pkl")
    try:
        acc = joblib.load("models/fraud_accuracy.pkl")
        cm = joblib.load("models/fraud_confusion.pkl")
    except:
        acc, cm = None, None
    return model, scaler, features, acc, cm

model, scaler, features, acc, cm = load_model()

# ---------------- Feature Info ----------------
feature_info = {
    "step": ("Transaction Step", "Time since dataset start (in hours)", "System logs"),
    "type": ("Transaction Type", "PAYMENT, TRANSFER, CASH_OUT, etc.", "Transaction record"),
    "amount": ("Transaction Amount", "Money transferred in the transaction", "Transaction record"),
    "oldbalanceOrg": ("Sender's Old Balance", "Sender’s balance before transaction", "Bank account"),
    "newbalanceOrig": ("Sender's New Balance", "Sender’s balance after transaction", "Bank account"),
    "oldbalanceDest": ("Receiver's Old Balance", "Receiver’s balance before transaction", "Bank account"),
    "newbalanceDest": ("Receiver's New Balance", "Receiver’s balance after transaction", "Bank account"),
    "isFraud": ("Fraud Indicator", "1 = Fraud, 0 = Legitimate", "Label by investigators")
}

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", 
                        ["Prediction & Fraud Table", "Visual Insights", "Feature Explanations"])

# ---------------- Page 1: Prediction & Fraud Table ----------------
if page == "Prediction & Fraud Table":
    st.header("Fraud vs Non-Fraud Transactions")

    try:
        data = pd.read_csv("data/fraud_dataset.csv").sample(20000, random_state=42)

        numeric_features = [c for c in feature_info.keys() if c in data.columns and data[c].dtype != "object"]
        categorical_features = [c for c in feature_info.keys() if c in data.columns and data[c].dtype == "object"]

        fraud_avg = data[data["isFraud"] == 1][numeric_features].mean()
        nonfraud_avg = data[data["isFraud"] == 0][numeric_features].mean()

        fraud_table = pd.DataFrame({
            "Feature": [feature_info[f][0] for f in numeric_features],
            "Meaning": [feature_info[f][1] for f in numeric_features],
            "Avg (Legit)": nonfraud_avg.values,
            "Avg (Fraud)": fraud_avg.values,
        })
        st.table(fraud_table)

        # Show categorical features comparison
        for col in categorical_features:
            legit_mode = data[data["isFraud"] == 0][col].mode()[0]
            fraud_mode = data[data["isFraud"] == 1][col].mode()[0]
            st.markdown(f"**{feature_info[col][0]}** – {feature_info[col][1]}")
            st.write(f"- Most common Legit: {legit_mode}")
            st.write(f"- Most common Fraud: {fraud_mode}")
    except Exception as e:
        st.warning("⚠️ Dataset not found for averages table. Only prediction available.")

    st.divider()
    st.header("Make a Prediction")

    input_data = {}
    for col in features:
        if col == "isFraud":
            continue
        friendly_name, meaning, _ = feature_info.get(col, (col, "", ""))
        input_data[col] = st.number_input(f"{friendly_name}", 0.0, 1e7, 1000.0, help=meaning)

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100
        if prediction == 1:
            st.error(f"⚠️ Fraud Detected (Risk: {prob:.2f}%)")
        else:
            st.success(f"Legit Transaction (Risk: {prob:.2f}%)")

# ---------------- Page 2: Visual Insights ----------------
elif page == "Visual Insights":
    st.header("Model Performance & Insights")
    if acc:
        st.write(f"**Model Accuracy:** {acc*100:.2f}%")

    if cm is not None:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax)
        st.pyplot(fig)

    try:
        st.subheader("Fraud Distribution")
        dist = pd.read_csv("data/fraud_dataset.csv")["isFraud"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=dist.index, y=dist.values, palette="Set2", ax=ax)
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_ylabel("Count")
        st.pyplot(fig)
    except:
        st.warning("⚠️ Could not load dataset for fraud distribution.")

    st.subheader("Global Fraud Losses (Simulated)")
    fraud_loss = pd.DataFrame({
        "Year": [2015, 2017, 2019, 2021, 2023],
        "Losses (Trillions $)": [2.5, 3.1, 3.8, 4.6, 5.2]
    })
    st.line_chart(fraud_loss.set_index("Year"))

# ---------------- Page 3: Feature Explanations ----------------
elif page == "Feature Explanations":
    st.header("Feature Information")
    feature_table = pd.DataFrame(
        [(k, v[0], v[1], v[2]) for k, v in feature_info.items()],
        columns=["Dataset Feature", "Friendly Name", "Explanation", "Source (Where Found)"]
    )
    st.table(feature_table)
