import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/churn.csv")

data = load_data()

# -------------------------------
# Fix Churn column
# -------------------------------
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("models/churn_model.pkl")

# -------------------------------
# Prepare data for prediction
# -------------------------------
X = data.drop(columns=["Churn"])

# Predict churn probability
data["Churn_Probability"] = model.predict_proba(X)[:, 1]

# -------------------------------
# TOP METRICS
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Customers", len(data))
col2.metric("❌ Churned Customers", int(data["Churn"].sum()))
col3.metric("📉 Churn Rate", f"{data['Churn'].mean() * 100:.2f}%")
col4.metric("⚠️ Avg Churn Risk", f"{data['Churn_Probability'].mean() * 100:.2f}%")

st.divider()

# -------------------------------
# CHARTS
# -------------------------------
st.subheader("📈 Churn Distribution")

fig, ax = plt.subplots()
data["Churn"].value_counts().plot(
    kind="bar",
    ax=ax,
    title="Churn vs Non-Churn",
)
ax.set_xticklabels(["No Churn", "Churn"], rotation=0)
st.pyplot(fig)

# -------------------------------
# HIGH RISK CUSTOMERS
# -------------------------------
st.subheader("🚨 High Risk Customers")

high_risk = data[data["Churn_Probability"] > 0.7][
    ["customerID", "Churn_Probability"]
].sort_values(by="Churn_Probability", ascending=False)

st.dataframe(high_risk)

# -------------------------------
# RAW DATA (OPTIONAL)
# -------------------------------
with st.expander("📄 View Full Dataset"):
    st.dataframe(data)
