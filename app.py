import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Page Config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load Model 
@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Title 
st.title("📊 Customer Churn Prediction App")
st.markdown("*Predict whether a customer will leave and understand reasons*")
st.divider()

# Sidebar Inputs 
st.sidebar.header("Enter Customer Details")

tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly        = st.sidebar.number_input("Monthly Charges (₹)", 0.0, 200.0, 65.0)
total          = st.sidebar.number_input("Total Charges (₹)", 0.0, 10000.0, float(tenure * monthly))
contract       = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet       = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment        = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
tech_support   = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
online_sec     = st.sidebar.selectbox("Online Security", ["Yes", "No"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior         = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])

# Build Input DataFrame 
contract_map   = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map   = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map    = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}
yn_map         = {"No": 0, "Yes": 1}

feature_cols = pd.read_csv('data/feature_columns.csv').iloc[:, 0].tolist()

input_dict = {col: [0] for col in feature_cols}
input_dict.update({
    'tenure':              [tenure],
    'MonthlyCharges':      [monthly],
    'TotalCharges':        [total],
    'Contract':            [contract_map[contract]],
    'InternetService':     [internet_map[internet]],
    'PaymentMethod':       [payment_map[payment]],
    'TechSupport':         [yn_map[tech_support]],
    'OnlineSecurity':      [yn_map[online_sec]],
    'PaperlessBilling':    [yn_map[paperless]],
    'SeniorCitizen':       [yn_map[senior]],
})

input_df = pd.DataFrame(input_dict)[feature_cols]

# Predict Button 
if st.sidebar.button("🔍 Predict Churn", use_container_width=True):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("## ⚠️ HIGH RISK\nThis customer is **likely to churn!**")
        else:
            st.success("## ✅ LOW RISK\nThis customer is **NOT likely to churn.**")

    with col2:
        st.metric("Churn Probability", f"{probability[1]*100:.1f}%")

    with col3:
        st.metric("Retention Probability", f"{probability[0]*100:.1f}%")

    st.divider()

    # INSIGHTS SECTION 
    st.subheader("📌 Key Insights & Reasons")

    insights = []

    if contract == "Month-to-month":
        insights.append(("🔴 High Risk", "Month-to-month contracts have 3x higher churn rate. Consider offering a long-term contract discount."))
    elif contract == "One year":
        insights.append(("🟡 Medium", "One year contract reduces churn risk significantly."))
    else:
        insights.append(("🟢 Low Risk", "Two year contract customers rarely churn — great retention!"))

    if tenure < 12:
        insights.append(("🔴 High Risk", f"Customer tenure is only {tenure} months. New customers churn most in the first year."))
    elif tenure < 24:
        insights.append(("🟡 Medium", "Tenure is between 1–2 years. Engagement programs can help retain this customer."))
    else:
        insights.append(("🟢 Low Risk", f"Tenure of {tenure} months shows strong loyalty."))

    if monthly > 70:
        insights.append(("🔴 High Risk", f"Monthly charges of ₹{monthly} are high. Price-sensitive customers may leave for cheaper alternatives."))
    elif monthly > 50:
        insights.append(("🟡 Medium", f"Monthly charges of ₹{monthly} are moderate."))
    else:
        insights.append(("🟢 Low Risk", f"Low monthly charges of ₹{monthly} reduce price-based churn."))

    if internet == "Fiber optic":
        insights.append(("🔴 High Risk", "Fiber optic customers churn more — often due to high costs or better competitor offers."))

    if tech_support == "No":
        insights.append(("🔴 High Risk", "No tech support increases churn. Customers with issues and no support tend to leave."))

    if online_sec == "No":
        insights.append(("🟡 Medium", "No online security service — customers feel less value. Upselling security may help."))

    if payment == "Electronic check":
        insights.append(("🟡 Medium", "Electronic check users have higher churn than auto-pay customers."))

    for tag, text in insights:
        st.markdown(f"**{tag}** — {text}")

    st.divider()

    # FEATURE IMPORTANCE CHART 
    st.subheader("📈 What Factors Matter Most for Churn?")

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    importances.plot(kind='barh', ax=ax, color='#1F4E79')
    ax.set_title("Top 10 Churn Predictors")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    # RECOMMENDATION 
    st.divider()
    st.subheader("💡 Business Recommendation")

    if prediction == 1:
        st.warning("""
        **Action Required!** This customer is at high risk of churning. Suggested actions:
        - 📞 Proactively contact the customer with a retention offer
        - 💰 Offer a discount or upgrade to a yearly contract
        - 🛠️ Provide free tech support for 3 months
        - 🎁 Loyalty rewards or cashback on next bill
        """)
    else:
        st.info("""
        **Customer looks stable.** Suggested actions to maintain loyalty:
        - 🌟 Enroll in loyalty rewards program
        - 📧 Send satisfaction survey to understand needs
        - 📦 Offer add-on services (security, backup) for more value
        """)

else:
    st.info("👈 Fill in customer details on the left panel and click **Predict Churn** to see results and insights!")

    # Show sample EDA charts on home screen
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    try:
        with col1:
            st.image("data/churn_distribution.png", caption="Churn Distribution")
        with col2:
            st.image("data/churn_by_contract.png", caption="Churn by Contract Type")
    except:
        st.info("Run churn_eda.py first to generate charts here!")