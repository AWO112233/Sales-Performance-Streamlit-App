import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# LOAD MODELS
# -------------------------------
stack_model = joblib.load("stack_model.pkl")
sem_model = joblib.load("sem_model.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sales Performance & SEM Predictor",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Sales Performance Prediction System")
st.markdown("Predict sales performance using CRM & behavioral constructs using ML and SEM models.")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("Input Features")

def user_input():
    data = {
        "Self_Efficacy": st.sidebar.slider("Self Efficacy (SE)", 1.0, 5.0, 3.0),
        "Playfulness": st.sidebar.slider("Playfulness (P)", 1.0, 5.0, 3.0),
        "Social_Norms": st.sidebar.slider("Social Norms (SN)", 1.0, 5.0, 3.0),
        "Voluntariness": st.sidebar.slider("Voluntariness (VN)", 1.0, 5.0, 3.0),
        "User_Involvement": st.sidebar.slider("User Involvement (UI)", 1.0, 5.0, 3.0),
        "User_Participation": st.sidebar.slider("User Participation (UP)", 1.0, 5.0, 3.0),
        "Management_Support": st.sidebar.slider("Management Support (MS)", 1.0, 5.0, 3.0),
        "Relative_Advantage": st.sidebar.slider("Relative Advantage (RAD)", 1.0, 5.0, 3.0),
        "Results_Demonstrability": st.sidebar.slider("Results Demonstrability (RD)", 1.0, 5.0, 3.0),
        "Image": st.sidebar.slider("Image (I)", 1.0, 5.0, 3.0),
        "Compatibility": st.sidebar.slider("Compatibility (C)", 1.0, 5.0, 3.0),
        "Professional_Fit": st.sidebar.slider("Professional Fit (PF)", 1.0, 5.0, 3.0),
        "Complexity": st.sidebar.slider("Complexity (CLX)", 1.0, 5.0, 3.0),
        "Job_Fit": st.sidebar.slider("Job Fit (JF)", 1.0, 5.0, 3.0),
        "Visibility": st.sidebar.slider("Visibility (V)", 1.0, 5.0, 3.0),
        "CRM": st.sidebar.slider("CRM", 1.0, 5.0, 3.0)
    }

    return pd.DataFrame([data])

input_df = user_input()

# -------------------------------
# DISPLAY INPUT
# -------------------------------
st.subheader("📥 Input Data")
st.write(input_df)

# -------------------------------
# PREDICTIONS
# -------------------------------
if st.button("Predict"):

    # -------- Stacking ML Model --------
    st.subheader("🟢 Stacking Model Prediction")
    prediction = stack_model.predict(input_df)
    confidence = (
        np.max(stack_model.predict_proba(input_df)) 
        if hasattr(stack_model, "predict_proba") else "N/A"
    )
    st.success(f"Predicted Class: {prediction[0]}")
    st.info(f"Confidence: {confidence}")

    if hasattr(stack_model, "feature_importances_"):
        st.subheader("Feature Importance (Stacking Model)")
        st.bar_chart(stack_model.feature_importances_)

    # -------- SEM Model --------
    st.subheader("🔵 SEM Model Output")
    try:
        # Display SEM path estimates / latent scores
        sem_scores = sem_model.predict(input_df)
        st.write(sem_scores)
    except:
        try:
            st.dataframe(sem_model.inspect())
        except:
            st.warning("SEM prediction/inspection not available — check SEM library API.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Models: Stacking Classifier & SEM | Built for CRM & Sales Analysis")