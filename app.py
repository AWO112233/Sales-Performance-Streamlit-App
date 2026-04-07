import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# LOAD MODELS
# -------------------------------
stack_model = joblib.load("stack_model.pkl")
sem_model = joblib.load("sem_model.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Sales Performance Intelligence",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("🚀 AI Sales Performance Intelligence System")
st.markdown("Advanced analytics platform combining **Machine Learning + SEM insights**")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Input Configuration")

preset = st.sidebar.selectbox(
    "Select Scenario",
    ["Custom", "Low Performance", "Average", "High Performance"]
)

def get_preset_values(preset):
    if preset == "Low Performance":
        return 2.0
    elif preset == "High Performance":
        return 4.5
    else:
        return 3.0

default_val = get_preset_values(preset)

def user_input():
    data = {
        "Self_Efficacy": st.sidebar.slider("Self Efficacy", 1.0, 5.0, default_val),
        "Playfulness": st.sidebar.slider("Playfulness", 1.0, 5.0, default_val),
        "Social_Norms": st.sidebar.slider("Social Norms", 1.0, 5.0, default_val),
        "Voluntariness": st.sidebar.slider("Voluntariness", 1.0, 5.0, default_val),
        "User_Involvement": st.sidebar.slider("User Involvement", 1.0, 5.0, default_val),
        "User_Participation": st.sidebar.slider("User Participation", 1.0, 5.0, default_val),
        "Management_Support": st.sidebar.slider("Management Support", 1.0, 5.0, default_val),
        "Relative_Advantage": st.sidebar.slider("Relative Advantage", 1.0, 5.0, default_val),
        "Results_Demonstrability": st.sidebar.slider("Results Demonstrability", 1.0, 5.0, default_val),
        "Image": st.sidebar.slider("Image", 1.0, 5.0, default_val),
        "Compatibility": st.sidebar.slider("Compatibility", 1.0, 5.0, default_val),
        "Professional_Fit": st.sidebar.slider("Professional Fit", 1.0, 5.0, default_val),
        "Complexity": st.sidebar.slider("Complexity", 1.0, 5.0, default_val),
        "Job_Fit": st.sidebar.slider("Job Fit", 1.0, 5.0, default_val),
        "Visibility": st.sidebar.slider("Visibility", 1.0, 5.0, default_val),
        "CRM": st.sidebar.slider("CRM", 1.0, 5.0, default_val)
    }
    return pd.DataFrame([data])

input_df = user_input()

# -------------------------------
# MAIN TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Insights", "🔬 Scenario Analysis"])

# ===============================
# TAB 1: PREDICTION
# ===============================
with tab1:

    st.subheader("📥 Input Overview")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Run Prediction"):

        prediction = stack_model.predict(input_df)[0]

        if hasattr(stack_model, "predict_proba"):
            probs = stack_model.predict_proba(input_df)[0]
            confidence = np.max(probs)
        else:
            probs = None
            confidence = None

        # ---------------- KPI CARDS ----------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted Class", prediction)
        col2.metric("Confidence", f"{confidence:.2f}" if confidence else "N/A")
        col3.metric("Model Used", "Stacking Ensemble")

        # ---------------- Probability Chart ----------------
        if probs is not None:
            st.subheader("📊 Prediction Probabilities")

            fig, ax = plt.subplots()
            ax.bar(["Class 0", "Class 1"], probs)
            ax.set_title("Prediction Probability Distribution")
            st.pyplot(fig)

        # ---------------- SEM OUTPUT ----------------
        st.subheader("🔵 SEM Insights")
        try:
            sem_output = sem_model.predict(input_df)
            st.dataframe(sem_output)
        except:
            try:
                st.dataframe(sem_model.inspect())
            except:
                st.warning("SEM output not available")

# ===============================
# TAB 2: FEATURE INSIGHTS
# ===============================
with tab2:

    st.subheader("📈 Feature Importance")

    try:
        importances = stack_model.named_estimators_['rf'].feature_importances_
        features = input_df.columns

        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"))

    except:
        st.info("Feature importance not directly available for stacking model.")

    st.subheader("🧠 Interpretation")
    st.markdown("""
    - High-impact variables (e.g., CRM, Job Fit) drive predictions  
    - Organizational factors dominate over technical ones  
    - Supports SEM findings → consistency across methods  
    """)

# ===============================
# TAB 3: SCENARIO ANALYSIS
# ===============================
with tab3:

    st.subheader("🔬 What-If Scenario Analysis")

    scenario_1 = input_df.copy()
    scenario_2 = input_df.copy()

    scenario_2["CRM"] += 1
    scenario_2["Management_Support"] += 1

    pred1 = stack_model.predict_proba(scenario_1)[0][1]
    pred2 = stack_model.predict_proba(scenario_2)[0][1]

    comparison = pd.DataFrame({
        "Scenario": ["Current", "Improved CRM + Management"],
        "Probability of High Performance": [pred1, pred2]
    })

    st.dataframe(comparison)

    fig, ax = plt.subplots()
    ax.bar(comparison["Scenario"], comparison["Probability of High Performance"])
    ax.set_title("Scenario Impact Comparison")
    st.pyplot(fig)

# -------------------------------
# DOWNLOAD
# -------------------------------
st.markdown("---")
st.subheader("📥 Export Results")

csv = input_df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Input Data",
    csv,
    "input_data.csv",
    "text/csv"
)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Advanced AI System | ML + SEM + Scenario Intelligence")