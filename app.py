import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="BDA Sales Performance Artefact", layout="wide")

# -------------------------------
# LOAD MODELS
# -------------------------------
stack_model = joblib.load("stack_model.pkl")
sem_model = joblib.load("sem_model.pkl")

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Big Data Analytics & Sales Performance System")
st.markdown("""
This system operationalises the research model by combining:
- **Machine Learning (Prediction)**
- **Structural Equation Modelling (Explanation)**
- **Scenario Simulation (Managerial Insight)**

It directly supports the study objectives on BDA, CRM, and Sales Performance.
""")

# -------------------------------
# LAYOUT
# -------------------------------
left, right = st.columns([1, 2])

# ===============================
# INPUT SECTION
# ===============================
with left:

    st.subheader("📥 Input Variables")

    preset = st.selectbox("Scenario Preset", ["Custom", "Low", "Average", "High"])

    def preset_val(p):
        return {"Low": 2.0, "Average": 3.0, "High": 4.5}.get(p, 3.0)

    d = preset_val(preset)

    def slider(label):
        return st.slider(label, 1.0, 5.0, d)

    input_dict = {
        "Self_Efficacy": slider("Self Efficacy"),
        "Playfulness": slider("Playfulness"),
        "Social_Norms": slider("Social Norms"),
        "Voluntariness": slider("Voluntariness"),
        "User_Involvement": slider("User Involvement"),
        "User_Participation": slider("User Participation"),
        "Management_Support": slider("Management Support"),
        "Relative_Advantage": slider("Relative Advantage"),
        "Results_Demonstrability": slider("Results Demonstrability"),
        "Image": slider("Image"),
        "Compatibility": slider("Compatibility"),
        "Professional_Fit": slider("Professional Fit"),
        "Complexity": slider("Complexity"),
        "Job_Fit": slider("Job Fit"),
        "Visibility": slider("Visibility"),
        "CRM": slider("CRM")
    }

    input_df = pd.DataFrame([input_dict])
    st.dataframe(input_df)

    run = st.button("🚀 Run Analysis")

# ===============================
# OUTPUT SECTION
# ===============================
with right:

    if run:

        st.subheader("📊 Model Prediction")

        prediction = stack_model.predict(input_df)[0]
        probs = stack_model.predict_proba(input_df)[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Class", prediction)
        col2.metric("Probability (High Performance)", f"{probs[1]:.2f}")
        col3.metric("Confidence", f"{np.max(probs):.2f}")

        # -------------------------------
        # INTERPRETATION (VERY IMPORTANT)
        # -------------------------------
        st.subheader("🧠 Interpretation")

        if probs[1] > 0.7:
            st.success("High likelihood of strong sales performance. This aligns with strong CRM and organisational support.")
        elif probs[1] > 0.5:
            st.warning("Moderate performance expected. Improvements in CRM and Job Fit may enhance outcomes.")
        else:
            st.error("Low performance predicted. Results suggest weak organisational or CRM capability.")

        # -------------------------------
        # PROBABILITY VISUAL
        # -------------------------------
        st.subheader("📈 Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(["Low", "High"], probs)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # -------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------
        st.subheader("📊 Key Drivers (ML)")

        try:
            importances = stack_model.named_estimators_['rf'].feature_importances_
            feat_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(feat_df.set_index("Feature"))

            top_feats = feat_df.head(3)["Feature"].tolist()
            st.info(f"Top drivers influencing prediction: {', '.join(top_feats)}")

        except:
            st.warning("Feature importance not available for stacking model.")

        # -------------------------------
        # SEM OUTPUT (THEORY VALIDATION)
        # -------------------------------
        st.subheader("🔵 SEM Output (Theoretical Model)")

        try:
            sem_out = sem_model.predict(input_df)
            st.dataframe(sem_out)

            st.markdown("""
            This reflects the structural model results, where CRM and organisational variables 
            are the strongest predictors of sales performance.
            """)

        except:
            st.warning("SEM output unavailable")

        # -------------------------------
        # SCENARIO ANALYSIS
        # -------------------------------
        st.subheader("🔬 Scenario Analysis")

        feature = st.selectbox("Adjust Variable", input_df.columns)

        scenario = input_df.copy()
        scenario[feature] += 1

        base = probs[1]
        new = stack_model.predict_proba(scenario)[0][1]

        st.write(f"Changing **{feature}** by +1 changes probability from **{base:.2f} → {new:.2f}**")

        st.markdown("""
        This demonstrates how managerial actions (e.g., improving CRM or Management Support)
        can influence predicted sales performance outcomes.
        """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Research Artefact: BDA, CRM, and Sales Performance | ML + SEM Integrated Framework")