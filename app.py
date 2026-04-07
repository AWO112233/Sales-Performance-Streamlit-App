import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Sales Intelligence",
    layout="wide"
)

# -------------------------------
# LOAD MODELS
# -------------------------------
stack_model = joblib.load("stack_model.pkl")
sem_model = joblib.load("sem_model.pkl")

# -------------------------------
# CUSTOM STYLE (THIS IS THE MAGIC)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.metric-container {
    background-color: #1C1F26;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("🚀 ASDA Sales Performance Intelligence")
st.caption("Machine Learning + SEM powered decision system")

# -------------------------------
# LAYOUT: INPUT LEFT | OUTPUT RIGHT
# -------------------------------
left, right = st.columns([1, 2])

# ===============================
# INPUT PANEL
# ===============================
with left:

    st.subheader("🎯 Input Variables")

    preset = st.selectbox(
        "Quick Scenario",
        ["Custom", "Low", "Average", "High"]
    )

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

    st.markdown("### 📥 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    run = st.button("🚀 Run Analysis", use_container_width=True)

# ===============================
# OUTPUT PANEL
# ===============================
with right:

    if run:

        st.subheader("📊 Prediction Overview")

        prediction = stack_model.predict(input_df)[0]

        probs = stack_model.predict_proba(input_df)[0]
        confidence = np.max(probs)

        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction", prediction)
        col2.metric("Confidence", f"{confidence:.2f}")
        col3.metric("Risk Level", "High" if probs[1] > 0.7 else "Moderate")

        st.markdown("---")

        # ---------------- PROBABILITY VISUAL ----------------
        st.subheader("📈 Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(["Low Performance", "High Performance"], probs)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("🧠 Key Drivers")

        try:
            importances = stack_model.named_estimators_['rf'].feature_importances_
            feat_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(feat_df.set_index("Feature"))

        except:
            st.info("Feature importance not available")

        # ---------------- SEM ----------------
        st.subheader("🔵 SEM Output")

        try:
            sem_out = sem_model.predict(input_df)
            st.dataframe(sem_out)
        except:
            try:
                st.dataframe(sem_model.inspect())
            except:
                st.warning("SEM output unavailable")

        # ---------------- SCENARIO ----------------
        st.subheader("🔬 Impact Simulation")

        scenario = input_df.copy()
        scenario["CRM"] += 1

        base = stack_model.predict_proba(input_df)[0][1]
        improved = stack_model.predict_proba(scenario)[0][1]

        st.write(f"📌 Increasing CRM by +1 changes probability from **{base:.2f} → {improved:.2f}**")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("AI Sales Intelligence System | Built for Strategic Decision-Making")