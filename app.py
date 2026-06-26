import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Health Checker", layout="centered")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- CSS (FINAL CLEAN UI) ----------------
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
body {
    background-color: #F4F7FB;
}

.stApp {
    background-color: #F4F7FB;
}

/* Fix spacing */
.block-container {
    padding-top: 2rem;
}

/* Force readable text */
* {
    color: #1f2d3d !important;
}

/* ---------- INPUTS ---------- */
.stNumberInput input {
    background-color: #ffffff !important;
    color: #1f2d3d !important;
    border: 1px solid #ccd6dd !important;
    border-radius: 8px !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #ccd6dd !important;
}

div[data-baseweb="select"] * {
    color: #1f2d3d !important;
    background-color: #ffffff !important;
}

/* Dropdown menu */
ul {
    background-color: #ffffff !important;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    width: 100%;
    background-color: #2E86C1 !important;
    color: white !important;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    border: none;
}

/* ---------- CARDS ---------- */
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* ---------- TITLES ---------- */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #2c3e50 !important;
}

.subtitle {
    text-align: center;
    color: #5d6d7e !important;
    margin-bottom: 20px;
}

/* ---------- RESULT ---------- */
.high-risk {
    background: #fdecea;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.low-risk {
    background: #eafaf1;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

/* ---------- METRICS ---------- */
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
}

/* ---------- PROGRESS ---------- */
.stProgress > div > div {
    background-color: #2E86C1;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>❤️ Heart Health Checker</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered heart disease risk prediction</div>", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])

with col2:
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220)
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Number of Major Vessels", [0,1,2,3,4])
    thal = st.selectbox("Thal", [0,1,2,3])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict", use_container_width=True):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    prob = 0.7 if prediction == 1 else 0.3
    heart_age = int(age + (prob * 20))

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 Result")

    if prediction == 1:
        st.markdown("""
        <div class='high-risk'>
            <h2 style='color:#e74c3c;'>⚠️ High Risk of Heart Disease</h2>
            <p>Your health data indicates a higher chance of heart disease.</p>
        </div>
        """, unsafe_allow_html=True)
        risk = "High"
        progress = 0.8
    else:
        st.markdown("""
        <div class='low-risk'>
            <h2 style='color:#27ae60;'>✅ Low Risk</h2>
            <p>Your heart health looks good. Keep maintaining a healthy lifestyle!</p>
        </div>
        """, unsafe_allow_html=True)
        risk = "Low"
        progress = 0.3

    col1, col2 = st.columns(2)
    col1.metric("❤️ Estimated Heart Age", heart_age)
    col2.metric("📊 Risk Level", risk)

    st.subheader("📈 Risk Indicator")
    st.progress(progress)

    st.subheader("💡 Recommendations")

    if prediction == 1:
        st.write("• Exercise regularly 🏃")
        st.write("• Reduce cholesterol intake 🥗")
        st.write("• Avoid smoking 🚭")
        st.write("• Consult a doctor 🩺")
    else:
        st.write("• Maintain a healthy diet 🥗")
        st.write("• Stay physically active 💪")
        st.write("• Regular health checkups 🩺")

    st.markdown("</div>", unsafe_allow_html=True)

    st.success("✨ Thank you for using Heart Health Checker")