import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("lung_cancer_model.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e3a8a,#06b6d4);
background-attachment:fixed;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.title{
font-size:52px;
font-weight:bold;
text-align:center;
color:white;
}

.subtitle{
text-align:center;
font-size:20px;
color:#dbeafe;
margin-bottom:25px;
}

div[data-testid="stVerticalBlock"]{
border-radius:18px;
}

.stButton>button{
width:100%;
height:55px;
border:none;
border-radius:15px;
font-size:20px;
font-weight:bold;
background:linear-gradient(90deg,#2563eb,#06b6d4);
color:white;
}

.stButton>button:hover{
transform:scale(1.02);
}

.result{
padding:25px;
border-radius:18px;
text-align:center;
font-size:32px;
font-weight:bold;
color:white;
margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🫁 Lung Cancer Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Predict Lung Cancer using Machine Learning (Logistic Regression)</div>", unsafe_allow_html=True)

st.divider()

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "👤 Age",
        1,
        100,
        45
    )

    smokes = st.slider(
        "🚬 Smoking Level",
        0,
        40,
        10
    )

with col2:

    areaq = st.slider(
        "🌍 Air Quality Index",
        0,
        10,
        6
    )

    alkhol = st.slider(
        "🍺 Alcohol Consumption",
        0,
        10,
        4
    )

st.divider()

# ---------------- PREDICT ----------------
if st.button("🔍 Predict Lung Cancer"):

    sample = np.array([[age, smokes, areaq, alkhol]])

    prediction = model.predict(sample)[0]

    if prediction == 1:

        st.markdown("""
        <div class='result' style='background:#dc2626;'>
        ⚠️ High Risk of Lung Cancer Detected
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='result' style='background:#16a34a;'>
        ✅ No Lung Cancer Detected
        </div>
        """, unsafe_allow_html=True)

st.divider()

st.caption("🩺 Developed using Streamlit • Scikit-learn • Logistic Regression")