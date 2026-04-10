import os
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="ML Predictor", page_icon="🤖")

st.markdown("## 🤖 AI Prediction App")
st.write("Enter details below to get smart predictions")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "model2.pkl")

if not os.path.exists(model_path):
    st.error(f"Error: {model_path} not found.")
    st.write("Files here:", os.listdir(BASE_DIR))
    st.stop()

model = pickle.load(open(model_path, "rb"))

st.title("Student performance predictor")

study_hours = st.slider("Study Hours", 0, 12, 4)
sleep_hours = st.slider("Sleep Hours", 0, 12, 6)

if st.button("Predict"):
    input_data = np.array([[study_hours, sleep_hours]])
    prediction = model.predict(input_data)
    if prediction[0] > 80:
        st.success("🔥 Excellent performance expected!")
    elif prediction[0] > 60:
        st.info("👍 Good performance expected")
    else:
        st.warning("⚠️ Needs improvement")
    st.success(f"Estimated performance:{prediction[0]:,.0f}")

st.markdown("---")
st.markdown("### 📌 About")
st.write("This app uses Machine Learning to predict outcomes based on user input.")