import streamlit as st
import pickle
import numpy as np
import os
from pathlib import Path

st.markdown("## 🏠 Smart House Price Predictor")
st.write("Enter details below to estimate house price")

# Get absolute path to model
base_dir = Path(__file__).parent
model_path = base_dir / 'models' / 'model.pkl'

if not model_path.exists():
    st.error(f"Error: {model_path} not found.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("House Price Predictor")

# Inputs
area = st.number_input("Enter area (sq ft)", min_value=100)
bedrooms = st.number_input("Enter number of bedrooms", min_value=1)

# Prediction
if st.button("Predict"):
    input_data = np.array([[area, bedrooms]])
    prediction = model.predict(input_data)
    if prediction[0] > 80:
        st.success("🔥 Excellent performance expected!")
    elif prediction[0] > 60:
        st.info("👍 Good performance expected")
    else:
        st.warning("⚠️ Needs improvement")
    st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.0f}")
    
st.markdown("---")
st.markdown("### 📌 About")
st.write("This app uses Machine Learning to predict outcomes based on user input.")