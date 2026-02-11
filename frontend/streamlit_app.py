import streamlit as st
import requests
import os

API_URL = os.getenv(
    "API_URL",
    "https://ci-cd-pipeleine2.onrender.com/predict"
)

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Predict house price using a Linear Regression model")

area = st.number_input("Area (sq ft)", 100, 5000, 1200)
bedrooms = st.number_input("Bedrooms", 1, 10, 2)

if st.button("Predict Price"):
    payload = {"area": area, "bedrooms": bedrooms}

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"Predicted Price: ₹ {price:,.2f}")
        else:
            st.error("Backend error")

    except Exception as e:
        st.error(f"Connection error: {e}")