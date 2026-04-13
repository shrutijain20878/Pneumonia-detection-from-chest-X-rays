import streamlit as st
import requests

API_URL = "https://pneumonia-detection-from-chest-x-rays-eo0p.onrender.com/predict"  # change after deployment

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🫁 Pneumonia Detection")
st.write("Upload a chest X-ray image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: **{result['prediction']}**")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Request failed: {e}")
