import streamlit as st
import requests
from PIL import Image

API_URL = "https://pneumonia-detection-from-chest-x-rays-eo0p.onrender.com/predict"  # change after deployment

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🫁 Pneumonia Detection")
st.write("Upload a chest X-ray image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            files = {"image": uploaded_file.getvalue()}
            response = requests.post(
    API_URL,
    files={"image": uploaded_file.getvalue()}
)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
        else:
            st.error("Prediction failed")
