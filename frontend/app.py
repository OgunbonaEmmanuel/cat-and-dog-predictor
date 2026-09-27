import streamlit as st
import requests

st.header("CAT OR DOG?")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    st.image(uploaded_file) #Displays the image

    files = {
        "file": uploaded_file.getvalue()
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files=files
    )

    result = response.json()

    st.success(result["prediction"])

    st.write(result)