import streamlit as st
from predict import predict_image

st.title("🍔 Food Recognition System and Calorie Estimation")

uploaded_image = st.file_uploader("Image loaded", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    st.image("temp_image.jpg", caption="Image chargée", use_column_width=True)
    st.write("**Result :**")
    predict_image("temp_image.jpg")
