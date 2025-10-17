import streamlit as st
from PIL import Image
import numpy as np
from rembg import remove

st.title("Cloud Facial Analysis Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
bg_removal = st.checkbox("Remove background")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    if bg_removal:
        img_nobg = remove(np.array(image))
        image = Image.fromarray(img_nobg)
        st.image(image, caption="Background removed")
    else:
        st.image(image, caption="Uploaded image")

    st.subheader("Age, Gender, Emotion, Face Recognition")
    st.write("All analysis placeholders (cannot run on Streamlit Cloud)")
