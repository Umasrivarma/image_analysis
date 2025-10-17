import streamlit as st
from PIL import Image
import numpy as np
from rembg import remove

st.set_page_config(page_title="Cloud Facial Analysis", layout="centered")
st.title("👤 Facial Analysis Demo (Streamlit Cloud)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
bg_removal = st.checkbox("Remove background (optional)")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Optional background removal
    if bg_removal:
        st.info("Removing background…")
        img_nobg = remove(np.array(image))
        image = Image.fromarray(img_nobg)
        st.image(image, caption="Background Removed", use_column_width=True)
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Placeholders for features Cloud cannot support
    st.subheader("Face Analysis (Cloud Demo)")
    st.write("**Age:** N/A (requires PyTorch / DeepFace)")
    st.write("**Gender:** N/A (requires PyTorch / DeepFace)")
    st.write("**Emotion:** N/A (requires FER / PyTorch)")
    st.write("**Face Recognition:** N/A (requires PyTorch / DeepFace)")
