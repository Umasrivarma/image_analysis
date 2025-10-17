import streamlit as st
from PIL import Image

st.set_page_config(page_title="Cloud Facial Analysis Demo", layout="centered")
st.title("👤 Facial Analysis Demo (Streamlit Cloud)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Placeholders for features Cloud cannot run
    st.subheader("Face Analysis (Cloud Demo)")
    st.write("**Age:** N/A (requires PyTorch / DeepFace)")
    st.write("**Gender:** N/A (requires PyTorch / DeepFace)")
    st.write("**Emotion:** N/A (requires FER / PyTorch)")
    st.write("**Face Recognition:** N/A (requires PyTorch / DeepFace)")
    st.write("**Background Removal:** N/A (requires rembg / PyTorch)")
