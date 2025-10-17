import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from fer import FER
from rembg import remove
import tempfile

st.set_page_config(page_title="Facial Analysis Cloud App", layout="centered")
st.title("👤 Facial Analysis App (Streamlit Cloud Compatible)")

# ------------------- Upload Image -------------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
bg_removal = st.checkbox("Remove background (optional)")

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Optional background removal
        if bg_removal:
            st.info("Removing background…")
            img_nobg = remove(np.array(image))
            image = Image.fromarray(img_nobg)
            st.image(image, caption="Background Removed", use_column_width=True)
        else:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # ------------------- Emotion Detection -------------------
        st.info("Analyzing emotion…")
        emotion_detector = FER(mtcnn=False)  # mtcnn=False avoids cv2 dependency
        img_array = np.array(image)
        emotions = emotion_detector.detect_emotions(img_array)

        if emotions:
            st.subheader("Detected Emotions:")
            st.json(emotions[0]["emotions"])
        else:
            st.warning("No face detected for emotion recognition.")

        # ------------------- Age & Gender (Dummy example) -------------------
        # Use placeholder predictions for demo (replace with real models if you have)
        st.subheader("Age & Gender (Demo)")
        st.write("**Age:** 25 (example, replace with model)")  
        st.write("**Gender:** Male (example, replace with model)")

    except Exception as e:
        st.error(f"Error processing image: {e}")
