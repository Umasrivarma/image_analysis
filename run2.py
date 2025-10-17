import streamlit as st
from PIL import Image
import numpy as np
from fer import FER
from rembg import remove


try:
    from fer import FER
    st.success("FER installed successfully!")
except ModuleNotFoundError as e:
    st.error(f"FER not installed: {e}")

st.set_page_config(page_title="Cloud Facial Analysis", layout="centered")
st.title("👤 Facial Analysis (Streamlit Cloud Version)")

# ------------------- Upload Image -------------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
bg_removal = st.checkbox("Remove background (optional)")

if uploaded_file:
    try:
        # Open image
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
        detector = FER(mtcnn=False)  # mtcnn=False avoids cv2 dependency
        emotions = detector.detect_emotions(np.array(image))

        if emotions:
            st.subheader("Detected Emotions:")
            st.json(emotions[0]["emotions"])
        else:
            st.warning("No face detected for emotion recognition.")

        # ------------------- Age, Gender, Face Recognition Placeholders -------------------
        st.subheader("Age, Gender, Face Recognition (Not available on Cloud)")
        st.write("**Age:** N/A")
        st.write("**Gender:** N/A")
        st.write("**Face Recognition:** N/A")

    except Exception as e:
        st.error(f"Error processing image: {e}")

