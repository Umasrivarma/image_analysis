import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from deepface import DeepFace
from rembg import remove

st.set_page_config(page_title="Face Analysis App", layout="centered")
st.title("👤 Facial Analysis App")

# ------------------- Image Upload -------------------
uploaded_file = st.file_uploader("Upload an image with a face", type=["jpg", "jpeg", "png"])

bg_removal = st.checkbox("Remove background (optional)")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Optional background removal
        if bg_removal:
            st.info("Removing background…")
            img_nobg = remove(img_array)
            st.image(Image.fromarray(img_nobg), caption="Background Removed", use_column_width=True)
            img_bgr = cv2.cvtColor(np.array(img_nobg), cv2.COLOR_RGB2BGR)

        # ------------------- Face Analysis -------------------
        st.info("Analyzing face for age, gender, emotion…")
        results = DeepFace.analyze(img_bgr, actions=['age', 'gender', 'emotion'], enforce_detection=True)

        # Display results
        st.subheader("Face Analysis Results")
        st.write(f"**Age:** {results['age']}")
        st.write(f"**Gender:** {results['gender']}")
        st.write("**Emotion:**")
        st.json(results['emotion'])

        # ------------------- Optional: Face Recognition -------------------
        st.info("Face recognition requires a reference database (not included in this example).")

    except Exception as e:
        st.error(f"Error processing image: {e}")
