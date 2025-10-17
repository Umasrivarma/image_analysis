import streamlit as st
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace
from rembg import remove
import os

st.set_page_config(page_title="Full Facial Analysis App", layout="centered")
st.title("👤 Full Facial Analysis App (Local / Server Deployment)")

# ------------------- Upload Image -------------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
bg_removal = st.checkbox("Remove background (optional)")

if uploaded_file:
    try:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Optional background removal
        if bg_removal:
            st.info("Removing background…")
            img_nobg = remove(np.array(image))
            image = Image.fromarray(img_nobg)
            st.image(image, caption="Background Removed", use_column_width=True)

        # Convert to BGR for DeepFace
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # ------------------- Face Analysis -------------------
        st.info("Analyzing face for age, gender, emotion…")
        results = DeepFace.analyze(img_bgr, actions=['age', 'gender', 'emotion'], enforce_detection=True)

        # Display results
        st.subheader("Face Analysis Results")
        st.write(f"**Age:** {results['age']}")
        st.write(f"**Gender:** {results['gender']}")
        st.write("**Emotion:**")
        st.json(results['emotion'])

        # ------------------- Face Recognition (Optional) -------------------
        st.info("Face recognition demo (requires reference folder)")

        # Example: if you have reference faces in './reference_faces'
        reference_dir = "./reference_faces"
        if os.path.exists(reference_dir) and len(os.listdir(reference_dir)) > 0:
            st.info("Matching uploaded face with reference faces…")
            df = DeepFace.find(img_path=img_bgr, db_path=reference_dir, enforce_detection=False)
            st.write(df)
        else:
            st.warning("No reference faces found in './reference_faces'")

    except Exception as e:
        st.error(f"Error processing image: {e}")
