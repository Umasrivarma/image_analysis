import streamlit as st
from PIL import Image
import numpy as np
import random

st.title("🌟 Demo Facial Analysis (Cloud-Compatible)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Predictions (Demo / Cloud-Friendly)")

    # 1️⃣ Dominant color of image
    img_array = np.array(image)
    avg_color = img_array.mean(axis=(0, 1))
    st.write(f"**Average Color (RGB):** {avg_color.astype(int)}")

    # 2️⃣ Demo age prediction
    age_demo = random.randint(18, 45)
    st.write(f"**Predicted Age:** {age_demo} years (demo)")

    # 3️⃣ Demo gender prediction
    gender_demo = random.choice(["Male", "Female"])
    st.write(f"**Predicted Gender:** {gender_demo} (demo)")
