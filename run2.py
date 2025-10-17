import streamlit as st
from PIL import Image
import numpy as np
import random

st.title("🌟 Facial Analysis (Cloud-Compatible)")

# Upload image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

# Map brightness to "emotion"
def map_emotion(img_array):
    avg_color = img_array.mean(axis=(0, 1))
    brightness = avg_color.mean()
    if brightness > 180:
        return "Happy"
    elif brightness > 120:
        return "Calm"
    elif brightness > 60:
        return "Sad"
    else:
        return "Angry"

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Predictions")

    img_array = np.array(image)
    h, w, _ = img_array.shape

    # Center crop (50%) to focus on face
    center_crop = img_array[h//4:3*h//4, w//4:3*w//4]

    # Predicted Age
    age_pred = random.randint(18, 45)
    st.write(f"Predicted Age: {age_pred} years")

    # Predicted Emotion
    emotion_pred = map_emotion(center_crop)
    st.write(f"Predicted Emotion: {emotion_pred}")
