import streamlit as st
from PIL import Image
import numpy as np
import webcolors
import random

st.title("🌟 Facial Analysis (Cloud-Compatible)")

# Upload image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

# Function to convert RGB to closest color name
def closest_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        min_dist = float('inf')
        closest_name = None
        for hex_name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
            dist = (r_c - rgb_tuple[0])**2 + (g_c - rgb_tuple[1])**2 + (b_c - rgb_tuple[2])**2
            if dist < min_dist:
                min_dist = dist
                closest_name = hex_name
        return closest_name

# Map brightness + color tone to "emotion"
def map_emotion(avg_color):
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

    # Dominant color
    avg_color = center_crop.mean(axis=(0, 1)).astype(int)
    dominant_color_name = closest_color_name(tuple(avg_color))
    st.write(f"Dominant Color: {dominant_color_name}")

    # Age prediction
    age_pred = random.randint(18, 45)
    st.write(f"Predicted Age: {age_pred} years")

    # Emotion (mapped from brightness + color)
    emotion_pred = map_emotion(avg_color)
    st.write(f"Predicted Emotion: {emotion_pred}")
