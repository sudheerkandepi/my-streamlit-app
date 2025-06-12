import streamlit as st
from deepface import DeepFace
import os
import cv2
from PIL import Image
import numpy as np

# Set your folder of known faces
KNOWN_IMAGES_FOLDER = "/Users/sudheer.kandepi/Downloads/Thumbnails"

# Ensure the folder exists
os.makedirs(KNOWN_IMAGES_FOLDER, exist_ok=True)

st.title("🔍 Face Match with DeepFace")
st.write("Upload a photo and compare it to faces in your folder.")

uploaded_file = st.file_uploader("Upload an image to compare", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    # Save temp uploaded image
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    uploaded_image_np = np.array(uploaded_image)

    matches = []

    with st.spinner("Comparing with known images..."):
        for filename in os.listdir(KNOWN_IMAGES_FOLDER):
            image_path = os.path.join(KNOWN_IMAGES_FOLDER, filename)

            try:
                result = DeepFace.verify(
                    img1_path=uploaded_image_np,
                    img2_path=image_path,
                    enforce_detection=False,
                    model_name="VGG-Face"
                )

                if result["verified"]:
                    matches.append((filename, result["distance"]))
            except Exception as e:
                st.warning(f"Skipped {filename}: {e}")

    if matches:
        matches.sort(key=lambda x: x[1])
        st.success(f"✅ Found {len(matches)} match(es)")
        for fname, dist in matches:
            st.image(os.path.join(KNOWN_IMAGES_FOLDER, fname), caption=f"{fname} (distance: {dist:.2f})", width=200)
    else:
        st.error("❌ No matching faces found.")
