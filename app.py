import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load model
@st.cache_resource
def load_colorization_model():
    return tf.keras.models.load_model("draft2(1800).h5")

model = load_colorization_model()

# Title
st.title("Black & White to Color Image Converter")
st.write("Upload a black and white image and see the colorized output using a deep learning model.")

# Upload image
uploaded_file = st.file_uploader("Upload a grayscale image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("L")
    st.subheader("Grayscale Image")
    st.image(image, use_column_width=True)

    # Resize and normalize
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 256, 256, 1)

    # Predict color image
    color_output = model.predict(img_input)[0]  # Shape: (256, 256, 3)
    color_output = np.clip(color_output, 0, 1)

    # Convert to displayable image
    color_image = Image.fromarray((color_output * 255).astype('uint8'))

    st.subheader("Colorized Output")
    st.image(color_image, use_column_width=True)
