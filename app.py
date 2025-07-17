import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import io

# Title
st.title("Black & White Image Colorizer")
st.write("Upload a black-and-white photo to colorize it using a deep learning model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((256, 256))
    img_array = img_to_array(image) / 255.0
    img_lab = rgb2lab(img_array)
    img_l = img_lab[:, :, 0]
    img_l = img_l.reshape(1, 256, 256, 1)

    # Load model
    model = load_model("draft2(1800).h5", compile=False)

    # Predict ab channels
    output = model.predict(img_l)
    output *= 128

    # Combine L with ab
    result = np.zeros((256, 256, 3))
    result[:, :, 0] = img_lab[:, :, 0]
    result[:, :, 1:] = output[0]
    colorized_img = lab2rgb(result)

    # Show result
    st.image(colorized_img, caption="Colorized Image", use_column_width=True)
