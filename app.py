import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS # Required for cross-origin requests from frontend
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 # Used for image resizing and color conversion

# --- Model Download (for deployment environment) ---
# This part is for the server environment where the model needs to be downloaded.
# On platforms like Render or Heroku, you might use a build script or
# include this in your main app.py.
# Ensure 'gdown' is installed in your server environment (pip install gdown)
import gdown

MODEL_ID = "1gVN5mHxqhXkcK_XRkneXTfXAzwjKC89r" # Your Google Drive File ID
MODEL_FILENAME = "draft2(1800).h5" # Name of your model file

# Check if model exists, if not, download it
if not os.path.exists(MODEL_FILENAME):
    print(f"Downloading model {MODEL_FILENAME} from Google Drive...")
    try:
        # gdown.download(f"https://drive.google.com/file/d/{MODEL_ID}/view?usp=sharing", MODEL_FILENAME, quiet=False)
        # The 'view?usp=sharing' link might not work directly for gdown programmatic download.
        # It's better to use the direct download link or share it publicly.
        # For programmatic access, a direct download link is usually like:
        # https://drive.google.com/uc?id={MODEL_ID}&export=download
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}&export=download", MODEL_FILENAME, quiet=False)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Exit or handle error appropriately if model download fails
        exit(1)
else:
    print(f"Model {MODEL_FILENAME} already exists.")

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to make requests

# Load the TensorFlow model
# Using @tf.function to compile the model for faster inference
@tf.function(experimental_relax_shapes=True)
def predict_colorization(img_input):
    return model(img_input)

# Load model globally when the app starts
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("TensorFlow model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    exit(1) # Exit if model cannot be loaded, as the app won't function

@app.route('/')
def home():
    return "Colorization Backend is running!"

@app.route('/colorize', methods=['POST'])
def colorize_image():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        # The frontend sends 'data:image/png;base64,....' or 'data:image/jpeg;base64,...'
        # We need to split to get only the base64 part
        image_data_b64 = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data_b64)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("L") # Ensure grayscale

        # Preprocess image for model
        # Resize to 256x256
        img_resized = image_pil.resize((256, 256))
        img_array = np.array(img_resized) / 255.0 # Normalize to [0, 1]
        img_input = np.expand_dims(img_array, axis=(0, -1)) # Shape: (1, 256, 256, 1)

        # Predict color image using the loaded model
        # Use the @tf.function compiled prediction
        color_output = predict_colorization(img_input).numpy()[0] # Get numpy array from TF tensor
        color_output = np.clip(color_output, 0, 1) # Clip values to [0, 1]

        # Convert output to displayable image (PIL Image)
        color_image_pil = Image.fromarray((color_output * 255).astype('uint8'))

        # Encode colorized image back to base64
        buffered = io.BytesIO()
        # Save as JPEG for potentially smaller size, or PNG if transparency is needed
        color_image_pil.save(buffered, format="JPEG")
        colorized_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"colorizedImage": f"data:image/jpeg;base64,{colorized_image_b64}"}), 200

    except Exception as e:
        print(f"Error during colorization: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local development:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # For production deployment (e.g., Gunicorn with Flask):
    # Use a WSGI server like Gunicorn: gunicorn -w 4 -b 0.0.0.0:8000 app:app
    # The host and port will be set by the deployment environment.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


