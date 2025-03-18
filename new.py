import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Define Image Size
IMAGE_SIZE = (128, 128)

# Load the trained model
try:
    model = load_model("svm_model.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Function to preprocess the uploaded image
def preprocess_image(image):
    """Convert PIL image to model-friendly format."""
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize(IMAGE_SIZE)  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict defect
def predict_defect(image, model):
    """Predict whether the image is defective or good."""
    prediction = model.predict(image)[0][0]  # No extra batch dimension here

    # Display results
    st.subheader("Prediction Result")
    if prediction > 0.5:
        st.error("❌ Defective Image Detected!")
    else:
        st.success("✅ Good Image")

# Streamlit UI
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("🔍 Anomaly Detection System")

# Sidebar Navigation
st.sidebar.image("logo.png", width=200)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Upload & Predict"])

if menu == "Home":
    st.subheader("About the Project")
    st.write("This project uses an **Autoencoder** to detect anomalies in images.")

elif menu == "Upload & Predict":
    st.subheader("Upload an Image for Anomaly Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image first
        processed_image = preprocess_image(image)

        with st.spinner('Analyzing the image...'):
            predict_defect(processed_image, model)  # Pass processed image
        
st.sidebar.info("Developed with ❤️ using Streamlit")
