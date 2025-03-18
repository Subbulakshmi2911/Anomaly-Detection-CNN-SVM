import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import joblib
from PIL import Image
import pandas as pd

# Define Image Size
IMAGE_SIZE = (128, 128)

# Load Pretrained ResNet50 (Feature Extractor)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Load Trained One-Class SVM Model
svm_model = joblib.load("svm_model.pkl")

# Function to Extract Features
def extract_features(img):
    """Ensure consistent preprocessing for file paths and Streamlit uploads."""
    
    if isinstance(img, str):  
        img = image.load_img(img, target_size=IMAGE_SIZE)
    else:
        img = img.convert("RGB")  # Ensure RGB mode
        img = img.resize((128, 128), Image.BICUBIC)  # Resize to match training data
        img = np.array(img, dtype=np.float32) # Convert to NumPy float32
    
    # Debug: Print Image Stats Before Preprocessing
    st.text(f"📌 Debug: Image Details Before Preprocessing\n"
            f"Image Shape: {img.shape}\n"
            f"Min Pixel Value: {img.min()} | Max Pixel Value: {img.max()}")

    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.resnet50.preprocess_input(img)  # Normalize

    # Debug: Print Image Stats After Preprocessing
    st.text(f"📌 Debug: Image Details After Preprocessing\n"
            f"Processed Image Shape: {img.shape}\n"
            f"Processed Min Pixel Value: {img.min()} | Processed Max Pixel Value: {img.max()}")

    return feature_extractor.predict(img).flatten()

# Function to Predict Anomaly
def detect_anomaly(img):
    """Predict if the image is an anomaly."""
    features = extract_features(img)
    
    # Save features for debugging
    np.save("debug_features.npy", features)

    prediction = svm_model.predict([features])  # 1 = Normal, -1 = Anomaly
    return features, prediction  # Return features along with prediction

# Streamlit UI
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("🔍 CNN Feature Extraction + One-Class SVM Anomaly Detection")

st.sidebar.image("logo.png", width=200)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Upload & Predict"])

if menu == "Home":
    st.subheader("About the Project")
    st.write("""
    This anomaly detection system uses **deep learning (CNN feature extraction)** and 
    **One-Class SVM** to detect anomalies **without needing defective images for training**.
    """)

elif menu == "Upload & Predict":
    st.subheader("Upload an Image for Anomaly Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_obj = Image.open(uploaded_file)  # Open image
        st.image(image_obj, caption="Uploaded Image", use_column_width=True)

        with st.spinner('🔎 Extracting Features & Analyzing...'):
            features, prediction = detect_anomaly(image_obj)  # Pass `image_obj`, not `uploaded_file`

     

        # Show Prediction
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("✅ Good Image")
        else:
            st.error("❌ Defective Image Detected")

st.sidebar.info("Developed with ❤️ using Streamlit")
