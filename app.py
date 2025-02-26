import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC

# Define image size
IMG_SIZE = (224, 224)

# Load the trained SVM model
@st.cache_resource
def load_models():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze layers

    # Load the trained SVM model from a pickle file
    with open("svm_model.pkl", "rb") as f:
        svm = pickle.load(f)

    return base_model, svm

# Load models
base_model, svm = load_models()

# Function to extract features using VGG16
def extract_features(model, image):
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    features = model.predict(image)
    return features.reshape(features.shape[0], -1)  # Flatten feature vector

# Function to predict tumor probability
def predict_iris_tumor(image):
    img_array = img_to_array(image) / 255.0  # Normalize
    features = extract_features(base_model, img_array)

    prediction = svm.predict(features)
    probability = svm.predict_proba(features)[0][1]  # Probability of having a tumor

    result = "Tumor Detected" if prediction[0] == 1 else "No Tumor"
    return result, probability

# 🎨 Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F7FA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploadedImage {
        border: 2px solid #4CAF50;
        padding: 5px;
        border-radius: 10px;
        display: block;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

# 🏥 Title & Description
st.title("🔬 Iris Tumor Detection App")
st.write(
    "Upload an eye image, and our AI model will predict whether a tumor is present. "
    "This tool is built using **Deep Learning (CNN) + SVM** for accurate predictions."
)

# 📂 File Uploader
uploaded_file = st.file_uploader("📤 Upload an Eye Image...", type=["jpg", "png", "jpeg"])

# 🎭 Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        .main {
            background-color: #222;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

# 📷 Image Preview & Prediction
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])  # Create two columns for layout
    
    with col1:
        st.subheader("🖼️ Uploaded Image")
        image = load_img(uploaded_file, target_size=IMG_SIZE)
        st.image(image, caption="✅ Image Preview", use_container_width=True, output_format="JPEG", channels="RGB")

    with col2:
        st.subheader("🧪 Prediction Results")
        
        # Predict tumor probability
        result, probability = predict_iris_tumor(image)

        # 🎯 Display Results
        if result == "Tumor Detected":
            st.error(f"🚨 {result} ({probability:.2%} probability)")
        else:
            st.success(f"✅ {result} ({probability:.2%} probability)")

        # 📊 Show Probability Bar Chart
        st.write("### 📊 Probability Breakdown")
        st.progress(probability)  # Show probability bar
        st.write(f"Probability of Tumor: **{probability:.2%}**")

# ℹ️ About Section
st.sidebar.subheader("ℹ️ About This Model")
st.sidebar.write(
    "This model uses **VGG16 (CNN) for feature extraction** and **SVM for classification** "
    "to detect **iris tumors** based on uploaded eye images. "
    "Developed for **medical diagnostics & research.**"
)

# 📩 Contact Info
st.sidebar.subheader("📩 Contact Us")
st.sidebar.write("📧 Email: **dhanshripchaudhari@gmail.com**")
st.sidebar.write("🌐 Website: [AI Healthcare](https://www.aihealthcare.com)")

