import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define image size
IMG_SIZE = (224, 224)

# Load dataset (replace with actual dataset path)
DATASET_PATH = "A:\Infosys_Internship\Images"

# Function to load images and labels
def load_data(dataset_path):
    images = []
    labels = []
    classes = ["normal", "tumor"]  # Ensure folders are named correctly

    for label in classes:
        path = os.path.join(dataset_path, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            labels.append(0 if label == "normal" else 1)  # 0 for normal, 1 for tumor
    
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_data(DATASET_PATH)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load VGG16 for feature extraction
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers

# Function to extract features
def extract_features(model, images):
    features = model.predict(images)
    return features.reshape(features.shape[0], -1)  # Flatten

# Extract features for training and testing sets
X_train_features = extract_features(base_model, X_train)
X_test_features = extract_features(base_model, X_test)

# Train SVM classifier
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train_features, y_train)

# Evaluate the model
y_pred = svm.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained SVM model
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("SVM Model saved successfully as svm_model.pkl!")
