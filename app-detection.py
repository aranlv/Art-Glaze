import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
from skimage.feature import hog
import base64
import io

# Load models and scalers
@st.cache_resource
def load_model_and_scaler():
    with open('artifacts/svm_model_color.pkl', 'rb') as file:
        svm_model_color = pickle.load(file)
    with open('artifacts/svm_model_hog.pkl', 'rb') as file:
        svm_model_hog = pickle.load(file)
    with open('artifacts/svm_model_sift_orb.pkl', 'rb') as file:
        svm_model_sift_orb = pickle.load(file)
    with open('artifacts/scaler_color.pkl', 'rb') as file:
        scaler_color = pickle.load(file)
    with open('artifacts/scaler_hog.pkl', 'rb') as file:
        scaler_hog = pickle.load(file)
    with open('artifacts/scaler_sift_orb.pkl', 'rb') as file:
        scaler_sift_orb = pickle.load(file)
    return svm_model_color, svm_model_hog, svm_model_sift_orb, scaler_color, scaler_hog, scaler_sift_orb

svm_model_color, svm_model_hog, svm_model_sift_orb, scaler_color, scaler_hog, scaler_sift_orb = load_model_and_scaler()

# Helper functions
def extract_color_histogram(image):
    chans = cv2.split(image)
    hist = [cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten() for chan in chans]
    return np.concatenate(hist)

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_sift_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # SIFT
    sift = cv2.SIFT_create()
    keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)

    # ORB
    orb = cv2.ORB_create()
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)

    max_descriptors = 512

    if descriptors_sift is not None:
        descriptors_sift = descriptors_sift[:max_descriptors]
    else:
        descriptors_sift = np.zeros((max_descriptors, 128))

    if descriptors_orb is not None:
        descriptors_orb = descriptors_orb[:max_descriptors]
    else:
        descriptors_orb = np.zeros((max_descriptors, 32))

    descriptors_sift = np.pad(descriptors_sift, ((0, max_descriptors - descriptors_sift.shape[0]), (0, 0)), mode='constant')
    descriptors_orb = np.pad(descriptors_orb, ((0, max_descriptors - descriptors_orb.shape[0]), (0, 0)), mode='constant')

    combined_descriptors = np.hstack((descriptors_sift.flatten(), descriptors_orb.flatten()))
    return combined_descriptors

def preprocess_image(image, image_size=(256, 256)):
    # Resize and convert the image to RGB
    image = image.resize(image_size)
    img_np = np.array(image)

    # Extract features
    color_features = extract_color_histogram(img_np)
    hog_features = extract_hog_features(img_np)
    sift_orb_features = extract_sift_orb_features(img_np)

    # Standardize features
    color_features = scaler_color.transform([color_features])
    hog_features = scaler_hog.transform([hog_features])
    sift_orb_features = scaler_sift_orb.transform([sift_orb_features])

    return color_features, hog_features, sift_orb_features

def predict_image_class(image):
    # Preprocess the image
    color_features, hog_features, sift_orb_features = preprocess_image(image)

    # Make predictions
    prediction_color = svm_model_color.predict(color_features)
    prediction_hog = svm_model_hog.predict(hog_features)
    prediction_sift_orb = svm_model_sift_orb.predict(sift_orb_features)

    return prediction_color[0], prediction_hog[0], prediction_sift_orb[0]

# Streamlit UI
st.title("Image Classification")
st.write("Upload an image to classify it as **Glazed** or **Unglazed**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Map predictions to readable labels
def map_prediction_label(prediction):
    return "Glazed" if prediction == 1 else "Unglazed"

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    image_base64 = image_to_base64(image)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{image_base64}" style="max-height: 300px;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Classify"):
        st.write("Processing...")
        try:
            # Predict class
            prediction_color, prediction_hog, prediction_sift_orb = predict_image_class(image)

            # Display predictions
            st.subheader("Predictions:")
            st.write(f"**Color Histogram Model:** {map_prediction_label(prediction_color)}")
            st.write(f"**HOG Features Model:** {map_prediction_label(prediction_hog)}")
            st.write(f"**SIFT+ORB Features Model:** {map_prediction_label(prediction_sift_orb)}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
