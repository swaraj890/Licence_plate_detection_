import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import easyocr
from PIL import Image

# Load the Keras Model (.h5 file)
model = tf.keras.models.load_model("car_plate_detector.h5")

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

# Function to preprocess image for the model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to model's input size
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to detect license plate
def detect_license_plate(image):
    original_image = np.array(image)
    processed_image = preprocess_image(original_image)

    # Predict bounding box coordinates (x1, y1, x2, y2)
    bbox = model.predict(processed_image)[0]  # Assuming the model outputs bounding box

    x1, y1, x2, y2 = [int(coord * original_image.shape[1]) for coord in bbox]  # Scale back

    # Crop the detected license plate
    plate_img = original_image[y1:y2, x1:x2]

    # Extract text using EasyOCR
    detected_text = reader.readtext(plate_img, detail=0)

    # Draw bounding box & text
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original_image, " ".join(detected_text), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return original_image, detected_text

# Streamlit UI
st.title("🚗 License Plate Detection (Keras Model)")
st.write("Upload an image, and the model will detect the license plate.")

# Image Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Detect license plate
    processed_image, extracted_text = detect_license_plate(image_np)

    # Show results
    st.image(processed_image, caption="Detected License Plate", use_column_width=True)
    st.write(f"**Extracted Text:** {', '.join(extracted_text)}")

st.write("👨‍💻 **Developed by Swaraj**")
