import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from deepface import DeepFace

# Load pre-trained models (ensure these files are in the correct paths)
model = load_model("model.h5")
face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define face detection function
def face_fun(img):
    faces = face_cas.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        crop_face = img[y:y+h, x:x+w]
    return crop_face

# Function to predict emotion and name
def analyze_and_predict(face):
    try:
        # Analyze emotion
        result = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
        emotion = result[0]["dominant_emotion"] if result else "No Emotion found"

        # Prepare face for prediction
        face_resized = cv2.resize(face, (150, 150))
        img_array = np.array(Image.fromarray(face_resized, "RGB"))
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        # Assign name based on prediction
        name = "Aditya" if pred[0][0] < 0.5 else "Mayank"
        return emotion, name
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        return "Error", "Unknown"

# Streamlit App
st.title("Real-time Face Detection and Emotion Recognition")

# Image upload section
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and convert uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display uploaded image
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Detect face and predict emotion and name
    face = face_fun(img)
    if face is not None:
        emotion, name = analyze_and_predict(face)

        # Display predictions
        st.write(f"Detected Emotion: {emotion}")
        st.write(f"Predicted Name: {name}")

        # Draw face rectangle and predictions
        cv2.putText(img_rgb, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img_rgb, f"Name: {name}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        st.image(img_rgb, caption="Processed Image", use_column_width=True)
    else:
        st.write("No face found in the image.")
else:
    st.write("Upload an image to start.")

