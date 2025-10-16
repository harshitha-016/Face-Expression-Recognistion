import streamlit as st
import cv2
from fer import FER
import numpy as np
import matplotlib.pyplot as plt

def image_upload_emotion_detection():
    st.header("Image Emotion Detection")
    # Create a file uploader component
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        detector = FER(mtcnn=True)

        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Convert the image to RGB (required by FER library)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect emotions in the image
        results = detector.detect_emotions(image_rgb)

        # Display the image with detected emotions
        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        for result in results:
            (x, y, w, h) = result['box']
            emotions = result['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            ax.text(x, y, dominant_emotion, color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            rect = plt.Rectangle((x, y), w, h, fill=False, color='blue')
            ax.add_patch(rect)

        plt.axis('off')
        st.pyplot(fig)
