import streamlit as st
import cv2
from fer import FER
from PIL import Image
import numpy as np

def webcam_emotion_detection():
    st.header("Webcam Emotion Detection")
    # Create a webcam input component
    webcam_input = st.camera_input("Webcam")

    # Create a button to start the emotion detection
    if webcam_input:
        start_button = st.button("Start Emotion Detection")

        # Create an empty placeholder for detected emotions
        emotion_placeholder = st.empty()

        if start_button:
            detector = FER(mtcnn=True)

            # Convert the webcam input to an image
            image = Image.open(webcam_input)
            image_np = np.array(image)

            # Convert the image to RGB (FER library requires RGB images)
            rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Detect emotions in the frame
            result = detector.detect_emotions(rgb_frame)

            # Loop through detected faces
            emotions = []
            for face in result:
                (x, y, w, h) = face['box']
                # Get the dominant emotion
                dominant_emotion = max(face['emotions'], key=face['emotions'].get)
                score = face['emotions'][dominant_emotion]
                emotions.append(f"{dominant_emotion} ({score:.2f})")
                # Draw rectangle around the face
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Display the emotion on the frame
                cv2.putText(image_np, f'{dominant_emotion} ({score:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Display the detected emotions in the placeholder
            emotion_placeholder.text("\n".join(emotions))

            # Display the resulting frame
            st.image(image_np, channels="BGR")

webcam_emotion_detection()
