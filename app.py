import streamlit as st
import platform

from webcam_emotion_detection import webcam_emotion_detection
from image_upload_emotion_detection import image_upload_emotion_detection
from video_upload_emotion_detection import video_upload_emotion_detection
from live_emotion_detection import live_emotion_detection

# Create a Streamlit app
st.title("Emotion Recognition App")
st.write("This app detects emotions from your webcam feed, an uploaded image, or a video.")

# Create a sidebar for mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ("Webcam", "Image Upload", "Video Upload", "Live Emotion Detection")
)

# Detect if running in Streamlit Cloud (Linux-based headless)
is_cloud = platform.system() == "Linux"

# Call the appropriate function based on the selected mode
if mode == "Webcam":
    if is_cloud:
        st.warning("Webcam access is not supported on Streamlit Cloud. Please run locally.")
    else:
        webcam_emotion_detection()

elif mode == "Image Upload":
    image_upload_emotion_detection()

elif mode == "Video Upload":
    video_upload_emotion_detection()

elif mode == "Live Emotion Detection":
    if is_cloud:
        st.warning("Live webcam detection is not supported on Streamlit Cloud. Please run locally.")
    else:
        live_emotion_detection()
