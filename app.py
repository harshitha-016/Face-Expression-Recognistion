import streamlit as st
from webcam_emotion_detection import webcam_emotion_detection
from image_upload_emotion_detection import image_upload_emotion_detection
from video_upload_emotion_detection import video_upload_emotion_detection
from live_emotion_detection import live_emotion_detection
# Import the new function

# Create a Streamlit app
st.title("Emotion Recognition App")
st.write("This app detects emotions from your webcam feed, an uploaded image, or a video.")

# Create a sidebar for mode selection
mode = st.sidebar.selectbox("Select Mode", ("Webcam", "Image Upload", "Video Upload", "Live Emotion Detection"))  # Add the new option

# Call the appropriate function based on the selected mode
if mode == "Webcam":
    webcam_emotion_detection()
elif mode == "Image Upload":
    image_upload_emotion_detection()
elif mode == "Video Upload":
    video_upload_emotion_detection()
elif mode == "Live Emotion Detection":  # Call the new function
    live_emotion_detection()
