import streamlit as st
import platform
import importlib
import logging
from types import ModuleType
from typing import Optional

logging.basicConfig(level=logging.INFO)

def lazy_import(module_name: str) -> Optional[ModuleType]:
    """
    Try to import a module by name. Return the module or None on failure.
    We log the exception so it's visible in Streamlit logs.
    """
    try:
        module = importlib.import_module(module_name)
        logging.info("Successfully imported %s", module_name)
        return module
    except Exception as e:
        logging.exception("Failed to import module %s: %s", module_name, e)
        return None

def call_mode_function(module_name: str, func_name: str):
    """
    Import the module lazily and call the function named func_name.
    If import or function lookup fails, show a Streamlit-friendly error.
    """
    mod = lazy_import(module_name)
    if mod is None:
        st.error(
            f"Failed to load `{module_name}`. This usually means a package failed to install "
            "or a binary dependency is missing. Check the app logs (Manage app → Logs) and "
            "verify your `requirements.txt` and `packages.txt`."
        )
        st.stop()
    func = getattr(mod, func_name, None)
    if not callable(func):
        st.error(f"Module `{module_name}` does not provide function `{func_name}`.")
        st.stop()
    # Call the function (they are expected to handle Streamlit UI internally)
    try:
        func()
    except Exception as e:
        logging.exception("Error while running %s.%s: %s", module_name, func_name, e)
        st.error("An error occurred while running the selected mode. Check the logs for details.")

# App header
st.set_page_config(page_title="Emotion Recognition App", layout="centered")
st.title("Emotion Recognition App")
st.write("Detect emotions from webcam, uploaded images or videos.")

# Create a sidebar for mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ("Webcam", "Image Upload", "Video Upload", "Live Emotion Detection")
)

# Detect platform (Streamlit Cloud is Linux headless; we check for that)
is_cloud = platform.system() == "Linux"

# Mode dispatch
if mode == "Webcam":
    if is_cloud:
        st.warning("Webcam access is not supported on Streamlit Cloud. Please run the app locally for webcam support.")
        st.info("You can use 'Image Upload' mode to test the detection in this environment.")
    else:
        # Lazily import and call webcam_emotion_detection
        call_mode_function("webcam_emotion_detection", "webcam_emotion_detection")

elif mode == "Image Upload":
    # This mode should work on Cloud or locally
    # If the module is missing, show helpful UI fallback (upload and display)
    mod = lazy_import("image_upload_emotion_detection")
    if mod is None:
        st.warning("Image detection module is not available. You can still upload an image to view it.")
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded:
            from PIL import Image
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)
    else:
        call_mode_function("image_upload_emotion_detection", "image_upload_emotion_detection")

elif mode == "Video Upload":
    mod = lazy_import("video_upload_emotion_detection")
    if mod is None:
        st.error("Video upload detection module failed to load. Check logs and dependencies.")
    else:
        call_mode_function("video_upload_emotion_detection", "video_upload_emotion_detection")

elif mode == "Live Emotion Detection":
    # Live webcam is not supported on Streamlit Cloud (headless)
    if is_cloud:
        st.warning("Live webcam detection is not supported on Streamlit Cloud. Please run locally for live detection.")
    else:
        call_mode_function("live_emotion_detection", "live_emotion_detection")
