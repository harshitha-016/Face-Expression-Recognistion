import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import logging

# Lazy import FER so we can show a user-friendly error if it fails to load
try:
    from fer import FER
    FER_AVAILABLE = True
except Exception as e:
    FER_AVAILABLE = False
    logging.exception("FER import failed: %s", e)

def draw_boxes_and_labels(pil_image, detections):
    """
    Draw rectangles and labels on a PIL image using detection results
    (detections is a list of faces in the FER format).
    """
    draw = ImageDraw.Draw(pil_image)
    # Try to use a truetype font; fallback to default if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    for face in detections:
        box = face.get("box", None)
        emotions = face.get("emotions", {})
        if box:
            x, y, w, h = box
            # Rectangle
            draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 0, 0), width=2)
            # Dominant emotion
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                score = emotions[dominant_emotion]
                label = f"{dominant_emotion} ({score:.2f})"
                text_pos = (x, max(y - 18, 0))
                draw.text(text_pos, label, fill=(255, 255, 255), font=font)

def webcam_emotion_detection():
    st.header("Webcam Emotion Detection")

    if not FER_AVAILABLE:
        st.error(
            "The `fer` package failed to import in this environment. "
            "This often means a binary dependency (like OpenCV) didn't install correctly. "
            "Check your app logs and make sure you installed the pinned dependencies."
        )
        st.info("You can still upload an image to test the UI, but automatic detection will be disabled.")
        # Allow image upload as fallback UI
        uploaded = st.file_uploader("Upload an image (fallback)", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image (FER not available)", use_column_width=True)
        return

    # If FER is available, proceed
    st.write("Take a photo with your webcam or upload an image to detect emotions.")
    webcam_input = st.camera_input("Webcam")

    uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

    # Initialize detector lazily (so errors are easier to catch and show)
    detector = None
    try:
        detector = FER(mtcnn=True)
    except Exception as e:
        logging.exception("Failed to create FER detector: %s", e)
        st.error("Failed to initialize FER detector. See logs for details.")
        return

    # Decide which image to use (prefer webcam if present)
    source = webcam_input or uploaded_file
    if not source:
        st.info("Please capture an image with the webcam or upload one.")
        return

    # Read image bytes and convert to PIL Image
    try:
        image = Image.open(source)
    except Exception:
        # camera_input returns a UploadedFile-like object; this handles both
        source_bytes = source.getvalue() if hasattr(source, "getvalue") else source.read()
        image = Image.open(io.BytesIO(source_bytes))

    image = image.convert("RGB")
    image_np = np.array(image)  # FER expects an RGB numpy array

    # Run the detection
    try:
        results = detector.detect_emotions(image_np)
    except Exception as e:
        logging.exception("FER detection failed: %s", e)
        st.error("Emotion detection failed. Check the app logs for details.")
        st.image(image, caption="Original image", use_column_width=True)
        return

    if not results:
        st.success("No faces detected.")
        st.image(image, use_column_width=True)
        return

    # Make a copy and draw boxes/labels using Pillow (no cv2 required)
    pil_copy = image.copy()
    draw_boxes_and_labels(pil_copy, results)

    # Prepare a textual summary of detected emotions
    emotion_summaries = []
    for face in results:
        emotions = face.get("emotions", {})
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            score = emotions[dominant_emotion]
            emotion_summaries.append(f"{dominant_emotion}: {score:.2f}")

    st.subheader("Detected emotions")
    st.write("\n".join(emotion_summaries))

    st.image(pil_copy, use_column_width=True)

if __name__ == "__main__":
    webcam_emotion_detection()
