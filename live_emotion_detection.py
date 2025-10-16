import streamlit as st
import cv2
from fer import FER

def live_emotion_detection():
    st.header("Live Emotion Detection")

    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()

    # Create a button to start/stop the live emotion detection
    start_button = st.button("Start Live Emotion Detection", key="start_button")

    if start_button:
        detector = FER(mtcnn=True)

        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        def draw_text(frame, text, x, y):
            """Draw text with background."""
            font_scale = 0.6
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x
            text_y = y - text_size[1]
            cv2.rectangle(frame, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, text, (text_x, y), font, font_scale, (255, 255, 255), font_thickness)

        stop_flag = False

        # Define a function to stop the loop
        def stop():
            nonlocal stop_flag
            stop_flag = True

        st.button("Stop Live Emotion Detection", key="stop_button", on_click=stop)

        while not stop_flag:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB (FER library requires RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect emotions in the frame
            result = detector.detect_emotions(rgb_frame)

            # Loop through detected faces
            for face in result:
                (x, y, w, h) = face['box']
                # Get the dominant emotion
                dominant_emotion = max(face['emotions'], key=face['emotions'].get)
                score = face['emotions'][dominant_emotion]
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Display the emotion on the frame
                draw_text(frame, f'{dominant_emotion} ({score:.2f})', x, y - 10)

            # Display the resulting frame
            frame_placeholder.image(frame, channels="BGR")

        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

live_emotion_detection()
