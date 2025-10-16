import streamlit as st
import cv2
from fer import FER
import platform

def live_emotion_detection():
    st.header("Live Emotion Detection")

    # Detect if running in Streamlit Cloud (Linux-based headless)
    if platform.system() == "Linux":
        st.warning("Live webcam detection is not supported on Streamlit Cloud. Please run locally.")
        return

    frame_placeholder = st.empty()
    start_button = st.button("Start Live Emotion Detection", key="start_button")

    if start_button:
        detector = FER(mtcnn=True)
        cap = cv2.VideoCapture(0)

        def draw_text(frame, text, x, y):
            font_scale = 0.6
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x
            text_y = y - text_size[1]
            cv2.rectangle(frame, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, text, (text_x, y), font, font_scale, (255, 255, 255), font_thickness)

        stop_flag = False

        def stop():
            nonlocal stop_flag
            stop_flag = True

        st.button("Stop Live Emotion Detection", key="stop_button", on_click=stop)

        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_emotions(rgb_frame)

            for face in result:
                (x, y, w, h) = face['box']
                dominant_emotion = max(face['emotions'], key=face['emotions'].get)
                score = face['emotions'][dominant_emotion]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                draw_text(frame, f'{dominant_emotion} ({score:.2f})', x, y - 10)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
