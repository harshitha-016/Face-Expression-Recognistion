import streamlit as st
import cv2
from fer import FER
import tempfile
import os

def video_upload_emotion_detection():
    st.header("Video Emotion Detection")
    video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if video_file is not None:
        detector = FER(mtcnn=True)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_emotions(frame_rgb)

            for result in results:
                (x, y, w, h) = result['box']
                emotions = result['emotions']
                dominant_emotion = max(emotions, key=emotions.get)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            st.image(frame, channels="BGR")

            # Optional: Save annotated video locally (not visible in Streamlit Cloud)
            if frame_count == 0:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(os.path.join(tempfile.gettempdir(), 'output_video.avi'),
                                      fourcc, 1.0, (frame.shape[1], frame.shape[0]))
            if out:
                out.write(frame)

            frame_count += 1

        cap.release()
        if out:
            out.release()
