import streamlit as st
import cv2
from fer import FER
import tempfile

def video_upload_emotion_detection():
    st.header("Video Emotion Detection")
    # Create a file uploader component for video
    video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if video_file is not None:
        detector = FER(mtcnn=True)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        # Load the video
        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB (required by FER library)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect emotions in the frame
            results = detector.detect_emotions(frame_rgb)

            # Process and display the results
            for result in results:
                (x, y, w, h) = result['box']
                emotions = result['emotions']
                dominant_emotion = max(emotions, key=emotions.get)

                # Annotate the frame with the dominant emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Display the frame with annotations
            st.image(frame, channels="BGR")

            # Save the annotated frame to an output video
            if frame_count == 0:
                # Initialize the video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('output_video.avi', fourcc, 1.0, (frame.shape[1], frame.shape[0]))
            out.write(frame)

            frame_count += 1

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
