import streamlit as st
import cv2
import os
import uuid
import numpy as np
import random

# ------------------ STREAMLIT SETUP ------------------
st.set_page_config(page_title="Bird Analytics Demo", layout="wide")

st.title("Bird Counting & Weight Estimation (Prototype)")
st.write(
    "Prototype demonstrating bird counting over time and relative weight "
    "estimation using bounding box area. Detection is mocked for demo purposes."
)

# ------------------ DIRECTORIES ------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ SIMPLE TRACKER ------------------
class SimpleTracker:
    def __init__(self):
        self.next_id = 1

    def update(self, detections):
        tracks = []
        for det in detections:
            x1, y1, x2, y2 = det
            tracks.append([x1, y1, x2, y2, self.next_id])
            self.next_id += 1
        return tracks

tracker = SimpleTracker()

# ------------------ MOCK DETECTION ------------------
def mock_bird_detection(frame):
    """
    Simulates bird detections.
    Returns random bounding boxes.
    """
    h, w, _ = frame.shape
    detections = []

    num_birds = random.randint(2, 5)
    for _ in range(num_birds):
        x1 = random.randint(0, w - 150)
        y1 = random.randint(0, h - 150)
        x2 = x1 + random.randint(60, 140)
        y2 = y1 + random.randint(60, 140)
        detections.append([x1, y1, x2, y2])

    return detections

# ------------------ FILE UPLOAD ------------------
uploaded_video = st.file_uploader(
    "Upload poultry CCTV video (.mp4)",
    type=["mp4"]
)

if uploaded_video:
    input_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success("Video uploaded successfully")

    if st.button("Analyze Video"):
        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join(
            OUTPUT_DIR, f"annotated_{uuid.uuid4().hex}.mp4"
        )

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_count = 0
        weight_index_data = {}

        progress = st.progress(0)
        status = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = mock_bird_detection(frame)
            tracks = tracker.update(detections)
            bird_count = len(tracks)

            for track in tracks:
                x1, y1, x2, y2, track_id = track
                area = (x2 - x1) * (y2 - y1)
                weight_index = round(area / 1000, 2)
                weight_index_data[track_id] = weight_index

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} W:{weight_index}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Bird Count: {bird_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            out.write(frame)
            frame_count += 1

            progress.progress(min(frame_count % 100 / 100, 1.0))
            status.text(f"Processing frame {frame_count}")

        cap.release()
        out.release()

        st.success("Analysis completed")

        st.subheader("Annotated Output Video")
        st.video(output_path)

        st.subheader("Relative Weight Index (Proxy)")
        st.json(weight_index_data)

        st.info(
            "NOTE: Bird detection is mocked for local demo. "
            "In production, YOLO-based detection runs in a Linux environment."
        )
