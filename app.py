from fastapi import FastAPI
import cv2
from ultralytics import YOLO
import os
import uuid
import numpy as np
from sort_tracker import Sort   # tracking

app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Sort()

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/analyze_video")
def analyze_video():
    video_path = "data/input_video.mp4"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.join(
        OUTPUT_DIR, f"annotated_{uuid.uuid4().hex}.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = model(frame)[0]

        # Collect bird detections
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 14:  # bird class in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

        # Update tracker
        if len(detections) > 0:
            dets_np = np.array(detections)
            tracks = tracker.update(dets_np)
        else:
            tracks = []

        # Draw tracked boxes
        count = 0
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        # Overlay count
        cv2.putText(
            frame,
            f"Bird Count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return {
        "frames_processed": frame_count,
        "output_video": out_path,
        "note": "Tracking-based bird count. Weight proxy can be added using bounding box area."
    }


