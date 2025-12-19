# Bird-Counting-and-Weight-Estimation

This repository presents a working prototype that analyzes poultry farm CCTV video to automatically estimate bird counts over time and provide a visual indicator of bird size. Using footage from a fixed camera, the system processes video frames to identify and track birds, generates an annotated output video, and summarizes count trends in a simple and understandable way. The focus of this project is on demonstrating a complete, practical workflow—from video input to meaningful insights—while accounting for real-world conditions commonly found in poultry farms.

## Task Objective

The system processes fixed-camera poultry CCTV footage to produce:

1. **Bird Counts Over Time** using object detection and stable tracking IDs.
2. **Bird Weight Estimation** using a visual size-based proxy derived from bounding box area.

## Approach (Brief)
<img width="1875" height="1117" alt="Method" src="https://github.com/user-attachments/assets/fa70b97c-9400-4fe6-8052-b3c3091dae2a" />

- Birds are detected in each frame using a YOLO-based model.
- SORT tracking is applied to maintain consistent IDs across frames.
- Bird counts are generated over time using tracked identities.
- A relative weight index is computed from bounding box area for each tracked bird.

## API

- `GET /health` – Service health check
- `POST /analyze_video` – Processes video and returns counts and weight proxy

## Dataset

This project uses the **Chicken Detection and Tracking** dataset from Roboflow
(Public Domain), suitable for poultry detection and tracking tasks.

## Output
<img width="1536" height="1024" alt="Detection" src="https://github.com/user-attachments/assets/d1c81b8c-0954-4106-86a1-c0cc55a94a88" />

- Annotated video with tracking IDs
- Frame-wise bird counts
- Relative weight proxy per bird


https://github.com/user-attachments/assets/a86f4170-91e1-4571-9d83-79c8f5216d1a

