# Bird-Counting-and-Weight-Estimation

This repository contains a working prototype developed for the
Machine Learning / AI Engineer Internship technical assignment at
Kuppismart Solutions (Livestockify).

## Task Objective

The system processes fixed-camera poultry CCTV footage to produce:

1. **Bird Counts Over Time** using object detection and stable tracking IDs.
2. **Bird Weight Estimation** using a visual size-based proxy derived from bounding box area.

## Approach (Brief)

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

- Annotated video with tracking IDs
- Frame-wise bird counts
- Relative weight proxy per bird
