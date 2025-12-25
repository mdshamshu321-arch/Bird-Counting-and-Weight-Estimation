import streamlit as st
import cv2
import numpy as np
import tempfile

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="Poultry Analytics (Classical CV)", layout="wide")
st.title("Poultry Detection & Tracking – Classical Computer Vision")

uploaded = st.file_uploader("Upload poultry CCTV video", type=["mp4", "avi"])

# ---------------- TRACKER STATE ----------------
next_id = 0
objects = {}  # id -> (x, y, last_seen_frame)

MAX_DISTANCE = 60        # centroid match threshold
MAX_MISSING_FRAMES = 30  # remove stale IDs

def assign_id(cx, cy, frame_idx):
    """
    Assigns a stable ID using centroid proximity.
    Removes stale IDs automatically.
    """
    global next_id, objects

    # Clean up stale objects
    objects = {
        oid: data for oid, data in objects.items()
        if frame_idx - data[2] <= MAX_MISSING_FRAMES
    }

    for oid, (ox, oy, _) in objects.items():
        if abs(cx - ox) < MAX_DISTANCE and abs(cy - oy) < MAX_DISTANCE:
            objects[oid] = (cx, cy, frame_idx)
            return oid

    # Assign new ID
    objects[next_id] = (cx, cy, frame_idx)
    next_id += 1
    return next_id - 1


# ---------------- WEIGHT PROXY ----------------
AVG_CHICKEN_AREA = 1800  # relative calibration constant

# ---------------- VIDEO PROCESSING ----------------
if uploaded:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())

    cap = cv2.VideoCapture(temp.name)
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=50, detectShadows=False
    )

    frame_box = st.empty()
    chart_placeholder = st.empty()

    counts_over_time = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ---------------- FOREGROUND EXTRACTION ----------------
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 7)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        live_count = 0

        for c in contours:
            area = cv2.contourArea(c)

            # Area filtering removes noise & merged blobs
            if area < 900 or area > 15000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2

            obj_id = assign_id(cx, cy, frame_idx)

            # Weight proxy (relative index)
            weight_index = round((w * h) / AVG_CHICKEN_AREA, 2)

            live_count += 1

            # ---------------- VISUALIZATION ----------------
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"ID:{obj_id}  W:{weight_index}",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

        counts_over_time.append(live_count)

        cv2.putText(
            frame,
            f"Live Count: {live_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        frame_box.image(frame, channels="BGR")

        # Update chart live
        if frame_idx % 10 == 0:
            chart_placeholder.line_chart(counts_over_time)

    cap.release()

    # ---------------- FINAL OUTPUT ----------------
    st.subheader("Bird Count Over Time")
    st.line_chart(counts_over_time)

    st.markdown("""
    **Notes:**
    - Detection is motion-based using background subtraction.
    - Tracking uses centroid proximity with ID expiration.
    - Weight is a **relative proxy index**, not real mass.
    - Designed for fixed-camera poultry environments.
    """)
