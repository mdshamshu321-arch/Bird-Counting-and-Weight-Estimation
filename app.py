import streamlit as st
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="Poultry Analytics (No YOLO)", layout="wide")
st.title("Poultry Detection – Classical CV (VS Code)")

uploaded = st.file_uploader("Upload poultry video", type=["mp4", "avi"])

# ---------------- SIMPLE TRACKER ----------------
next_id = 0
objects = {}

def assign_id(cx, cy):
    global next_id
    for oid, (ox, oy) in objects.items():
        if abs(cx - ox) < 60 and abs(cy - oy) < 60:
            objects[oid] = (cx, cy)
            return oid
    objects[next_id] = (cx, cy)
    next_id += 1
    return next_id - 1

AVG_CHICKEN_AREA = 1800  # relative calibration

if uploaded:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())

    cap = cv2.VideoCapture(temp.name)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)

    frame_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 7)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        live_count = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 900 or area > 15000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2

            obj_id = assign_id(cx, cy)
            weight_index = round((w * h) / AVG_CHICKEN_AREA, 2)

            live_count += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
            cv2.putText(
                frame,
                f"ID:{obj_id} W:{weight_index}",
                (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,0),
                2
            )

        cv2.putText(
            frame,
            f"Live Count: {live_count}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        frame_box.image(frame, channels="BGR")

    cap.release()
