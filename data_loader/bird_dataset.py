import os
import cv2

class BirdVideoDataset:
    """
    Dataset loader for fixed-camera poultry CCTV videos.

    Uses the Roboflow Chicken Detection and Tracking dataset
    for bird detection, tracking, and size-based analysis.
    """

    def __init__(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def metadata(self):
        return {
            "video_path": self.video_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "dataset": "Roboflow Chicken Detection and Tracking (v2)"
        }

Add dataset loader for poultry video processing
