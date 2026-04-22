"""
modules/detector.py
--------------------
YOLOv8 vehicle detector — CPU optimised.
Detects: car, motorcycle, bus, truck.
Frames are resized to 640x360 before inference for speed.
"""

import cv2
import numpy as np
from typing import List

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# COCO class IDs for vehicles only
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Colours for drawing (BGR)
CLASS_COLORS = {
    "car":        (0, 200, 255),
    "motorcycle": (0, 255, 100),
    "bus":        (255, 140,   0),
    "truck":      (180,   0, 255),
}

# Inference resolution (width, height) — smaller = faster on CPU
INFER_W, INFER_H = 640, 360


class VehicleDetector:
    """
    YOLOv8 wrapper for CPU-only vehicle detection.

    Args:
        model_path : Path to .pt weights (auto-downloads yolov8n.pt)
        conf       : Minimum confidence threshold
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        if not YOLO_AVAILABLE:
            raise RuntimeError("Install ultralytics: pip install ultralytics")
        self.model = YOLO(model_path)
        self.conf  = conf

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Run detection on a BGR frame.
        Internally resizes to INFER_W x INFER_H for speed.

        Returns list of dicts:
            bbox       : (x1, y1, x2, y2) in ORIGINAL frame coords
            label      : vehicle class string
            confidence : float 0-1
        """
        orig_h, orig_w = frame.shape[:2]

        # Resize for faster inference
        small = cv2.resize(frame, (INFER_W, INFER_H))
        sx = orig_w / INFER_W
        sy = orig_h / INFER_H

        results = self.model(small, verbose=False, device="cpu")[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < self.conf:
                continue

            # Scale bbox back to original resolution
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * sx); y1 = int(y1 * sy)
            x2 = int(x2 * sx); y2 = int(y2 * sy)

            detections.append({
                "bbox":       (x1, y1, x2, y2),
                "label":      VEHICLE_CLASSES[cls_id],
                "confidence": round(conf, 3),
            })

        return detections

    @staticmethod
    def draw(frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        """Draw tracked bounding boxes with ID labels."""
        out = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            color = CLASS_COLORS.get(t["label"], (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"#{t['track_id']} {t['label']} {t['confidence']:.2f}"
            cv2.putText(out, label, (x1, max(y1-8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
            cv2.circle(out, t["centroid"], 4, color, -1)
        return out
