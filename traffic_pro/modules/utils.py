"""
modules/utils.py
-----------------
Utility helpers:
  - CSV logging of vehicle crossing events
  - FPS calculator
  - Frame annotator (FPS + status overlay)
"""

import csv
import os
import time
from datetime import datetime
from typing import Dict

import cv2
import numpy as np

# Default CSV log path
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "vehicle_log.csv")

CSV_HEADERS = ["timestamp", "vehicle_type", "track_id", "confidence", "total_count"]


# ─────────────────────────────────────────────────────────────────────────────
# CSV Logger
# ─────────────────────────────────────────────────────────────────────────────

def init_csv(path: str = LOG_FILE) -> str:
    """Create log directory and CSV file with headers if not exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
    return path


def log_event(event: dict, total: int, path: str = LOG_FILE) -> None:
    """Append one crossing event to the CSV log."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow({
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vehicle_type": event.get("label", "unknown"),
            "track_id":     event.get("track_id", -1),
            "confidence":   round(event.get("confidence", 0.0), 3),
            "total_count":  total,
        })


def read_csv(path: str = LOG_FILE) -> list[dict]:
    """Read all rows from CSV as list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def clear_csv(path: str = LOG_FILE) -> None:
    """Reset the CSV file (keep headers only)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()


# ─────────────────────────────────────────────────────────────────────────────
# FPS Calculator
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling FPS calculator."""

    def __init__(self, window: int = 30):
        self.window    = window
        self._times:   list[float] = []

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self.window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])

    def reset(self):
        self._times.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, fps: float, status: str = "Running") -> np.ndarray:
    """Draw FPS and status bar at the top of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Top status bar
    cv2.rectangle(out, (0, 0), (w, 30), (20, 20, 20), -1)
    cv2.putText(out, f"FPS: {fps:.1f}   Status: {status}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 255, 200), 1)
    return out
