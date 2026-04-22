"""
modules/counter.py
-------------------
Virtual line crossing counter.
Each unique track_id is counted EXACTLY ONCE when its centroid
crosses the virtual line.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class LineCounter:
    """
    Counts vehicles crossing a virtual horizontal line.

    Args:
        line_y   : Y-coordinate of the counting line (pixels)
        frame_w  : Frame width (for drawing)
    """

    def __init__(self, line_y: int, frame_w: int):
        self.line_y  = line_y
        self.frame_w = frame_w

        # track_id → last Y position
        self._prev_y: Dict[int, int] = {}

        # track_ids already counted
        self._counted: set = set()

        # Per-class counts
        self.counts: Dict[str, int] = {
            "car": 0, "motorcycle": 0, "bus": 0, "truck": 0
        }
        self.total = 0

    # ------------------------------------------------------------------ #

    def update(self, tracks: List[dict]) -> List[dict]:
        """
        Check each track for a line crossing event.

        Returns list of crossing events (dicts with 'crossed'=True).
        """
        events = []

        for t in tracks:
            tid      = t["track_id"]
            _, cy    = t["centroid"]
            prev_y   = self._prev_y.get(tid)
            self._prev_y[tid] = cy

            if prev_y is None or tid in self._counted:
                continue

            # Crossing = centroid moves from one side of line to the other
            crossed = (prev_y < self.line_y <= cy) or (prev_y > self.line_y >= cy)

            if crossed:
                self._counted.add(tid)
                label = t["label"]
                self.counts[label] = self.counts.get(label, 0) + 1
                self.total += 1
                events.append({**t, "crossed": True})

        return events

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting line and live counts on the frame."""
        out = frame.copy()

        # Counting line
        cv2.line(out, (0, self.line_y), (self.frame_w, self.line_y),
                 (0, 255, 255), 2)
        cv2.putText(out, "COUNTING LINE", (10, self.line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Live count overlay (top-left box)
        labels = [
            f"Cars:        {self.counts.get('car', 0)}",
            f"Motorcycles: {self.counts.get('motorcycle', 0)}",
            f"Buses:       {self.counts.get('bus', 0)}",
            f"Trucks:      {self.counts.get('truck', 0)}",
            f"TOTAL:       {self.total}",
        ]
        box_w, box_h = 260, len(labels) * 22 + 14
        cv2.rectangle(out, (8, 35), (8 + box_w, 35 + box_h), (0, 0, 0), -1)
        cv2.rectangle(out, (8, 35), (8 + box_w, 35 + box_h), (50, 50, 50), 1)

        for i, txt in enumerate(labels):
            color = (0, 255, 200) if i < 4 else (0, 255, 255)
            cv2.putText(out, txt, (16, 35 + 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        return out

    def reset(self):
        """Reset all counts and tracking state."""
        self._prev_y.clear()
        self._counted.clear()
        self.counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.total  = 0
