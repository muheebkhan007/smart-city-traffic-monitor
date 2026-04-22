"""
modules/tracker.py
------------------
Lightweight SORT (Simple Online and Realtime Tracking) implementation.
Uses Kalman Filter + Hungarian Algorithm for vehicle tracking.
100% CPU based — no GPU required.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter for single object tracking
# ─────────────────────────────────────────────────────────────────────────────

class KalmanBox:
    """
    Kalman Filter tracker for one bounding box.
    State: [x, y, w, h, dx, dy, dw, dh]
    """
    count = 0  # global ID counter

    def __init__(self, bbox):
        # State transition matrix
        self.kf_F = np.eye(8, 8)
        for i in range(4):
            self.kf_F[i, i + 4] = 1.0

        # Measurement matrix
        self.kf_H = np.eye(4, 8)

        # Covariance matrices
        self.kf_R = np.eye(4) * 10.0          # measurement noise
        self.kf_P = np.eye(8) * 100.0         # initial uncertainty
        self.kf_Q = np.eye(8)                 # process noise
        self.kf_Q[4:, 4:] *= 0.01

        # State vector: [cx, cy, w, h, 0, 0, 0, 0]
        x, y, w, h = self._to_xywh(bbox)
        self.x = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=float)

        self.id         = KalmanBox.count
        KalmanBox.count += 1
        self.hits       = 1
        self.no_losses  = 0
        self.active     = True

    # ── Kalman predict ────────────────────────────────────────────────────
    def predict(self):
        self.x = self.kf_F @ self.x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.no_losses += 1
        return self._to_xyxy()

    # ── Kalman update ─────────────────────────────────────────────────────
    def update(self, bbox):
        x, y, w, h = self._to_xywh(bbox)
        z = np.array([[x], [y], [w], [h]], dtype=float)
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.kf_H @ self.x)
        self.kf_P = (np.eye(8) - K @ self.kf_H) @ self.kf_P
        self.no_losses = 0
        self.hits += 1

    # ── Helpers ───────────────────────────────────────────────────────────
    def _to_xyxy(self):
        cx, cy, w, h = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    @staticmethod
    def _to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def iou(boxA, boxB):
    """Intersection over Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    denom = aA + aB - inter
    return inter / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SORT Tracker
# ─────────────────────────────────────────────────────────────────────────────

class SORTTracker:
    """
    SORT multi-object tracker.

    Args:
        max_age     : Frames to keep a track alive without a match
        min_hits    : Minimum hits before a track is confirmed
        iou_thresh  : IoU threshold for matching
    """

    def __init__(self, max_age=10, min_hits=2, iou_thresh=0.25):
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thresh = iou_thresh
        self.trackers: list[KalmanBox] = []
        KalmanBox.count = 0             # reset IDs on new session

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Match detections to existing tracks.

        Args:
            detections: list of dicts with keys bbox, label, confidence

        Returns:
            list of dicts enriched with track_id and centroid
        """
        # ── Predict all existing tracks ───────────────────────────────────
        predicted = [t.predict() for t in self.trackers]

        # ── Build IoU cost matrix ─────────────────────────────────────────
        results = []
        if detections and self.trackers:
            det_boxes  = [d["bbox"] for d in detections]
            cost       = np.zeros((len(self.trackers), len(det_boxes)))
            for ti, pred in enumerate(predicted):
                for di, db in enumerate(det_boxes):
                    cost[ti, di] = 1.0 - iou(pred, db)

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_t = set()
            matched_d = set()

            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] < (1.0 - self.iou_thresh):
                    self.trackers[ti].update(detections[di]["bbox"])
                    matched_t.add(ti)
                    matched_d.add(di)

            # New tracks for unmatched detections
            for di, det in enumerate(detections):
                if di not in matched_d:
                    self.trackers.append(KalmanBox(det["bbox"]))

        elif detections:
            for det in detections:
                self.trackers.append(KalmanBox(det["bbox"]))

        # ── Build output ──────────────────────────────────────────────────
        det_map = {tuple(map(int, d["bbox"])): d for d in detections}

        active_trackers = []
        for ti, trk in enumerate(self.trackers):
            if trk.no_losses > self.max_age:
                continue
            if trk.hits < self.min_hits and trk.no_losses > 0:
                active_trackers.append(trk)
                continue

            pred_box = [int(v) for v in trk._to_xyxy()]
            x1,y1,x2,y2 = pred_box
            cx, cy = (x1+x2)//2, (y1+y2)//2

            # Find matching detection for label/confidence
            best_det = None
            best_iou = 0
            for det in detections:
                sc = iou(pred_box, det["bbox"])
                if sc > best_iou:
                    best_iou, best_det = sc, det

            label = best_det["label"] if best_det else "vehicle"
            conf  = best_det["confidence"] if best_det else 0.0

            results.append({
                "track_id":   trk.id,
                "bbox":       (x1, y1, x2, y2),
                "centroid":   (cx, cy),
                "label":      label,
                "confidence": conf,
            })
            active_trackers.append(trk)

        self.trackers = active_trackers
        return results

    def reset(self):
        self.trackers = []
        KalmanBox.count = 0
