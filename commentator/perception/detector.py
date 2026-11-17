"""People and ball detection (Ultralytics YOLO), with ByteTrack run on the people only.

Model choice was measured, not assumed. On 200 SoccerNet GSR frames with ground truth:

    model                       people found   ball found   CPU s/frame
    Soccana YOLO11n @1280            94%           39%          0.12   <- default
    martinjolif YOLO11m @960         83%           36%          0.43
    uisikdag YOLOv8 @1280            58%           15%          0.32

The ball is the weak link for every pretrained model we could test. Tracking is deliberately
applied to people only: a tracker throws away low-confidence boxes, and most ball detections
are low-confidence. Ball candidates are passed through untouched for the ball picker.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

from ..core import BALL, GOALKEEPER, PLAYER, REFEREE, Detection

_ALIASES = {"player": PLAYER, "person": PLAYER, "ball": BALL, "referee": REFEREE,
            "goalkeeper": GOALKEEPER}
_PEOPLE = (PLAYER, GOALKEEPER, REFEREE)


class YoloDetector:
    def __init__(self, weights: Path, imgsz: int = 1280, conf: float = 0.15,
                 device: Optional[str] = None, track_seconds: float = 2.0,
                 frame_rate: float = 12.5):
        from ultralytics import YOLO  # imported lazily so the rule engine needs no torch
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.model = YOLO(str(weights))
        self.names = {k: _ALIASES.get(str(v).lower(), str(v).lower())
                      for k, v in self.model.names.items()}
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.half = bool(device) and str(device).startswith("cuda")  # half precision on GPU
        self.people_ids = [k for k, v in self.names.items() if v in _PEOPLE]
        self.tracker = BYTETracker(SimpleNamespace(
            tracker_type="bytetrack", track_high_thresh=0.3, track_low_thresh=0.1,
            new_track_thresh=0.4, track_buffer=max(1, int(round(track_seconds * frame_rate))),
            match_thresh=0.8, fuse_score=True))

    def detect(self, frame: np.ndarray, track: bool = True) -> List[Detection]:
        result = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                            device=self.device, half=self.half)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        boxes = boxes.cpu().numpy()
        cls = boxes.cls.astype(int)
        out: List[Detection] = []

        is_person = np.isin(cls, self.people_ids)
        if not track:
            for xyxy, conf, c in zip(boxes.xyxy[is_person], boxes.conf[is_person],
                                     cls[is_person]):
                out.append(Detection(self.names[c], float(conf), tuple(map(float, xyxy))))
        elif is_person.any():
            for row in self.tracker.update(boxes[is_person], frame):
                x1, y1, x2, y2, tid, score, c = row[:7]
                out.append(Detection(self.names.get(int(c), PLAYER), float(score),
                                     (float(x1), float(y1), float(x2), float(y2)), int(tid)))

        for xyxy, conf, c in zip(boxes.xyxy[~is_person], boxes.conf[~is_person],
                                 cls[~is_person]):
            if self.names.get(c) == BALL:
                out.append(Detection(BALL, float(conf), tuple(map(float, xyxy))))
        return out

    def probe_ball(self, frame: np.ndarray, points_px, size: int = 256, zoom: int = 4,
                   conf: float = 0.15) -> List[Detection]:
        """Look for the ball in magnified crops around given pixels.

        A ball waiting at a far corner flag or on the penalty spot is about ten pixels wide:
        too small in the whole frame, easy once magnified. The calibration tells us where
        those spots are, so we only magnify there.
        """
        import cv2

        h, w = frame.shape[:2]
        crops, origins = [], []
        for x, y in points_px:
            if not (0 <= x < w and 0 <= y < h) or w < size or h < size:
                continue
            x0 = int(np.clip(x - size / 2, 0, w - size))
            y0 = int(np.clip(y - size / 2, 0, h - size))
            crops.append(cv2.resize(frame[y0:y0 + size, x0:x0 + size], (size * zoom, size * zoom),
                                    interpolation=cv2.INTER_CUBIC))
            origins.append((x0, y0))
        if not crops:
            return []
        out: List[Detection] = []
        results = self.model(crops, imgsz=size * zoom, conf=conf, verbose=False,
                             device=self.device, half=self.half)
        for (x0, y0), result in zip(origins, results):
            if result.boxes is None:
                continue
            b = result.boxes.cpu().numpy()
            for xyxy, c, k in zip(b.xyxy, b.conf, b.cls.astype(int)):
                if self.names.get(k) == BALL:
                    x1, y1, x2, y2 = (float(v) / zoom for v in xyxy)
                    out.append(Detection(BALL, float(c), (x0 + x1, y0 + y1, x0 + x2, y0 + y2)))
        return out

