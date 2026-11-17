"""Per-frame pitch calibration: image pixels -> pitch metres.

Broadcast cameras pan and zoom constantly, so one homography per clip is useless. Each analysed
frame gets its own, solved from pitch landmarks found by a keypoint model.

Model choice was measured on SoccerNet GSR ground truth: the Soccana 29-keypoint model places
landmarks within 0.2-1.6 m; the 32-keypoint model we tried first scattered them by 2-40 m on
these camera angles. Frames with too few or badly spread landmarks are marked uncalibrated
rather than given a wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..pitch import LENGTH, SOCCANA_LANDMARKS, WIDTH, pitch_polylines

XY = Tuple[float, float]
_LANDMARKS = np.asarray(SOCCANA_LANDMARKS, dtype=np.float64)


@dataclass
class Homography:
    H: np.ndarray  # 3x3, image px -> pitch metres
    inliers: int
    landmarks: int

    def to_pitch(self, pts_px: Sequence[XY]) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts_m: Sequence[XY]) -> np.ndarray:
        pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def pitch_lines_px(self) -> List[np.ndarray]:
        """Pitch markings projected into the image, for drawing."""
        out = []
        for line in pitch_polylines():
            px = self.to_image(line)
            if np.all(np.isfinite(px)) and np.abs(px).max() < 1e5:
                out.append(px.round().astype(np.int32).reshape(-1, 1, 2))
        return out


def solve_homography(keypoints: np.ndarray, frame_size: Tuple[int, int],
                     min_conf: float = 0.5, ransac_m: float = 1.0) -> Optional[Homography]:
    """Fit image->pitch from predicted landmarks (N x 3: x, y, confidence).

    Returns None unless the fit is trustworthy: at least 4 agreeing landmarks that are not all
    on one line, and a result that puts the middle of the picture somewhere near the pitch.
    """
    w, h = frame_size
    ok = (keypoints[:, 2] > min_conf) & (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) \
        & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)
    if ok.sum() < 4:
        return None
    src = keypoints[ok, :2].astype(np.float64)
    dst = _LANDMARKS[ok]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_m)
    if H is None or mask is None:
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4:
        return None
    # Landmarks all on one line (e.g. only the halfway line) leave the fit unconstrained.
    centred = dst[inl] - dst[inl].mean(axis=0)
    minor = float(np.sqrt(max(np.linalg.eigvalsh(centred.T @ centred / inl.sum())[0], 0.0)))
    if minor < 2.0:
        return None
    hom = Homography(H, int(inl.sum()), int(ok.sum()))
    probe = hom.to_pitch([(w / 2.0, h * 0.75)])[0]
    if not np.all(np.isfinite(probe)) or not (-30 <= probe[0] <= LENGTH + 30
                                              and -30 <= probe[1] <= WIDTH + 30):
        return None
    return hom


class PitchCalibrator:
    def __init__(self, weights: Path, imgsz: int = 640, device: Optional[str] = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz, self.device = imgsz, device
        self.half = bool(device) and str(device).startswith("cuda")

    def calibrate(self, frame: np.ndarray) -> Optional[Homography]:
        result = self.model(frame, imgsz=self.imgsz, verbose=False, device=self.device,
                            half=self.half)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return None
        best = int(result.boxes.conf.argmax())
        keypoints = result.keypoints.data[best].cpu().numpy()
        if keypoints.shape[0] != len(_LANDMARKS):
            raise ValueError(f"expected {len(_LANDMARKS)} landmarks, model gave {keypoints.shape[0]}")
        return solve_homography(keypoints, (frame.shape[1], frame.shape[0]))
