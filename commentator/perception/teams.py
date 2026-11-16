"""Team assignment from shirt colour, and a plain-English name for each team's colour."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_NAMED = {
    "white": (240, 240, 240), "black": (25, 25, 25), "grey": (128, 128, 128),
    "red": (200, 30, 40), "maroon": (110, 20, 40), "orange": (240, 130, 30),
    "yellow": (235, 215, 40), "green": (40, 150, 70), "sky blue": (120, 190, 240),
    "blue": (30, 70, 200), "navy": (20, 30, 90), "purple": (120, 50, 150), "pink": (240, 130, 180),
}


def colour_name(bgr: np.ndarray) -> str:
    """A plain word for a shirt colour. Washed-out colours are judged on brightness alone,
    so a white shirt in shadow is still 'white' rather than 'grey'."""
    pixel = np.clip(np.asarray(bgr, dtype=float), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    _, sat, val = (int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])
    if sat < 60:
        return "white" if val >= 135 else "black" if val < 70 else "grey"
    rgb = np.asarray(bgr, dtype=float)[::-1]
    coloured = {n: c for n, c in _NAMED.items() if n not in ("white", "black", "grey")}
    return min(coloured, key=lambda n: float(np.sum((rgb - np.array(coloured[n])) ** 2)))


def shirt_colour(frame: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """Median BGR of the chest area: the upper-middle part of a person box, where grass,
    shorts and skin intrude least."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 16:
        return None
    xa, xb = int(x1 + 0.30 * w), int(x1 + 0.70 * w)
    ya, yb = int(y1 + 0.18 * h), int(y1 + 0.45 * h)
    crop = frame[max(ya, 0):max(yb, 0), max(xa, 0):max(xb, 0)]
    if crop.size == 0:
        return None
    return np.median(crop.reshape(-1, 3), axis=0)


class TeamClassifier:
    """Collects shirt colours per track while frames go by, then splits tracks into two teams."""

    def __init__(self, max_samples: int = 40):
        self.samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.max_samples = max_samples

    def observe(self, frame: np.ndarray, track_id: int,
                box: Tuple[float, float, float, float]) -> None:
        if len(self.samples[track_id]) >= self.max_samples:
            return
        colour = shirt_colour(frame, box)
        if colour is not None:
            self.samples[track_id].append(colour)

    def finalize(self, tracks: List[int]) -> "TeamSplit":
        """Split tracks into two teams by shirt colour.

        Clusters in Lab, where distance tracks how different two colours look. Also reports
        how far each track's colour is from its team, which is how goalkeepers are spotted:
        their kit must differ from both teams.
        """
        usable = [t for t in tracks if len(self.samples.get(t, [])) >= 3]
        if len(usable) < 4:
            return TeamSplit({}, {}, {}, 0.0)
        bgr = np.array([np.median(np.array(self.samples[t]), axis=0) for t in usable])
        lab = cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        data = lab.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        cv2.setRNGSeed(7)
        _, labels, centres = cv2.kmeans(data, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        distance = np.linalg.norm(data - centres[labels], axis=1)
        typical = float(np.median(distance))
        team_of = {t: int(lbl) for t, lbl in zip(usable, labels)}
        off_colour = {t: float(d) for t, d in zip(usable, distance)}
        # Team colour = median of its well-matching tracks, so a keeper cannot tint it.
        colours = {}
        for k in (0, 1):
            core = (labels == k) & (distance <= max(typical * 1.5, 1e-6))
            if np.any(core):
                colours[k] = np.median(bgr[core], axis=0)
        return TeamSplit(team_of, colours, off_colour, typical)


@dataclass
class TeamSplit:
    team_of: Dict[int, int]
    colours: Dict[int, np.ndarray]  # BGR per team
    off_colour: Dict[int, float]  # Lab distance of each track from its team's colour
    typical: float  # the median of those distances
