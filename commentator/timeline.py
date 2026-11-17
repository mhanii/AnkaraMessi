"""Per-frame world state as arrays, plus the two primitives every rule builds on:
short-gap filling for the ball and stationary-ball windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .core import FrameState

XY = Tuple[float, float]


@dataclass(frozen=True)
class Window:
    """A stretch during which the ball stayed within a small radius."""

    i0: int
    i1: int
    t0: float
    t1: float
    xy: XY

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


class Timeline:
    """Frames plus derived arrays. `ball` is (N, 2) in metres with NaN where unknown."""

    def __init__(self, frames: Sequence[FrameState], fps: float, max_gap_s: float = 0.5):
        self.frames: List[FrameState] = list(frames)
        self.fps = float(fps)
        n = len(self.frames)
        self.t = np.array([f.t for f in self.frames], dtype=float)
        raw = np.full((n, 2), np.nan)
        for i, f in enumerate(self.frames):
            if f.ball is not None:
                raw[i] = f.ball
        self.raw_ball = raw
        self.ball = _fill_gaps(raw, max(1, int(round(max_gap_s * self.fps))))
        self.known = ~np.isnan(self.ball[:, 0])

    def __len__(self) -> int:
        return len(self.frames)

    def frames_for(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self) - 1))

    def stationary_windows(self, radius_m: float, min_s: float,
                           max_gap_s: float = 3.0) -> List[Window]:
        """Maximal stretches where the ball stays within `radius_m` of the stretch's median.

        Unknown frames are tolerated for up to `max_gap_s` as long as the ball is found in the
        same place afterwards: a waiting ball is often hidden behind the taker, or simply too
        small for the detector in most frames. A ball that moved would not reappear on the spot.
        """
        n = len(self)
        gap_max = self.frames_for(max_gap_s)
        min_n = self.frames_for(min_s)
        outlier_n = self.frames_for(1.0)
        out: List[Window] = []
        k = 0
        while k < n:
            if not self.known[k]:
                k += 1
                continue
            pts = [self.ball[k]]
            j = k
            median = self.ball[k]
            while True:
                nxt = j + 1
                while nxt < n and not self.known[nxt] and nxt - j <= gap_max:
                    nxt += 1
                if nxt >= n or not self.known[nxt]:
                    break
                if float(np.hypot(*(self.ball[nxt] - median))) > radius_m:
                    # One stray detection elsewhere does not mean the ball left: if it is
                    # found back on the spot within a second, carry on from there.
                    back = next((m for m in range(nxt + 1, min(n, nxt + outlier_n + 1))
                                 if self.known[m]
                                 and float(np.hypot(*(self.ball[m] - median))) <= radius_m),
                                None)
                    if back is None:
                        break
                    nxt = back
                pts.append(self.ball[nxt])
                j = nxt
                if len(pts) % 5 == 0:
                    median = np.median(np.asarray(pts), axis=0)
            if j - k + 1 >= min_n:
                median = np.median(np.asarray(pts), axis=0)
                out.append(Window(k, j, float(self.t[k]), float(self.t[j]),
                                  (float(median[0]), float(median[1]))))
                k = j + 1
            else:
                k += 1
        return out


def _fill_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs no longer than `max_gap` frames between known samples."""
    out = arr.copy()
    for c in range(arr.shape[1]):
        col = out[:, c]
        valid = np.flatnonzero(~np.isnan(col))
        for a, b in zip(valid[:-1], valid[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                idx = np.arange(a + 1, b)
                col[idx] = np.interp(idx, [a, b], [col[a], col[b]])
    return out
