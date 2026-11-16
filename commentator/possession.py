"""Possession segments: which player the ball is with, and for how long.

Possession is the most trustworthy signal in the system. A ball at a player's feet is on the
ground, so its projected position is correct, and the player's own position is always correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import DEFAULT, RuleConfig
from .core import XY, PlayerState
from .timeline import Timeline


@dataclass(frozen=True)
class Possession:
    track_id: int
    team: Optional[int]
    role: str
    i0: int
    i1: int
    t0: float
    t1: float
    first_xy: XY  # player position when possession starts
    last_xy: XY  # player position when possession ends

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def _nearest(players: List[PlayerState], xy: np.ndarray, radius: float) -> Optional[PlayerState]:
    best, best_d = None, radius
    for p in players:
        d = math.hypot(p.xy[0] - xy[0], p.xy[1] - xy[1])
        if d <= best_d:
            best, best_d = p, d
    return best


def possession_segments(tl: Timeline, cfg: RuleConfig = DEFAULT) -> List[Possession]:
    n = len(tl)
    cand: List[Optional[PlayerState]] = [None] * n
    for i, frame in enumerate(tl.frames):
        if tl.known[i] and frame.players:
            cand[i] = _nearest(frame.players, tl.ball[i], cfg.possession_radius_m)

    runs: List[List] = []  # [track_id or None, i0, i1]
    for i, p in enumerate(cand):
        pid = p.track_id if p is not None else None
        if runs and runs[-1][0] == pid:
            runs[-1][2] = i
        else:
            runs.append([pid, i, i])

    # Bridge short dropouts where the same player has the ball on both sides.
    gap_max = tl.frames_for(cfg.possession_merge_gap_s)
    for k in range(1, len(runs) - 1):
        r = runs[k]
        if (r[0] is None and r[2] - r[1] + 1 <= gap_max
                and runs[k - 1][0] is not None and runs[k - 1][0] == runs[k + 1][0]):
            r[0] = runs[k - 1][0]
    merged: List[List] = []
    for r in runs:
        if merged and r[0] is not None and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))

    min_n = tl.frames_for(cfg.possession_min_s)
    segments: List[Possession] = []
    for pid, i0, i1 in merged:
        if pid is None or (i1 - i0 + 1) < min_n:
            continue
        first = next(cand[i] for i in range(i0, i1 + 1)
                     if cand[i] is not None and cand[i].track_id == pid)
        last = next(cand[i] for i in range(i1, i0 - 1, -1)
                    if cand[i] is not None and cand[i].track_id == pid)
        segments.append(Possession(pid, first.team, first.role, i0, i1,
                                   float(tl.t[i0]), float(tl.t[i1]), first.xy, last.xy))
    return segments
