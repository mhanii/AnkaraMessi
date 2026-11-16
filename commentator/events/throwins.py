"""Throw-ins.

A throw-in is the one time an outfield player has the ball in their hands. In the picture the
ball sits inside the upper part of the player's outline (held at the waist, then raised
overhead) while the player stands still on a touchline. Perception records that as
`FrameState.ball_holder`; this rule adds the where and the how long.

Checked by eye on one real example (SoccerNet GSR SNGS-144, 23-25 s). The dataset has no
throw-in labels, so there is no score for this rule yet.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..config import RuleConfig
from ..core import PLAYER, XY, Event, PlayerState
from ..pitch import WIDTH, dist, touchline_name
from ..possession import Possession
from ..timeline import Timeline


def _on_touchline(y: float, cfg: RuleConfig) -> bool:
    return (-cfg.throw_outside_m <= y <= cfg.throw_band_m
            or WIDTH - cfg.throw_band_m <= y <= WIDTH + cfg.throw_outside_m)


def _holder(tl: Timeline, k: int, cfg: RuleConfig) -> Optional[PlayerState]:
    frame = tl.frames[k]
    if frame.ball_holder is None:
        return None
    for p in frame.players:
        if p.track_id == frame.ball_holder and p.role == PLAYER and _on_touchline(p.xy[1], cfg):
            # Being above the ground, a held ball projects further from the camera than the
            # holder's feet. A ball at the feet of a player on the line does not.
            if tl.known[k] and p.xy[1] - tl.ball[k, 1] < cfg.throw_min_lift_m:
                return None
            return p
    return None


def detect_throw_ins(tl: Timeline, segments: Sequence[Possession],
                     cfg: RuleConfig) -> List[Event]:
    # Runs of frames where the same touchline player holds the ball, bridging short dropouts.
    runs: List[List] = []  # [track_id, i0, i1, first PlayerState, last PlayerState]
    gap = tl.frames_for(cfg.throw_merge_gap_s)
    for k in range(len(tl)):
        p = _holder(tl, k, cfg)
        if p is None:
            continue
        if runs and runs[-1][0] == p.track_id and k - runs[-1][2] <= gap:
            runs[-1][2], runs[-1][4] = k, p
        else:
            runs.append([p.track_id, k, k, p, p])

    events: List[Event] = []
    for tid, i0, i1, first, last in runs:
        held = float(tl.t[i1] - tl.t[i0])
        if held < cfg.throw_min_hold_s or dist(first.xy, last.xy) > cfg.throw_max_drift_m:
            continue
        t_throw = float(tl.t[i1])
        taken = i1 < len(tl) - 1 - tl.frames_for(0.3)  # else still holding when the clip ends
        receiver, reach = _receiver(segments, tid, t_throw, last.xy, cfg) if taken else (None, None)
        kind = "long_throw" if reach is not None and reach >= cfg.long_throw_m else "throw_in"
        events.append(Event(kind, t_throw, t_throw, team=last.team, player=tid,
                            receiver=receiver, xy=last.xy, distance_m=reach, confidence=0.6,
                            detail={"touchline": touchline_name(last.xy[1]), "taken": taken,
                                    "setup_t": round(float(tl.t[i0]), 2),
                                    "held_s": round(held, 1)}))
    return events


def _receiver(segments: Sequence[Possession], thrower: int, t_throw: float, origin: XY,
              cfg: RuleConfig) -> Tuple[Optional[int], Optional[float]]:
    """Who the throw reached and how far away they were. Uses player positions only."""
    for seg in segments:
        if seg.track_id != thrower and 0.0 < seg.t0 - t_throw <= cfg.throw_receive_s:
            return seg.track_id, dist(origin, seg.first_xy)
    return None, None
