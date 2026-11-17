"""Passes, long passes, turnovers and goalkeeper distribution.

Built only on possession segments and player positions, so an airborne ball in between cannot
corrupt the result. Dribbles produce nothing: the same player keeps the ball.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..config import RuleConfig
from ..core import GOALKEEPER, Event
from ..pitch import dist, in_penalty_area, nearest_goal_side
from ..possession import Possession


def detect_passes(segments: Sequence[Possession], blocked: Sequence[Tuple[float, float]],
                  cfg: RuleConfig) -> List[Event]:
    """`blocked` are time spans (shots, set-piece waits) inside which no pass is reported."""
    events: List[Event] = []
    for n, (a, b) in enumerate(zip(segments[:-1], segments[1:])):
        gap = b.t0 - a.t1
        if gap < 0 or gap > cfg.pass_max_gap_s or a.track_id == b.track_id:
            continue
        if a.team is None or b.team is None:
            continue
        if any(s0 <= b.t0 and a.t1 <= s1 for s0, s1 in blocked):
            continue
        d = dist(a.last_xy, b.first_xy)

        if a.team != b.team:
            # Only call it once the new team has actually kept the ball for a moment.
            held = b.duration
            for nxt in segments[n + 2:]:
                if nxt.team != b.team or nxt.t0 - b.t1 > cfg.pass_max_gap_s:
                    break
                held = nxt.t1 - b.t0
            if held >= cfg.turnover_min_hold_s:
                events.append(Event("turnover", b.t0, b.t0, team=b.team, player=b.track_id,
                                    xy=b.first_xy, confidence=0.6,
                                    detail={"from_team": a.team}))
            continue

        if d < cfg.pass_min_m:
            continue
        speed = d / max(gap, 0.2)
        side = nearest_goal_side(a.last_xy[0])
        keeper_release = (a.role == GOALKEEPER and a.duration >= cfg.keeper_hold_min_s
                          and in_penalty_area(a.last_xy[0], a.last_xy[1], side, margin=1.0))
        lo, hi = cfg.hand_throw_range_m
        if keeper_release and lo <= d <= hi and speed <= cfg.hand_throw_max_speed_mps:
            kind, conf = "hand_throw", 0.5  # thrown vs passed short along the ground: a guess
        elif d >= cfg.long_pass_m:
            kind, conf = "long_pass", 0.8
        else:
            kind, conf = "pass", 0.8
        events.append(Event(kind, a.t1, b.t0, team=a.team, player=a.track_id,
                            receiver=b.track_id, xy=a.last_xy, distance_m=d, confidence=conf,
                            detail={"speed_mps": round(speed, 1),
                                    "from_keeper": a.role == GOALKEEPER,
                                    "to_xy": b.first_xy}))
    return events
