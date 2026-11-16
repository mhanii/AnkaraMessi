"""Phases of play that need no ball at all: sustained pressure and replays.

Player positions are the most reliable thing the system knows, so these give the commentary
something true to say during the long stretches when the ball detector sees nothing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..config import RuleConfig
from ..core import GOALKEEPER, PLAYER, Event
from ..pitch import goal_line_depth
from ..timeline import Timeline


def attacked_sides(tl: Timeline) -> Dict[int, str]:
    """Which goal each team attacks: {team: 'left' | 'right'}. Empty if it cannot be told.

    A goalkeeper standing in a goal settles it. Otherwise the team that sits deeper on average
    is the one defending that end.
    """
    step = max(1, tl.frames_for(0.5))
    keeper_votes = {"left": {0: 0, 1: 0}, "right": {0: 0, 1: 0}}
    depth_sum = {0: 0.0, 1: 0.0}
    count = {0: 0, 1: 0}
    for frame in tl.frames[::step]:
        for p in frame.players:
            if p.team not in (0, 1):
                continue
            if p.role == GOALKEEPER:
                for side in ("left", "right"):
                    if goal_line_depth(p.xy[0], side) <= 25:
                        keeper_votes[side][p.team] += 1
            else:
                depth_sum[p.team] += p.xy[0]
                count[p.team] += 1
    for side, other in (("left", "right"), ("right", "left")):
        votes = keeper_votes[side]
        if votes[0] != votes[1] and max(votes.values()) >= 3:
            defender = 0 if votes[0] > votes[1] else 1
            return {defender: other, 1 - defender: side}
    if count[0] < 20 or count[1] < 20:
        return {}
    mean = {t: depth_sum[t] / count[t] for t in (0, 1)}
    if abs(mean[0] - mean[1]) < 3.0:
        return {}
    left_defender = 0 if mean[0] < mean[1] else 1
    return {left_defender: "right", 1 - left_defender: "left"}


def detect_phases(tl: Timeline, cfg: RuleConfig) -> List[Event]:
    events: List[Event] = []
    events += _pressure(tl, cfg)
    events += _replays(tl, cfg)
    return events


def _pressure(tl: Timeline, cfg: RuleConfig) -> List[Event]:
    sides = attacked_sides(tl)
    if not sides:
        return []
    step = max(1, tl.frames_for(0.4))
    need = max(1, int(round(cfg.pressure_min_s / 0.4)))
    events: List[Event] = []
    streak = {0: 0, 1: 0}
    last_said = {0: -1e9, 1: -1e9}
    for k in range(0, len(tl), step):
        frame = tl.frames[k]
        for team, side in sides.items():
            forward = sum(1 for p in frame.players
                          if p.team == team and p.role == PLAYER
                          and goal_line_depth(p.xy[0], side) <= cfg.pressure_zone_m)
            streak[team] = streak[team] + 1 if forward >= cfg.pressure_min_players else 0
            t = float(tl.t[k])
            if streak[team] == need and t - last_said[team] >= cfg.pressure_repeat_s:
                last_said[team] = t
                events.append(Event("pressure", t, t, team=team, confidence=0.8,
                                    detail={"players_forward": forward, "goal_side": side}))
    return events


def _replays(tl: Timeline, cfg: RuleConfig) -> List[Event]:
    """Stretches where the pitch cannot be located: close-ups and replays."""
    events: List[Event] = []
    start: Optional[int] = None
    for k in range(len(tl) + 1):
        lost = k < len(tl) and not tl.frames[k].calibrated
        if lost and start is None:
            start = k
        elif not lost and start is not None:
            if start > 0 and (k - start) / tl.fps >= cfg.replay_min_s:
                events.append(Event("replay", float(tl.t[start]), float(tl.t[k - 1]),
                                    confidence=0.5))
            start = None
    return events
