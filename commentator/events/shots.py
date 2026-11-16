"""Shots, saves and goals.

What a shot looks like in real, projected data: the ball leaves an attacker (or a dead-ball
spot) within range of goal, is past that goal line within a second or two, and is then seen
near the goal mouth. Speed and the first crossing point are not trustworthy, because a rising
ball projects far off its true path.

A goal is detected independently of who shot: the ball comes to rest in the net zone (behind
the line, between the posts) shortly after being in play. That works for headers and
scrambles where no clean "release" was ever seen.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..config import RuleConfig
from ..core import GOALKEEPER, XY, Event
from ..pitch import GOAL_CENTRES, MID_Y, dist, goal_line_depth, inside, nearest_goal_side
from ..possession import Possession
from ..timeline import Timeline

# (release frame index, position, team, player id, source)
Release = Tuple[int, XY, Optional[int], Optional[int], str]


def detect_shots(tl: Timeline, segments: Sequence[Possession], set_pieces: Sequence[Event],
                 cfg: RuleConfig) -> List[Event]:
    releases: List[Release] = [(s.i1, s.last_xy, s.team, s.track_id, "open_play")
                               for s in segments]
    for sp in set_pieces:
        if sp.type in ("penalty", "free_kick") and sp.xy is not None and sp.detail.get("taken"):
            releases.append((tl.index_at(sp.t), sp.xy, sp.team, sp.player, sp.type))
    releases.sort(key=lambda r: r[0])

    defenders = {side: defending_team(tl, side) for side in ("left", "right")}
    events: List[Event] = []
    busy_until = -1
    for i_rel, pos, team, player, source in releases:
        if i_rel <= busy_until:
            continue
        side = nearest_goal_side(pos[0])
        if dist(pos, GOAL_CENTRES[side]) > cfg.shot_max_range_m:
            continue
        if team is not None and team == defenders[side]:
            # A defender or keeper clearing their own lines is not a shot. But in a crowded box
            # the nearest player is often a defender while an attacker actually struck it.
            rival = _nearest_opponent(tl, i_rel, pos, team, cfg.shot_contest_radius_m)
            if rival is None:
                continue
            team, player = rival.team, rival.track_id
        hit = _trace(tl, segments, i_rel, pos, side, team, cfg)
        if hit is None:
            continue
        kind, i_end, extra = hit
        detail = {"source": source, "goal_side": side,
                  "range_m": round(dist(pos, GOAL_CENTRES[side]), 1)}
        t_rel = float(tl.t[i_rel])
        if kind == "save":
            events.append(Event("shot", t_rel, float(tl.t[i_end]), team=team, player=player,
                                xy=pos, confidence=0.6, detail={**detail, "outcome": "saved"}))
            events.append(Event("save", float(tl.t[i_end]), float(tl.t[i_end]),
                                team=extra["keeper_team"], player=extra["keeper"],
                                xy=pos, confidence=0.6, detail={"goal_side": side}))
        else:
            events.append(Event("shot", t_rel, float(tl.t[i_end]), team=team, player=player,
                                xy=pos, confidence=0.7, detail={**detail, "outcome": "unknown"}))
        busy_until = i_end

    return _apply_goals(tl, events, cfg)


def _apply_goals(tl: Timeline, events: List[Event], cfg: RuleConfig) -> List[Event]:
    """Turn the shot before each ball-in-net moment into a goal, or add an unattributed goal."""
    last_goal = {"left": -1e9, "right": -1e9}
    for i_enter, side in _net_dwells(tl, cfg):
        t_enter = float(tl.t[i_enter])
        if t_enter - last_goal[side] < cfg.goal_repeat_s:
            continue
        last_goal[side] = t_enter
        shot = next((e for e in reversed(events)
                     if e.type == "shot" and e.detail.get("goal_side") == side
                     and 0.0 <= t_enter - e.t <= cfg.goal_link_s), None)
        if shot is not None:
            shot.type = "goal"
            shot.end_t = t_enter
            shot.confidence = 0.7
            shot.detail["outcome"] = "goal"
            # A "save" reported for the same shot was the ball passing the keeper on its way in.
            events[:] = [e for e in events
                         if not (e.type == "save" and shot.t <= e.t <= t_enter + 0.5)]
        else:
            events.append(Event("goal", max(0.0, t_enter - 0.5), t_enter, team=None,
                                xy=GOAL_CENTRES[side], confidence=0.5,
                                detail={"goal_side": side, "source": "ball_in_net",
                                        "outcome": "goal"}))
    return events


def _net_dwells(tl: Timeline, cfg: RuleConfig) -> List[Tuple[int, str]]:
    """Frames where the ball starts a stay in a net zone, having just been in play."""
    lo, hi = cfg.goal_net_depth_m
    need = tl.frames_for(cfg.goal_net_min_s)
    gap_max = tl.frames_for(0.4)
    recent = tl.frames_for(cfg.goal_in_play_within_s)
    out: List[Tuple[int, str]] = []

    def close(start: Optional[int], count: int, side: str) -> None:
        if start is None or count < need:
            return
        before = range(max(0, start - recent), start)
        if any(tl.known[k] and inside(tl.ball[k, 0], tl.ball[k, 1], -0.5) for k in before):
            out.append((start, side))

    for side in ("left", "right"):
        start: Optional[int] = None
        last_in = -1
        count = 0
        for k in range(len(tl)):
            if not tl.known[k]:
                continue
            x, y = tl.ball[k]
            in_zone = lo <= goal_line_depth(x, side) <= hi and \
                abs(y - MID_Y) <= cfg.goal_net_halfwidth_m
            if in_zone:
                if start is None or k - last_in > gap_max:
                    close(start, count, side)
                    start, count = k, 0
                count += 1
                last_in = k
            elif start is not None and k - last_in > gap_max:
                close(start, count, side)
                start, count = None, 0
        close(start, count, side)
    out.sort()
    return out


def _trace(tl: Timeline, segments: Sequence[Possession], i_rel: int, pos: XY, side: str,
           team: Optional[int], cfg: RuleConfig):
    """Follow the ball after release. Returns (kind, end_index, extra) or None.

    The first crossing point is ignored on purpose: a rising ball projects far wide of its true
    path and only comes back to a believable position once it drops.
    """
    end = min(len(tl) - 1, i_rel + tl.frames_for(cfg.shot_lookahead_s))
    t_rel = tl.t[i_rel]
    next_seg = next((s for s in segments if s.i0 > i_rel + tl.frames_for(0.2)), None)
    first_cross: Optional[int] = None
    for k in range(i_rel + 1, end + 1):
        if next_seg is not None and k >= next_seg.i0 and first_cross is None:
            if (next_seg.role == GOALKEEPER and next_seg.team is not None
                    and team is not None and next_seg.team != team
                    and dist(pos, next_seg.first_xy) >= 6.0):
                elapsed = max(1e-3, float(tl.t[k] - t_rel))
                if dist(pos, next_seg.first_xy) / elapsed >= cfg.shot_min_speed_mps:
                    return "save", k, {"keeper_team": next_seg.team,
                                       "keeper": next_seg.track_id}
            return None  # somebody else received it: a pass, cross or block
        if not tl.known[k]:
            continue
        x, y = tl.ball[k]
        if goal_line_depth(x, side) >= -0.5:
            continue
        if first_cross is None:
            elapsed = max(1e-3, float(tl.t[k] - t_rel))
            if elapsed > cfg.shot_cross_within_s or \
                    dist(pos, (x, y)) / elapsed < cfg.shot_min_speed_mps:
                return None  # rolled out slowly: not a shot
            first_cross = k
        if abs(y - MID_Y) <= cfg.shot_cross_halfwidth_m:
            return "crossed", k, {}
    return None


def _nearest_opponent(tl: Timeline, i: int, pos: XY, team: int, radius: float):
    best, best_d = None, radius
    for p in tl.frames[i].players:
        if p.team is not None and p.team != team:
            d = dist(p.xy, pos)
            if d <= best_d:
                best, best_d = p, d
    return best


def defending_team(tl: Timeline, side: str) -> Optional[int]:
    """Team of the goalkeeper guarding `side`, by majority over the clip. None if unseen."""
    votes = {0: 0, 1: 0}
    for frame in tl.frames[::max(1, tl.frames_for(0.5))]:
        for p in frame.players:
            if p.role == GOALKEEPER and p.team in votes and goal_line_depth(p.xy[0], side) <= 25:
                votes[p.team] += 1
    if votes[0] == votes[1]:
        return None
    return 0 if votes[0] > votes[1] else 1
