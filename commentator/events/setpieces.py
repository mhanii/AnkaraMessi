"""Set pieces: corner, penalty, kick-off, goal kick, free kick.

All of them share one signature that survives real footage: the ball lies still for a while at
a telling place, then leaves. A ball lying still is on the ground, so its projected position is
trustworthy. The event time is the moment the ball leaves (the restart is taken).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..config import RuleConfig
from ..core import GOALKEEPER, Event, PlayerState
from ..pitch import (CENTRE, GOAL_CENTRES, LENGTH, MID_Y, PENALTY_SPOTS, corner_name, corner_nearest, dist,
                     goal_line_depth, in_penalty_area, inside, nearest_goal_side)
from ..timeline import Timeline, Window


def merged_windows(tl: Timeline, cfg: RuleConfig) -> List[Window]:
    """Stationary windows, with fragments of the same wait glued back together."""
    out: List[Window] = []
    for w in tl.stationary_windows(cfg.still_radius_m, cfg.still_min_s):
        if (out and w.t0 - out[-1].t1 <= cfg.still_merge_gap_s
                and dist(w.xy, out[-1].xy) <= cfg.still_merge_dist_m):
            prev = out[-1]
            total = prev.duration + w.duration
            xy = ((prev.xy[0] * prev.duration + w.xy[0] * w.duration) / total,
                  (prev.xy[1] * prev.duration + w.xy[1] * w.duration) / total)
            out[-1] = Window(prev.i0, w.i1, prev.t0, w.t1, xy)
        else:
            out.append(w)
    return out


def detect_set_pieces(tl: Timeline, cfg: RuleConfig) -> List[Event]:
    events: List[Event] = []
    for w in merged_windows(tl, cfg):
        kind = _classify(tl, w, cfg)
        if kind is None:
            continue
        taken, confidence, t_event = _taken(tl, w, cfg)
        taker = _nearest_player(tl, w)
        detail = {"taken": taken, "setup_t": round(w.t0, 2),
                  "waited_s": round(t_event - w.t0, 1)}
        if kind == "corner_kick":
            detail["corner"] = corner_name(corner_nearest(w.xy)[0])
            if taker is not None and taker.team is not None:
                side = nearest_goal_side(w.xy[0])
                detail["attackers_in_box"] = sum(
                    1 for p in _players_at(tl, w.i1)
                    if p.team == taker.team and in_penalty_area(p.xy[0], p.xy[1], side))
        if kind == "free_kick":
            detail["range_m"] = round(dist(w.xy, GOAL_CENTRES[nearest_goal_side(w.xy[0])]), 1)
        if kind in ("goal_kick", "penalty"):
            detail["side"] = nearest_goal_side(w.xy[0])
        event = Event(kind, t_event, t_event, team=taker.team if taker else None,
                      player=taker.track_id if taker else None, xy=w.xy,
                      confidence=confidence, detail=detail)
        # The same dead ball being nudged or re-placed is one set piece: keep the last wait.
        if (events and events[-1].type == kind
                and w.t0 - events[-1].t <= cfg.repeat_gap_s
                and dist(w.xy, events[-1].xy) <= cfg.repeat_dist_m):
            event.detail["setup_t"] = events[-1].detail["setup_t"]
            events[-1] = event
        else:
            events.append(event)
    return events


def detect_kickoffs_by_formation(tl: Timeline, known: Sequence[Event],
                                 cfg: RuleConfig) -> List[Event]:
    """Kick-offs recognised from the formation alone, for when the ball is not in view.

    Both teams entirely in their own halves with someone on the centre spot is unmistakable.
    The kick-off is taken when that formation breaks.
    """
    step = max(1, tl.frames_for(0.2))
    need = max(1, int(round(cfg.kickoff_formation_min_s / 0.2)))
    events: List[Event] = []
    run_start: Optional[int] = None
    last_ok = -1
    count = 0

    def close(end_i: int) -> None:
        if run_start is None or count < need:
            return
        t_end = float(tl.t[end_i])
        if any(e.type == "kick_off" and tl.t[run_start] - 3.0 <= e.t <= t_end + 3.0
               for e in known):
            return
        taken = end_i < len(tl) - 1 - tl.frames_for(0.5)
        events.append(Event("kick_off", t_end, t_end, xy=CENTRE, confidence=0.5,
                            detail={"taken": taken, "setup_t": round(float(tl.t[run_start]), 2),
                                    "by": "formation"}))

    for k in range(0, len(tl), step):
        players = tl.frames[k].players
        ok = (bool(players) and _teams_in_own_halves(players, cfg, strict=True)
              and any(dist(p.xy, CENTRE) <= 3.0 for p in players))
        if ok:
            if run_start is None:
                run_start, count = k, 0
            count += 1
            last_ok = k
        elif run_start is not None and k - last_ok > tl.frames_for(0.6):
            close(last_ok)
            run_start, count = None, 0
    close(last_ok)
    return events


def _classify(tl: Timeline, w: Window, cfg: RuleConfig) -> Optional[str]:
    x, y = w.xy
    side = nearest_goal_side(x)
    depth = goal_line_depth(x, side)
    players = _players_at(tl, w.i1)

    if corner_nearest(w.xy)[1] <= cfg.corner_radius_m and w.duration >= cfg.corner_min_s:
        return "corner_kick"

    if min(dist(w.xy, s) for s in PENALTY_SPOTS) <= cfg.penalty_radius_m \
            and w.duration >= cfg.penalty_min_s:
        in_area = sum(in_penalty_area(p.xy[0], p.xy[1], side, -cfg.penalty_area_inset_m)
                      for p in players)
        on_line = any(goal_line_depth(p.xy[0], side) <= cfg.penalty_keeper_depth_m
                      and abs(p.xy[1] - MID_Y) <= 5.0 for p in players)
        taker = min(players, key=lambda p: dist(p.xy, w.xy)) if players else None
        # Everyone but the taker and the keeper is out of the area, the keeper is on the line,
        # and the nearest player is not a goalkeeper standing over a dead ball.
        if in_area <= cfg.penalty_max_in_area and on_line \
                and taker is not None and taker.role != GOALKEEPER:
            return "penalty"

    if dist(w.xy, CENTRE) <= cfg.kickoff_radius_m and w.duration >= cfg.kickoff_min_s \
            and _teams_in_own_halves(players, cfg):
        return "kick_off"

    lo, hi = cfg.goal_kick_depth_m
    if lo <= depth <= hi and abs(y - MID_Y) <= cfg.goal_kick_halfwidth_m \
            and w.duration >= cfg.goal_kick_min_s:
        # For a goal kick the keeper places the ball and steps back. A keeper standing on the
        # ball the whole time is holding it after a save or a back-pass.
        return None if _keeper_holding(tl, w, cfg) else "goal_kick"

    if w.duration >= cfg.free_kick_min_s and inside(x, y) and not _keeper_holding(tl, w, cfg):
        return "free_kick"
    return None


def _taken(tl: Timeline, w: Window, cfg: RuleConfig):
    """Did the ball leave the spot after the wait? Returns (taken, confidence, time).

    With a perfect ball track the wait ends at the kick. A real detector often loses a waiting
    ball seconds early and finds it again only once it is flying, so when there is a blind gap
    the kick is timed from the first sighting away from the spot instead.
    """
    if w.i1 >= len(tl) - 1 - tl.frames_for(0.3):
        return False, 0.6, w.t1  # the clip ends while the ball is still waiting
    end = min(len(tl) - 1, w.i1 + tl.frames_for(cfg.taken_window_s))
    for k in range(w.i1 + 1, end + 1):
        if tl.known[k] and dist(tl.ball[k], w.xy) >= cfg.taken_move_m:
            blind = float(tl.t[k]) - w.t1
            if blind <= cfg.taken_blind_s:
                return True, 0.9, w.t1
            return True, 0.75, max(w.t1, float(tl.t[k]) - cfg.taken_refound_lag_s)
    if len(tl) - 1 - w.i1 < tl.frames_for(cfg.taken_window_s):
        return False, 0.6, w.t1  # not enough footage after the wait to say; still pending
    seen_after = any(tl.known[k] for k in range(w.i1 + 1, end + 1))
    # Ball vanished (camera cut, or struck out of view): most likely taken, less certain.
    return (not seen_after), 0.6, w.t1


def _players_at(tl: Timeline, i: int, lookback_s: float = 1.0) -> List[PlayerState]:
    """Players in the most recent calibrated frame at or before i."""
    for k in range(i, max(-1, i - tl.frames_for(lookback_s)), -1):
        if tl.frames[k].players:
            return tl.frames[k].players
    return []


def _nearest_player(tl: Timeline, w: Window) -> Optional[PlayerState]:
    players = _players_at(tl, w.i1)
    if not players:
        return None
    return min(players, key=lambda p: dist(p.xy, w.xy))


def _teams_in_own_halves(players: Sequence[PlayerState], cfg: RuleConfig,
                         strict: bool = False) -> bool:
    """Kick-off formation: each team entirely in one half, on opposite sides.

    `strict` is for use without a ball in view: it needs enough players of both teams visible
    and never falls back to a guess.
    """
    slack = cfg.kickoff_halfway_slack_m
    by_team = {0: [], 1: []}
    for p in players:
        if p.team in by_team:
            by_team[p.team].append(p.xy[0])
    minimum = cfg.kickoff_min_players if strict else 4
    if len(by_team[0]) < minimum or len(by_team[1]) < minimum:
        if strict:
            return False
        # Teams unknown: fall back to "nobody but the takers inside the centre circle".
        near = sum(dist(p.xy, CENTRE) <= 9.15 for p in players)
        return len(players) >= 8 and near <= 4
    half = LENGTH / 2.0
    need = cfg.kickoff_own_half_fraction

    def frac_left(xs):
        return float(np.mean([x <= half + slack for x in xs]))

    def frac_right(xs):
        return float(np.mean([x >= half - slack for x in xs]))

    return (frac_left(by_team[0]) >= need and frac_right(by_team[1]) >= need) or \
           (frac_right(by_team[0]) >= need and frac_left(by_team[1]) >= need)


def _keeper_holding(tl: Timeline, w: Window, cfg: RuleConfig) -> bool:
    """A goalkeeper stood on the ball for most of the window (holding it, not a free kick)."""
    close = total = 0
    for k in range(w.i0, w.i1 + 1, max(1, tl.frames_for(0.2))):
        if not tl.known[k]:
            continue
        total += 1
        close += any(p.role == GOALKEEPER and dist(p.xy, tl.ball[k]) <= cfg.keeper_hold_radius_m
                     for p in tl.frames[k].players)
    return total > 0 and close / total >= 0.5
