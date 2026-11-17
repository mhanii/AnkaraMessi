"""Running play-by-play: the small things a commentator fills the gaps with.

Who is on the ball, a run with it, play moving into the final third or the box, a switch of
flank. These are deliberately chatty and are NOT scored against ground truth: they exist so
the commentary never goes quiet. Expect some to be wrong.

When the ball is not detected the "play point" falls back to the middle of the players in
view. A broadcast camera follows the ball, so that is where the action is.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RuleConfig
from ..core import Event
from ..pitch import LENGTH, WIDTH, dist, goal_line_depth, in_penalty_area, inside
from ..possession import Possession
from ..timeline import Timeline
from .phases import attacked_sides

XY = Tuple[float, float]


def play_points(tl: Timeline) -> List[Optional[XY]]:
    """Where the action is, frame by frame: the ball if seen on the pitch, else the middle of
    the players in view."""
    out: List[Optional[XY]] = []
    for k, frame in enumerate(tl.frames):
        if tl.known[k] and inside(tl.ball[k, 0], tl.ball[k, 1], 2.0):
            out.append((float(tl.ball[k, 0]), float(tl.ball[k, 1])))
        elif len(frame.players) >= 4:
            xs = np.array([p.xy for p in frame.players])
            out.append((float(np.median(xs[:, 0])), float(np.median(xs[:, 1]))))
        else:
            out.append(None)
    return out


def _third(x: float) -> str:
    return "left" if x < LENGTH / 3 else "right" if x > 2 * LENGTH / 3 else "mid"


def _flank(y: float) -> str:
    return "far" if y < 18.0 else "near" if y > WIDTH - 18.0 else "centre"


def _team_attacking(sides: Dict[int, str], side: str) -> Optional[int]:
    return next((t for t, s in sides.items() if s == side), None)


def where_words(xy: XY, team: Optional[int], sides: Dict[int, str]) -> str:
    flank = _flank(xy[1])
    wide = {"far": " on the far side", "near": " on the near side", "centre": ""}[flank]
    if team in sides:
        depth = goal_line_depth(xy[0], sides[team])  # distance from the goal they attack
        if depth < LENGTH / 3:
            return "in the final third" + wide
        if depth > 2 * LENGTH / 3:
            return "at the back" if not wide else "deep" + wide
    return "out wide" + wide if wide else "in midfield"


def detect_play(tl: Timeline, segments: Sequence[Possession], passes: Sequence[Event],
                cfg: RuleConfig) -> List[Event]:
    sides = attacked_sides(tl)
    events: List[Event] = []
    events += _zones(tl, sides, cfg)
    events += _on_the_ball(segments, sides, cfg)
    events += _shape(tl, sides, cfg)
    events += _breaks(tl, sides, cfg)
    events += _situation(tl, sides, cfg)
    _describe_passes(passes, sides)
    return events


def _situation(tl: Timeline, sides: Dict[int, str], cfg: RuleConfig) -> List[Event]:
    """A regular plain description of where the game is being played, so there is always
    something true to say. Also notes when one team has more players around the action."""
    points = play_points(tl)
    events: List[Event] = []
    last = -1e9
    last_numbers = -1e9
    for k, pt in enumerate(points):
        t = float(tl.t[k])
        if pt is None or t - last < cfg.situation_every_s:
            continue
        last = t
        third = _third(pt[0])
        defender = None
        if third != "mid":  # the team defending that end is the one attacking the other
            defender = _team_attacking(sides, "left" if third == "right" else "right")
        wide = {"far": " on the far side", "near": " on the near side", "centre": ""}[_flank(pt[1])]
        events.append(Event("situation", t, t, team=defender, xy=pt, confidence=0.5,
                            detail={"zone": "third" if third != "mid" else "middle",
                                    "wide": wide, "side": third}))
        around = {0: 0, 1: 0}
        for p in tl.frames[k].players:
            if p.team in around and dist(p.xy, pt) <= cfg.numbers_radius_m:
                around[p.team] += 1
        if abs(around[0] - around[1]) >= 2 and t - last_numbers >= cfg.numbers_repeat_s:
            last_numbers = t
            team = 0 if around[0] > around[1] else 1
            events.append(Event("numbers", t + 0.1, t + 0.1, team=team, xy=pt, confidence=0.5,
                                detail={"count": around[team]}))
    return events


def _shape(tl: Timeline, sides: Dict[int, str], cfg: RuleConfig) -> List[Event]:
    """Team shape, from player positions alone: everybody back, or a high defensive line."""
    events: List[Event] = []
    if not sides:
        return events
    step = max(1, tl.frames_for(0.4))
    need = max(1, int(round(cfg.shape_min_s / 0.4)))
    streak = {("deep", t): 0 for t in sides}
    streak.update({("high", t): 0 for t in sides})
    last = {key: -1e9 for key in streak}
    for k in range(0, len(tl), step):
        frame = tl.frames[k]
        t = float(tl.t[k])
        for team, attacks in sides.items():
            own = "left" if attacks == "right" else "right"
            depths = [goal_line_depth(p.xy[0], own) for p in frame.players
                      if p.team == team and p.role == "player"]
            if len(depths) < 6:
                streak[("deep", team)] = streak[("high", team)] = 0
                continue
            back = sum(d <= cfg.shape_deep_zone_m for d in depths)
            for kind, ok, detail in (("deep", back >= cfg.shape_deep_players, {"count": back}),
                                     ("high", min(depths) >= cfg.shape_high_line_m, {})):
                key = (kind, team)
                streak[key] = streak[key] + 1 if ok else 0
                if streak[key] == need and t - last[key] >= cfg.shape_repeat_s:
                    last[key] = t
                    events.append(Event("shape_deep" if kind == "deep" else "high_line", t, t,
                                        team=team, confidence=0.5, detail=dict(detail)))
    return events


def _breaks(tl: Timeline, sides: Dict[int, str], cfg: RuleConfig) -> List[Event]:
    """The action racing up the pitch: a break."""
    points = play_points(tl)
    span = tl.frames_for(cfg.break_window_s)
    events: List[Event] = []
    last = -1e9
    for k in range(span, len(points)):
        a, b = points[k - span], points[k]
        if a is None or b is None:
            continue
        t = float(tl.t[k])
        gained = b[0] - a[0]
        if abs(gained) >= cfg.break_min_m and t - last >= cfg.break_repeat_s:
            last = t
            side = "right" if gained > 0 else "left"
            events.append(Event("break", t, t, team=_team_attacking(sides, side), xy=b,
                                confidence=0.5, detail={"side": side}))
    return events


def _zones(tl: Timeline, sides: Dict[int, str], cfg: RuleConfig) -> List[Event]:
    points = play_points(tl)
    hold = tl.frames_for(cfg.play_zone_hold_s)
    events: List[Event] = []
    third: Optional[str] = None
    flank: Optional[str] = None
    pending_third: Tuple[Optional[str], int] = (None, 0)
    pending_flank: Tuple[Optional[str], int] = (None, 0)
    last_box = -1e9
    box_run = 0
    seen_outside = False  # "into the box" needs the ball to have been seen outside it first
    for k, pt in enumerate(points):
        if pt is None:
            continue
        t = float(tl.t[k])

        now_third = _third(pt[0])
        if now_third == third:
            pending_third = (None, 0)
        else:
            count = pending_third[1] + 1 if pending_third[0] == now_third else 1
            pending_third = (now_third, count)
            if count >= hold:
                if third is not None:
                    if now_third in ("left", "right"):
                        events.append(Event("final_third", t, t,
                                            team=_team_attacking(sides, now_third),
                                            xy=pt, confidence=0.5,
                                            detail={"side": now_third}))
                    else:
                        events.append(Event("midfield", t, t, xy=pt, confidence=0.5))
                third, pending_third = now_third, (None, 0)

        now_flank = _flank(pt[1])
        if now_flank == flank:
            pending_flank = (None, 0)
        else:
            count = pending_flank[1] + 1 if pending_flank[0] == now_flank else 1
            pending_flank = (now_flank, count)
            if count >= hold:
                if flank is not None and now_flank != "centre":
                    events.append(Event("flank", t, t, xy=pt, confidence=0.5,
                                        detail={"flank": now_flank}))
                flank, pending_flank = now_flank, (None, 0)

        # The ball itself (not the fallback) inside a penalty area.
        side = "left" if pt[0] < LENGTH / 2 else "right"
        in_box = tl.known[k] and in_penalty_area(pt[0], pt[1], side, -0.5)
        box_run = box_run + 1 if in_box else 0
        if tl.known[k] and not in_box:
            seen_outside = True
        if seen_outside and box_run == tl.frames_for(0.3) \
                and t - last_box >= cfg.play_box_repeat_s:
            last_box = t
            events.append(Event("box_entry", t, t, team=_team_attacking(sides, side), xy=pt,
                                confidence=0.5, detail={"side": side}))
    return events


def _on_the_ball(segments: Sequence[Possession], sides: Dict[int, str],
                 cfg: RuleConfig) -> List[Event]:
    events: List[Event] = []
    last_said = -1e9
    previous: Optional[Possession] = None
    for seg in segments:
        moved = dist(seg.first_xy, seg.last_xy)
        if seg.duration >= cfg.play_carry_min_s and moved >= cfg.play_carry_min_m:
            forward = None
            if seg.team in sides:
                gained = goal_line_depth(seg.first_xy[0], sides[seg.team]) - \
                    goal_line_depth(seg.last_xy[0], sides[seg.team])
                forward = gained >= 0.5 * moved
            events.append(Event("carry", seg.t0 + 0.6 * seg.duration, seg.t1, team=seg.team,
                                player=seg.track_id, xy=seg.last_xy, distance_m=moved,
                                confidence=0.5, detail={"forward": forward}))
        new_player = previous is None or previous.track_id != seg.track_id
        if new_player and seg.duration >= cfg.play_on_ball_min_s and seg.team is not None \
                and seg.t0 - last_said >= cfg.play_on_ball_repeat_s:
            last_said = seg.t0
            events.append(Event("on_ball", seg.t0 + 0.3, seg.t1, team=seg.team,
                                player=seg.track_id, xy=seg.first_xy, confidence=0.5,
                                detail={"where": where_words(seg.first_xy, seg.team, sides)}))
        previous = seg
    return events


def _describe_passes(passes: Sequence[Event], sides: Dict[int, str]) -> None:
    """Label each pass forward, back or square, from the passer's point of view."""
    for e in passes:
        if e.type not in ("pass", "long_pass") or e.team not in sides or e.xy is None:
            continue
        receiver_xy = e.detail.get("to_xy")
        if receiver_xy is None:
            continue
        gained = goal_line_depth(e.xy[0], sides[e.team]) - \
            goal_line_depth(receiver_xy[0], sides[e.team])
        e.detail["direction"] = "forward" if gained > 5.0 else "back" if gained < -5.0 \
            else "square"
