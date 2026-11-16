"""Runs every rule over a timeline and returns one time-ordered event list."""

from __future__ import annotations

from typing import List, Tuple

from ..config import DEFAULT, RuleConfig
from ..core import GOALKEEPER, Event
from ..pitch import goal_line_depth, nearest_goal_side
from ..possession import possession_segments
from ..timeline import Timeline
from .passing import detect_passes
from .phases import detect_phases
from .setpieces import detect_kickoffs_by_formation, detect_set_pieces
from .shots import detect_shots
from .throwins import detect_throw_ins


def detect_events(tl: Timeline, cfg: RuleConfig = DEFAULT) -> List[Event]:
    segments = possession_segments(tl, cfg)
    set_pieces = detect_set_pieces(tl, cfg)
    set_pieces += detect_kickoffs_by_formation(tl, set_pieces, cfg)
    throw_ins = detect_throw_ins(tl, segments, cfg)
    shots = detect_shots(tl, segments, set_pieces, cfg)

    # No pass is reported while a set piece is waiting or a shot is in flight.
    blocked: List[Tuple[float, float]] = []
    for e in set_pieces:
        blocked.append((float(e.detail.get("setup_t", e.t)), e.t + cfg.after_set_piece_s))
    for e in shots:
        if e.type in ("shot", "goal"):
            blocked.append((e.t - 0.2, e.end_t))
    for e in throw_ins:
        blocked.append((e.t - 1.0, e.end_t + cfg.after_set_piece_s))
    passes = detect_passes(segments, blocked, cfg)

    _reconcile(set_pieces, shots, passes, segments, cfg)
    events = set_pieces + throw_ins + shots + passes + detect_phases(tl, cfg)
    events.sort(key=lambda e: (e.t, e.type))
    return events


def _reconcile(set_pieces: List[Event], shots: List[Event], passes: List[Event],
               segments, cfg: RuleConfig) -> None:
    """Use what happens next to settle what a shot was.

    A goal kick at the same end means the ball went out off the attacker: no goal. A corner at
    the same end means it was saved or blocked behind. The goalkeeper playing the ball out from
    inside the pitch also rules out a goal.
    """
    for sh in shots:
        if sh.type not in ("shot", "goal"):
            continue
        side = sh.detail.get("goal_side")
        for sp in set_pieces:
            if not (0.0 < sp.t - sh.t <= cfg.reconcile_window_s) or sp.xy is None:
                continue
            if nearest_goal_side(sp.xy[0]) != side:
                continue
            if sp.type == "goal_kick":
                sh.type = "shot"
                sh.detail["outcome"] = "off_target"
                break
            if sp.type == "corner_kick" and sh.type == "shot":
                sh.detail["outcome"] = "behind_for_corner"
                break
        if sh.type == "goal":
            for p in passes:
                keeper_restart = (p.type in ("hand_throw", "pass", "long_pass")
                                  and p.detail.get("from_keeper")
                                  and 0.0 < p.t - sh.t <= cfg.keeper_restart_window_s
                                  and p.xy is not None
                                  and nearest_goal_side(p.xy[0]) == side
                                  and goal_line_depth(p.xy[0], side) >= 0.0)
                if keeper_restart:
                    sh.type = "shot"
                    sh.detail["outcome"] = "off_target"
                    break
        if sh.type == "goal":
            # The keeper back on the pitch with the ball at their feet or in their hands:
            # it was saved or gathered, not scored.
            for seg in segments:
                if (seg.role == GOALKEEPER and seg.duration >= cfg.keeper_hold_min_s
                        and 0.0 < seg.t0 - sh.t <= cfg.keeper_restart_window_s
                        and nearest_goal_side(seg.last_xy[0]) == side
                        and goal_line_depth(seg.last_xy[0], side) >= 1.0):
                    sh.type = "shot"
                    sh.detail["outcome"] = "saved"
                    break
    # A penalty that produced a shot or goal: record the outcome on the penalty itself.
    for sp in set_pieces:
        if sp.type != "penalty":
            continue
        for sh in shots:
            if sh.detail.get("source") == "penalty" and abs(sh.t - sp.t) < 1.0:
                sp.detail["outcome"] = sh.detail.get("outcome")
    # A ball-in-net "goal" with no known shooter that turned out not to be one leaves nothing.
    shots[:] = [e for e in shots
                if not (e.type == "shot" and e.detail.get("source") == "ball_in_net")]
