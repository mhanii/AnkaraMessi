"""Events in, spoken lines out.

Two jobs, kept apart on purpose:

* The **narrator** words a single moment. `TemplateNarrator` is the free, deterministic one.
  An LLM narrator can replace it behind the same `Narrator` protocol; it must only choose
  words, never facts.
* The **director** decides which moments get a line and when, so that speech never talks over
  the next important thing.

Facts come only from events. A line is never invented without one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from .core import Event


@dataclass
class Moment:
    """Something worth saying: an event seen from one angle (waiting, happening, outcome)."""

    t: float
    kind: str  # e.g. "corner_kick.setup", "goal", "shot.off_target", "spell"
    priority: int
    team: Optional[int] = None
    event: Optional[Event] = None
    data: Dict[str, object] = field(default_factory=dict)


@dataclass
class Line:
    t: float
    text: str
    kind: str
    priority: int
    duration: float = 0.0  # seconds of speech; estimated until audio exists

    @property
    def end(self) -> float:
        return self.t + self.duration


class Narrator(Protocol):
    def say(self, moment: Moment) -> Optional[str]:
        ...


PRIORITY = {
    "goal": 100, "penalty": 90, "penalty.setup": 88, "save": 80, "shot": 75,
    "corner_kick": 60, "corner_kick.setup": 58, "free_kick": 55, "free_kick.setup": 50,
    "kick_off": 52, "kick_off.setup": 45, "goal_kick": 40, "goal_kick.setup": 30,
    "hand_throw": 38, "throw_in": 36, "long_throw": 42, "long_pass": 35, "turnover": 25,
    "replay": 32, "pressure": 28, "spell": 20,
}
BIG_MOMENTS = ("goal", "penalty", "shot", "save")

TEMPLATES: Dict[str, List[str]] = {
    "goal": ["GOAL! {team} have scored!", "It's in! A goal for {team}!",
             "He's done it! {team} find the net!"],
    "goal.unsure": ["That looks like it's in! {team} think they've scored!",
                    "Is that a goal? It looks like it's crossed the line for {team}!"],
    "goal.nobody": ["It's in the net! A goal!", "The ball is in the back of the net!"],
    "penalty.setup": ["Penalty to {team}! The ball is on the spot.",
                      "{team} have a penalty. Everyone else is out of the box."],
    "penalty": ["Here's the run-up... and it's struck!", "The penalty is taken!"],
    "save": ["Saved! The {team} keeper gets there.", "Good stop by the {team} goalkeeper!"],
    "shot": ["A shot from {team}!", "{team} have a go from {range} metres!"],
    "shot.saved": ["{team} shoot, and the keeper deals with it.",
                   "An effort on target from {team}."],
    "shot.off_target": ["{team} shoot, but it's off target.",
                        "A chance for {team}, and it goes wide."],
    "shot.behind_for_corner": ["{team} shoot, it's turned behind! That will be a corner.",
                               "An effort from {team}, deflected behind for a corner."],
    "corner_kick.setup": ["Corner for {team}, over on the {corner} side.",
                          "{team} have a corner to come from the {corner} flag."],
    "corner_kick": ["The corner is swung in!", "In it comes!"],
    "free_kick.setup": ["Free kick to {team}.", "{team} have a free kick here."],
    "free_kick": ["The free kick is taken.", "{team} take it."],
    "kick_off.setup": ["Both teams are set. We're ready for kick-off.",
                       "The players are in position for the kick-off."],
    "kick_off": ["And we're under way!", "Kick-off! Here we go."],
    "goal_kick.setup": ["It's a goal kick.", "Goal kick to come."],
    "goal_kick": ["The goal kick is taken long.", "The keeper restarts play."],
    "hand_throw": ["The {team} keeper throws it out to start again.",
                   "Quick distribution from the {team} goalkeeper, out of his hands."],
    "throw_in": ["Throw-in for {team}.", "{team} take the throw."],
    "long_throw": ["A long throw from {team}, launched into the mix!",
                   "{team} hurl a long throw towards the box!"],
    "long_pass": ["{team} go long, {distance} metres through the air.",
                  "A big switch of play from {team}."],
    "turnover": ["{team} win it back.", "Possession changes hands, {team} have it now."],
    "pressure": ["{team} have {count} players up in the final third.",
                 "{team} are pushing men forward here.",
                 "It's all {team} for the moment, camped in the attacking third."],
    "replay": ["Let's have another look at that.", "Here's the replay."],
    "spell": ["{team} are keeping the ball well.", "Patient passing from {team}.",
              "{team} knocking it around."],
}


class TemplateNarrator:
    def __init__(self, team_names: Optional[Dict[int, str]] = None, seed: int = 7):
        self.team_names = team_names or {}
        self.rng = random.Random(seed)
        self._last: Dict[str, str] = {}

    def team(self, team: Optional[int]) -> Optional[str]:
        if team is None or team not in self.team_names:
            return None
        return self.team_names[team]

    def say(self, moment: Moment) -> Optional[str]:
        kind = moment.kind
        team = self.team(moment.team)
        if kind == "goal":
            if team is None:
                kind = "goal.nobody"
            elif moment.event is not None and moment.event.confidence < 0.6:
                kind = "goal.unsure"
        options = TEMPLATES.get(kind)
        if not options:
            return None
        usable = [o for o in options if team is not None or "{team}" not in o]
        if not usable:
            return None
        # Avoid saying the exact same sentence twice in a row for one kind of moment.
        fresh = [o for o in usable if o != self._last.get(kind)] or usable
        template = self.rng.choice(fresh)
        self._last[kind] = template
        text = template.format(team=team, **moment.data)
        return text[0].upper() + text[1:]


def moments_from_events(events: Sequence[Event]) -> List[Moment]:
    """Expand events into sayable moments, including the wait before a set piece."""
    out: List[Moment] = []
    for e in events:
        data: Dict[str, object] = {}
        if e.distance_m is not None:
            data["distance"] = int(round(e.distance_m))
        if "range_m" in e.detail:
            data["range"] = int(round(float(e.detail["range_m"])))
        if "corner" in e.detail:
            data["corner"] = str(e.detail["corner"]).replace("-", " ")

        if e.type in ("corner_kick", "free_kick", "kick_off", "goal_kick", "penalty"):
            setup_t = float(e.detail.get("setup_t", e.t))
            taken = bool(e.detail.get("taken"))
            if e.t - setup_t >= 2.5 or not taken:
                out.append(Moment(setup_t + 0.8, f"{e.type}.setup", PRIORITY[f"{e.type}.setup"],
                                  e.team, e, data))
            if taken:
                out.append(Moment(e.t, e.type, PRIORITY[e.type], e.team, e, data))
        elif e.type == "shot":
            outcome = e.detail.get("outcome")
            kind = f"shot.{outcome}" if f"shot.{outcome}" in TEMPLATES else "shot"
            out.append(Moment(e.t, kind, PRIORITY["shot"], e.team, e, data))
        elif e.type == "replay":
            # Only worth a word when the broadcast cuts away right after something big.
            if any(b.type in BIG_MOMENTS and 0.0 <= e.t - b.t <= 12.0 for b in events):
                out.append(Moment(e.t + 0.6, "replay", PRIORITY["replay"], None, e, data))
        elif e.type == "pressure":
            data["count"] = spell_small(int(e.detail.get("players_forward", 5)))
            out.append(Moment(e.t, "pressure", PRIORITY["pressure"], e.team, e, data))
        elif e.type in PRIORITY:
            out.append(Moment(e.t, e.type, PRIORITY[e.type], e.team, e, data))

    # A run of completed passes by one team is worth one remark, not one per pass.
    run_team, run = None, []
    for e in [x for x in events if x.type in ("pass", "long_pass", "turnover")]:
        if e.type == "turnover" or e.team != run_team or (run and e.t - run[-1].t > 8.0):
            run_team, run = (e.team if e.type != "turnover" else None), []
            if e.type == "turnover":
                continue
        run.append(e)
        if len(run) == 3:
            out.append(Moment(e.end_t, "spell", PRIORITY["spell"], e.team, e))
    out.sort(key=lambda m: m.t)
    return out


def spell_small(n: int) -> str:
    words = "zero one two three four five six seven eight nine ten eleven".split()
    return words[n] if 0 <= n < len(words) else str(n)


def estimate_duration(text: str, words_per_second: float = 2.9) -> float:
    return max(0.8, len(text.split()) / words_per_second + 0.2)


def direct(moments: Sequence[Moment], narrator: Narrator, clip_length: Optional[float] = None,
           gap_s: float = 0.3, max_delay_s: float = 1.5, quiet_s: float = 1.5) -> List[Line]:
    """Pick the lines to speak. The most important moments are placed first; a lesser one is
    kept only if it fits in a silence, sliding up to `max_delay_s` later to do so. Minor
    remarks also need `quiet_s` of silence around them, so the commentary can breathe."""
    placed: List[Line] = []
    for m in sorted(moments, key=lambda m: (-m.priority, m.t)):
        text = narrator.say(m)
        if not text:
            continue
        duration = estimate_duration(text)
        pad = gap_s if m.priority >= 50 else quiet_s
        start = _first_free_slot(placed, m.t, duration, pad, max_delay_s)
        if start is None:
            continue
        if clip_length is not None and start + duration > clip_length + 2.0:
            continue
        placed.append(Line(start, text, m.kind, m.priority, duration))
    placed.sort(key=lambda ln: ln.t)
    return placed


def _first_free_slot(placed: Sequence[Line], t: float, duration: float, pad: float,
                     max_delay: float) -> Optional[float]:
    start = max(0.0, t)
    for _ in range(len(placed) + 1):
        clash = next((ln for ln in placed
                      if start < ln.end + pad and ln.t < start + duration + pad), None)
        if clash is None:
            return start
        start = clash.end + pad
        if start - t > max_delay:
            return None
    return None


def retime(lines: List[Line], durations: Sequence[float], gap_s: float = 0.25) -> List[Line]:
    """Replace estimated durations with real audio lengths and remove any overlap that
    creates: a later line slides back a little, or is dropped if it is the lesser one."""
    for ln, d in zip(lines, durations):
        ln.duration = d
    out: List[Line] = []
    for ln in sorted(lines, key=lambda x: x.t):
        if out and ln.t < out[-1].end + gap_s:
            delay = out[-1].end + gap_s - ln.t
            if delay <= 1.5:
                ln.t += delay
            elif ln.priority > out[-1].priority:
                out[-1] = ln
                continue
            else:
                continue
        out.append(ln)
    return out


def format_script(lines: Sequence[Line]) -> str:
    return "\n".join(f"[{int(ln.t // 60):02d}:{ln.t % 60:04.1f}] {ln.text}" for ln in lines)
