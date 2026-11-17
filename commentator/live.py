"""Live commentary: decide and speak while the match is still going.

The picture is shown a few seconds behind the camera (`delay_s`), as TV broadcasts are. Every
rule in the event engine reaches its verdict within that delay, so a line can be spoken
exactly when its moment appears on screen. Without a delay only set-piece build-ups would feel
live; shots and goals would trail the action.

Three parts:

* `EventTracker` runs the rule engine over a rolling window and reports only what is new: a set
  piece being prepared, the same set piece being taken, a shot, a goal, and, when it has to,
  a goal call that later evidence overturns.
* `Analyst` is the second voice. It keeps running match facts (shots, corners per team) and
  reacts a beat after the lead commentator, the way a co-commentator does.
* `LiveDirector` owns the microphone: one voice at a time, big moments first, stale remarks
  dropped, and a goal is allowed to cut across small talk.

All clocks are in seconds of stream time. Nothing here looks at the future.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .commentary import (PRIORITY, Moment, Narrator, event_data, shot_kind, spell_small,
                         tone_of)
from .config import DEFAULT, RuleConfig
from .core import Event, FrameState
from .events import detect_events
from .speech import SAMPLE_RATE, Booth
from .timeline import Timeline

SET_PIECES = ("corner_kick", "free_kick", "kick_off", "goal_kick", "penalty", "throw_in",
              "long_throw")
ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth".split()


@dataclass
class Spoken:
    """A line that went to air."""

    start: float  # stream clock when the voice started
    text: str
    kind: str
    priority: int
    speaker: str
    tone: str
    audio: np.ndarray
    event_t: float  # when the thing happened in the match
    decided_t: float  # when the system knew
    due: float  # when the moment appears on the delayed picture
    cut_at: Optional[float] = None  # set if a bigger moment interrupted this line

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.cut_at if self.cut_at is not None else self.start + self.duration

    @property
    def late_by(self) -> float:
        """Seconds after its moment appeared on screen that the line started. 0 = in sync."""
        return max(0.0, self.start - self.due)


class EventTracker:
    """Turns successive rule-engine outputs into a stream of new moments."""

    def __init__(self, setup_after_s: float = 1.5):
        self.setup_after_s = setup_after_s
        self.set_pieces: List[dict] = []
        self.shots: List[dict] = []
        self.others: List[tuple] = []
        self.pass_run: List[Event] = []

    def diff(self, events: Sequence[Event], now: float) -> List[Moment]:
        out: List[Moment] = []
        for e in events:
            if e.type in SET_PIECES:
                out += self._set_piece(e, now)
            elif e.type in ("shot", "goal"):
                out += self._shot(e, now)
            else:
                out += self._other(e, events)
        self._forget(now)
        return out

    # -- set pieces: say it is coming, then say it was taken --------------------------------
    def _set_piece(self, e: Event, now: float) -> List[Moment]:
        family = "throw_in" if e.type == "long_throw" else e.type
        setup_t = float(e.detail.get("setup_t", e.t))
        state = next((s for s in self.set_pieces
                      if s["family"] == family and abs(s["setup_t"] - setup_t) <= 2.5), None)
        if state is None:
            state = {"family": family, "setup_t": setup_t, "setup": False, "crowd": False,
                     "taken": False}
            self.set_pieces.append(state)
        data = event_data(e)
        out: List[Moment] = []
        taken = bool(e.detail.get("taken", True))
        waited = now - setup_t

        if not state["setup"] and not state["taken"] and waited >= self.setup_after_s \
                and (not taken or e.t - setup_t >= 2.5):
            kind = f"{family}.setup"
            if family == "free_kick" and data.get("range", 99) <= 35:
                kind = "free_kick.setup.near"
            out.append(Moment(setup_t + 0.8, kind, PRIORITY.get(f"{family}.setup", 34),
                              e.team, e, data))
            state["setup"] = True
        crowd = int(e.detail.get("attackers_in_box", 0))
        if family == "corner_kick" and state["setup"] and not state["crowd"] \
                and not taken and crowd >= 3 and waited >= 3.2:
            out.append(Moment(setup_t + 3.6, "corner_kick.crowd", 40, e.team, e,
                              {**data, "count": spell_small(crowd).capitalize()}))
            state["crowd"] = True
        if taken and not state["taken"]:
            out.append(Moment(e.t, e.type, PRIORITY[e.type], e.team, e, data))
            state["taken"] = True
        return out

    # -- shots and goals: call it, and take it back if later evidence says otherwise --------
    def _shot(self, e: Event, now: float) -> List[Moment]:
        side = e.detail.get("goal_side")
        state = next((s for s in self.shots
                      if s["side"] == side and abs(s["t"] - e.t) <= 2.5), None)
        data = event_data(e)
        if state is None:
            self.shots.append({"side": side, "t": e.t, "said": e.type})
            if e.type == "goal":
                return [Moment(e.end_t, "goal", PRIORITY["goal"], e.team, e, data)]
            return [Moment(e.t, shot_kind(e), PRIORITY["shot"], e.team, e, data)]
        if state["said"] == "shot" and e.type == "goal":
            state["said"] = "goal"
            return [Moment(e.end_t, "goal", PRIORITY["goal"], e.team, e, data)]
        if state["said"] == "goal" and e.type == "shot":
            state["said"] = "cancelled"
            # Due the instant it is known: a correction refers to "now", not to the past.
            return [Moment(now, "goal.cancelled", PRIORITY["goal.cancelled"], e.team, e,
                           {**data, "immediate": True})]
        return []

    # -- everything else: once ---------------------------------------------------------------
    def _other(self, e: Event, events: Sequence[Event]) -> List[Moment]:
        if any(kind == e.type and abs(t - e.t) <= 1.0 for kind, t in self.others):
            return []
        self.others.append((e.type, e.t))
        data = event_data(e)
        if e.type in ("pass", "long_pass"):
            if self.pass_run and (self.pass_run[-1].team != e.team
                                  or e.t - self.pass_run[-1].t > 8.0):
                self.pass_run = []
            self.pass_run.append(e)
            out = []
            if len(self.pass_run) == 3:
                out.append(Moment(e.end_t, "spell", PRIORITY["spell"], e.team, e))
            if e.type == "long_pass":
                out.append(Moment(e.t, "long_pass", PRIORITY["long_pass"], e.team, e, data))
            return out
        if e.type == "turnover":
            self.pass_run = []
        if e.type == "replay":
            big = any(b.type in ("goal", "penalty", "shot", "save") and 0.0 <= e.t - b.t <= 12.0
                      for b in events)
            return [Moment(e.t + 0.6, "replay", PRIORITY["replay"], None, e, data)] if big else []
        if e.type in PRIORITY:
            return [Moment(e.t, e.type, PRIORITY[e.type], e.team, e, data)]
        return []

    def _forget(self, now: float, keep_s: float = 90.0) -> None:
        self.set_pieces = [s for s in self.set_pieces if now - s["setup_t"] <= keep_s]
        self.shots = [s for s in self.shots if now - s["t"] <= keep_s]
        self.others = [(k, t) for k, t in self.others if now - t <= keep_s]


class Analyst:
    """The co-commentator: reacts to what the lead has just called, using running facts."""

    def __init__(self, min_gap_s: float = 7.0):
        self.min_gap_s = min_gap_s
        self.shots: Counter = Counter()
        self.corners: Counter = Counter()
        self.pressure: Counter = Counter()
        self.last_remark = -1e9

    def react(self, moment: Moment, after: float) -> Optional[Moment]:
        """A remark to follow `moment`, due at stream-match time `after`. None = stay quiet."""
        kind, team = moment.kind, moment.team
        topic: Optional[str] = None
        data: Dict[str, object] = {}
        base = kind.split(".")[0]

        if kind == "goal":
            topic = "colour.goal"
        elif kind == "goal.cancelled":
            topic = "colour.cancelled"
        elif base == "shot":
            self.shots[team] += 1
            n = self.shots[team]
            if kind == "shot.off_target":
                topic = "colour.miss"
            elif n == 1:
                topic = "colour.first_shot"
            elif n <= len(ORDINALS):
                topic, data = "colour.shots", {"nth": ORDINALS[n - 1]}
        elif kind == "save":
            topic = "colour.save"
        elif kind == "penalty.setup":
            topic = "colour.penalty"
        elif kind == "corner_kick":
            self.corners[team] += 1
            n = self.corners[team]
            if n == 1:
                topic = "colour.first_corner"
            elif n <= len(ORDINALS):
                topic, data = "colour.corner", {"n": spell_small(n), "nth": ORDINALS[n - 1]}
        elif kind == "free_kick.setup.near":
            topic = "colour.free_kick"
        elif kind == "kick_off":
            topic = "colour.kick_off"
        elif kind == "long_throw":
            topic = "colour.long_throw"
        elif kind == "pressure":
            self.pressure[team] += 1
            if self.pressure[team] >= 2:
                topic = "colour.pressure"

        if topic is None:
            return None
        big = kind in ("goal", "goal.cancelled", "save", "penalty.setup")
        if not big and after - self.last_remark < self.min_gap_s:
            return None
        self.last_remark = after
        return Moment(after, topic, PRIORITY["colour"], team, moment.event, data,
                      speaker="analyst")


def _ttl(priority: int) -> float:
    """How long after its moment a line is still worth saying."""
    return 6.0 if priority >= 75 else 3.5 if priority >= 40 else 2.0


@dataclass
class _Pending:
    moment: Moment
    text: str
    audio: np.ndarray
    due: float
    ready: float
    decided: float
    tone: str


class LiveDirector:
    """Decides, at every instant, whether somebody should start speaking and what they say."""

    def __init__(self, lead: Narrator, booth: Booth, analyst_voice: Optional[Narrator] = None,
                 analyst: Optional[Analyst] = None, delay_s: float = 4.0, gap_s: float = 0.3,
                 quiet_s: float = 1.5):
        self.lead = lead
        self.analyst_voice = analyst_voice or lead
        self.analyst = analyst
        self.booth = booth
        self.delay_s, self.gap_s, self.quiet_s = delay_s, gap_s, quiet_s
        self.pending: List[_Pending] = []
        self.aired: List[Spoken] = []
        self.dropped: List[tuple] = []
        self.busy_until = -1e9
        self.seconds_wording = 0.0

    def offer(self, moment: Moment, now: float) -> None:
        """Hand the director something that could be said. Its voice is prepared at once, so
        it is ready by the time the moment reaches the screen."""
        import time

        narrator = self.analyst_voice if moment.speaker == "analyst" else self.lead
        started = time.time()
        text = narrator.say(moment)
        thinking = time.time() - started  # near zero for templates, real for a language model
        if not text:
            return
        if any(s.text == text and now - s.start < 12.0 for s in self.aired[-4:]):
            return  # do not repeat a sentence that has just been said
        tone = tone_of(moment.kind)
        audio, cost = self.booth.say(text, moment.speaker, tone)
        due = now if moment.data.get("immediate") else moment.t + self.delay_s
        self.seconds_wording += thinking
        self.pending.append(_Pending(moment, text, audio, due, now + thinking + cost, now, tone))

    def poll(self, now: float) -> List[Spoken]:
        """Start whichever line should start now. Call this often (every frame)."""
        started: List[Spoken] = []
        for p in list(self.pending):
            if now - p.due > _ttl(p.moment.priority):
                self.pending.remove(p)
                self.dropped.append((p.moment.kind, p.text, "too late to be worth saying"))
        while True:
            ready = [p for p in self.pending if p.due <= now and p.ready <= now]
            if not ready:
                break
            best = max(ready, key=lambda p: (p.moment.priority, -p.due))
            if now < self.busy_until:
                current = self.aired[-1]
                may_interrupt = (best.moment.priority >= 90 and current.priority < 60
                                 and self.busy_until - now > 0.6)
                if not may_interrupt:
                    break
                current.cut_at = now
                self.busy_until = now
            else:
                pad = self.gap_s if best.moment.priority >= 40 else self.quiet_s
                if self.aired and now < self.aired[-1].end + pad:
                    break
            self.pending.remove(best)
            m = best.moment
            line = Spoken(now, best.text, m.kind, m.priority, m.speaker, best.tone, best.audio,
                          event_t=m.t, decided_t=best.decided, due=best.due)
            self.aired.append(line)
            started.append(line)
            self.busy_until = line.start + line.duration
            if self.analyst is not None and m.speaker == "lead":
                follow_up = self.analyst.react(m, after=line.start + line.duration + 0.4
                                               - self.delay_s)
                if follow_up is not None:
                    self.offer(follow_up, now)
        return started


class LiveCommentator:
    """Glue: feed it the current game-state window as the stream advances."""

    def __init__(self, director: LiveDirector, cfg: RuleConfig = DEFAULT,
                 tick_s: float = 0.5):
        self.director = director
        self.tracker = EventTracker()
        self.cfg = cfg
        self.tick_s = tick_s
        self._last_tick = -1e9
        self.events: List[Event] = []
        self.seconds_in_rules = 0.0
        self.ticks = 0

    def due_for_rules(self, now: float) -> bool:
        return now - self._last_tick >= self.tick_s

    def update(self, frames: Sequence[FrameState], fps: float, now: float) -> None:
        """Run the rules on everything known up to `now` and queue whatever is new."""
        import time

        self._last_tick = now
        start = time.time()
        self.events = detect_events(Timeline(frames, fps), self.cfg)
        self.seconds_in_rules += time.time() - start
        self.ticks += 1
        for moment in self.tracker.diff(self.events, now):
            self.director.offer(moment, now)

    def speak(self, now: float) -> List[Spoken]:
        return self.director.poll(now)


def latency_report(aired: Sequence[Spoken], delay_s: float) -> Dict[str, object]:
    """How live the commentary was."""
    if not aired:
        return {"lines": 0}
    knew_after = np.array([max(0.0, s.decided_t - s.event_t) for s in aired])
    late = np.array([s.late_by for s in aired])
    return {
        "lines": len(aired),
        "delay_buffer_s": delay_s,
        "knew_after_event_s": {"median": round(float(np.median(knew_after)), 2),
                               "worst": round(float(knew_after.max()), 2)},
        "started_late_vs_picture_s": {"median": round(float(np.median(late)), 2),
                                      "worst": round(float(late.max()), 2)},
        "in_sync_lines": int((late <= 0.3).sum()),
        "interrupted_lines": sum(s.cut_at is not None for s in aired),
    }
