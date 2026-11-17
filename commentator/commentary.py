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
    speaker: str = "lead"  # "lead" calls the play; "analyst" reacts to it


@dataclass
class Line:
    t: float
    text: str
    kind: str
    priority: int
    duration: float = 0.0  # seconds of speech; estimated until audio exists
    speaker: str = "lead"
    tone: str = "calm"

    @property
    def end(self) -> float:
        return self.t + self.duration


class Narrator(Protocol):
    def say(self, moment: Moment) -> Optional[str]:
        ...


PRIORITY = {
    # 55 and above are time-critical calls: they may cut across small talk (see live.py).
    "goal": 100, "goal.cancelled": 96, "penalty": 90, "penalty.setup": 88, "save": 80,
    "shot": 75, "corner_kick": 60, "corner_kick.setup": 58, "long_throw": 57,
    "goal_kick": 56, "free_kick": 55, "throw_in": 55, "hand_throw": 55, "kick_off": 55,
    # Below: said when there is room.
    "free_kick.setup": 50, "box_entry": 48, "break": 46, "kick_off.setup": 45, "colour": 44,
    "goal_kick.setup": 43, "long_pass": 35, "final_third": 34, "replay": 32, "carry": 31,
    "pressure": 28, "turnover": 25, "flank": 24, "pass": 22, "spell": 20, "shape_deep": 19,
    "high_line": 19, "on_ball": 18, "midfield": 16, "numbers": 15, "situation": 14,
    "filler": 5,
}
BIG_MOMENTS = ("goal", "penalty", "shot", "save")

# How a line should sound. The voice is rendered differently for each (see speech.py).
TONES = {"excited": ("goal", "penalty", "save", "shot", "corner_kick", "long_throw",
                     "kick_off", "free_kick", "break", "box_entry", "carry"),
         "tense": ("penalty.setup", "free_kick.setup.near", "corner_kick.setup", "final_third",
                   "pressure",
                   "corner_kick.crowd", "colour.penalty", "colour.free_kick"),
         "down": ("shot.off_target", "goal.cancelled", "colour.cancelled", "colour.miss")}


def tone_of(kind: str) -> str:
    for tone in ("down", "tense", "excited"):  # most specific first
        if kind in TONES[tone]:
            return tone
    base = kind.split(".")[0]
    if kind.startswith("shot.") or base in ("goal",):
        return "excited"
    return "calm"


TEMPLATES: Dict[str, List[str]] = {
    # ---- lead commentator: calls the play ----
    "goal": ["GOAL! {team} have scored!", "It's in! What a moment for {team}!",
             "Back of the net! {team} strike!", "Oh, he's done it! {team} find the net!",
             "Yes! {team} have the breakthrough!"],
    "goal.unsure": ["That looks like it's in! {team} think they've scored!",
                    "Is that a goal? It looks like it's crossed the line for {team}!"],
    "goal.nobody": ["It's in the net! A goal!", "The ball is in the back of the net!",
                    "Oh, that's in! Goal!"],
    "goal.cancelled": ["No! Hold on, it's not a goal. It stayed out.",
                       "Wait, no goal! I called that too early."],
    "penalty.setup": ["Penalty to {team}! The ball is on the spot.",
                      "{team} have a penalty. Everyone else is out of the box.",
                      "It's a penalty for {team}. Just the taker and the keeper now."],
    "penalty": ["Here's the run-up... and it's struck!", "The penalty is taken!",
                "He steps up... and hits it!"],
    "save": ["Saved! The {team} keeper gets there.", "Good stop by the {team} goalkeeper!",
             "Kept out! Strong hands from the {team} keeper."],
    "shot": ["A shot from {team}!", "{team} have a go from {range} metres!",
             "{team} let fly!", "An effort at goal from {team}!"],
    "shot.saved": ["{team} shoot, and the keeper deals with it.",
                   "An effort on target from {team}, and it's held.",
                   "{team} test the keeper!"],
    "shot.off_target": ["{team} shoot, but it's off target.",
                        "A chance for {team}, and it goes wide.",
                        "Oh, {team} drag it past the post."],
    "shot.behind_for_corner": ["{team} shoot, it's turned behind! That will be a corner.",
                               "An effort from {team}, deflected behind for a corner."],
    "corner_kick.setup": ["Corner for {team}, over on the {corner} side.",
                          "{team} have a corner to come from the {corner} flag.",
                          "It's a corner to {team}, {corner} side."],
    "corner_kick.crowd": ["{count} {team} players are waiting in the box.",
                          "{team} have {count} in the area for this one."],
    "corner_kick": ["The corner is swung in!", "In it comes!", "Here's the delivery!"],
    "free_kick.setup": ["Free kick to {team}.", "{team} have a free kick here.",
                        "The referee gives {team} a free kick."],
    "free_kick.setup.near": ["Free kick to {team}, about {range} metres from goal.",
                             "{team} have a free kick in a dangerous spot, {range} metres out."],
    "free_kick": ["The free kick is taken.", "{team} take it.", "And it's played."],
    "kick_off.setup": ["Both teams are set. We're ready for kick-off.",
                       "The players are in position for the kick-off."],
    "kick_off": ["And we're under way!", "Kick-off! Here we go.", "Off we go!"],
    "goal_kick.setup": ["It's a goal kick.", "Goal kick to come."],
    "goal_kick": ["The goal kick is taken long.", "The keeper restarts play.",
                  "Launched upfield by the keeper."],
    "hand_throw": ["The {team} keeper throws it out to start again.",
                   "Quick distribution from the {team} goalkeeper, out of his hands."],
    "throw_in.setup": ["Throw-in to come for {team}, on the {touchline} side.",
                       "{team} have a throw on the {touchline} touchline."],
    "throw_in": ["{team} take the throw.", "The throw-in is taken by {team}."],
    "long_throw": ["A long throw from {team}, launched into the mix!",
                   "{team} hurl a long throw towards the box!"],
    "long_pass": ["{team} go long, {distance} metres through the air.",
                  "A big switch of play from {team}.",
                  "{team} look for the long ball."],
    "turnover": ["{team} win it back.", "Possession changes hands, {team} have it now.",
                 "Nicked away by {team}."],
    "pressure": ["{team} have {count} players up in the final third.",
                 "{team} are pushing men forward here.",
                 "It's all {team} for the moment, camped in the attacking third."],
    "replay": ["Let's have another look at that.", "Here's the replay.",
               "Another angle of it for you.", "Watch this again."],
    "spell": ["{team} are keeping the ball well.", "Patient passing from {team}.",
              "{team} knocking it around."],

    # ---- running play-by-play: short and frequent, so it never goes quiet ----
    "pass": ["Neat pass.", "Finds a team-mate.", "{team} keep it moving.", "Tidy from {team}.",
             "Simple ball.", "On it goes."],
    "pass.forward": ["Played forward.", "{team} look to go through the lines.",
                     "Good ball up the pitch.", "Forward it goes.", "Positive from {team}."],
    "pass.back": ["Back it goes.", "{team} recycle possession.", "Played back to keep it.",
                  "No rush from {team}."],
    "pass.square": ["Squared across.", "Simple ball sideways.",
                    "{team} shift it across the pitch.", "Worked across."],
    "on_ball": ["{team} on the ball {where}.", "{team} have it {where}.",
                "It's with {team} {where}.", "{team} in possession {where}."],
    "carry": ["{team} drive forward with it!", "He carries it {distance} metres!",
              "Good run with the ball from {team}!", "He's away, striding forward!",
              "{team} run at them!"],
    "final_third": ["{team} are into the final third.", "{team} push up towards goal.",
                    "Danger building for {team} here.", "{team} are knocking on the door."],
    "final_third.nobody": ["Play moves up towards the {side}-hand goal.",
                           "The action is down at the {side} end now."],
    "box_entry": ["It's in the box!", "Into the penalty area!", "{team} get it into the area!",
                  "Danger here, it's in the box!"],
    "flank": ["Out to the {flank} side.", "They switch it to the {flank} flank.",
              "Width on the {flank} side.", "Worked out to the {flank} touchline."],
    "midfield": ["Back into midfield.", "The game settles in the middle third.",
                 "It's being contested in the middle of the park."],
    "break": ["{team} break forward at pace!", "Here come {team} on the counter!",
              "{team} are away, and they've got numbers!", "Quick transition from {team}!"],
    "break.nobody": ["It's a break, and it's quick!", "Suddenly it opens up!",
                     "They pour forward towards the {side}-hand goal!"],
    "situation.third": ["Play is deep in {team} territory{wide}.",
                        "We're down in the {team} third{wide}.",
                        "{team} are being asked to defend{wide}.",
                        "It's all happening around the {team} box{wide}."],
    "situation.third.nobody": ["Play is down at the {side}-hand end{wide}.",
                               "It's all happening in the {side}-hand third{wide}."],
    "situation.middle": ["They're scrapping for it in the middle third{wide}.",
                         "It's congested in midfield{wide}.",
                         "The game is being played in the middle of the park{wide}.",
                         "No way through yet, it stays in midfield{wide}."],
    "numbers": ["{team} have the extra men around the ball.",
                "{team} are swarming around it.", "Plenty of {team} shirts near the ball."],
    "shape_deep": ["{team} have {count} men behind the ball.", "{team} are sitting deep.",
                   "{team} are happy to defend in numbers."],
    "high_line": ["{team} holding a high line.", "{team} squeeze right up the pitch.",
                  "Brave from {team}, with the back line that high."],
    "filler": ["Good tempo to this game.", "Both sides probing for an opening.",
               "Still all to play for.", "Tight, this. Very tight.",
               "The intensity has not dropped at all.", "You would not want to call this one.",
               "They are making each other work for every yard.",
               "Lovely evening for it, and a game to match.",
               "Concentration is everything at this level.",
               "It could swing either way from here.",
               "You sense a chance is coming.", "The crowd are right into this.",
               "Neither side giving an inch.", "Plenty of energy out there.",
               "This is nicely poised.", "Patience is the word here.",
               "End to end stuff at times.", "A real battle for control."],
    "filler.short": ["Here we go.", "Good tempo.", "Nicely poised.", "Tense stuff.",
                     "On we go.", "Plenty going on.", "Everyone's switched on."],
    "colour.filler.short": ["Good shape.", "I like that.", "Smart play.", "Real quality on show.",
                            "That's the standard."],
    "colour.break": ["That's the pace they have on the counter.",
                     "When they go, they really go.", "That's a lot of grass to run into."],
    "colour.box": ["You cannot switch off for a second in there.",
                   "That's where defenders earn their money.",
                   "Any touch in there could be a goal."],
    "colour.deep": ["They'll invite pressure defending that deep.",
                    "It's a risk, sitting back like that."],
    "colour.high_line": ["That's brave with pace in behind them.",
                         "One ball over the top and they are in trouble."],
    "colour.long_pass": ["That's a lovely range of passing.",
                         "He's seen that early, and it's a great ball."],
    "colour.wait": ["The big men are coming up for this.", "This is a real chance to deliver.",
                    "Watch the movement in the box here.", "Everybody is picking up a man."],
    "colour.filler": ["I like the intent from both sides.", "It's a proper contest, this.",
                      "The first goal is going to be massive.",
                      "Look at the work rate, it's relentless.",
                      "This is where games are won, in the details.",
                      "Somebody needs to take a risk, for me.",
                      "They've clearly done their homework on each other.",
                      "The midfield battle is deciding everything right now.",
                      "I'd like to see them be a bit braver on the ball.",
                      "The shape of both teams is very disciplined.",
                      "Whoever takes their chance wins this, for me.",
                      "You can see how much it means to them.",
                      "They are working so hard off the ball.",
                      "It only takes one moment of quality."],

    # ---- analyst: reacts a beat later, from the running match facts ----
    "colour.goal": ["That is a finish of real quality. {team} took their moment.",
                    "You have to say {team} had been threatening. Now they've got it."],
    "colour.goal.nobody": ["The keeper had no chance with that one.",
                           "That changes the whole feel of this game."],
    "colour.cancelled": ["It looked in from up here, I have to say. Fine margins.",
                         "That's the trouble with calling it live. A let-off for the defence."],
    "colour.first_shot": ["That's the first real sight of goal for {team}.",
                          "{team} hadn't threatened until then."],
    "colour.shots": ["That's the {nth} effort from {team} in this spell. They are getting closer.",
                     "{nth} shot for {team} now. The pressure is building."],
    "colour.miss": ["He'll be disappointed with that. The chance was there.",
                    "You'd want the target hit from there."],
    "colour.save": ["Big save. That keeps his side right in this.",
                    "That's what the keeper is there for. Excellent reactions."],
    "colour.penalty": ["Huge moment. It's keeper against taker and nothing else matters.",
                       "All the pressure is on the taker here."],
    "colour.corner": ["That's corner number {n} for {team}. They are asking questions.",
                      "Another corner for {team}, their {nth}. Deliveries will be key."],
    "colour.first_corner": ["A chance to put it in a dangerous area.",
                            "Set pieces can decide games like this."],
    "colour.free_kick": ["That's shooting range. Expect a wall and a strike.",
                         "From there you fancy a shot, don't you?"],
    "colour.kick_off": ["Let's see who settles first.", "Both sides will want the early ball."],
    "colour.pressure": ["{team} have really pinned them back here.",
                        "It's wave after wave from {team} at the moment."],
    "colour.long_throw": ["A long throw like that is as good as a corner.",
                          "That's a real weapon, the long throw."],
}


class TemplateNarrator:
    def __init__(self, team_names: Optional[Dict[int, str]] = None, seed: int = 7):
        # Keep the caller's dict (even when empty): live, names are filled in as they are learned.
        self.team_names = team_names if team_names is not None else {}
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
        if team is None and f"{kind}.nobody" in TEMPLATES:
            kind = f"{kind}.nobody"
        options = TEMPLATES.get(kind)
        if not options:
            return None
        usable = [o for o in options if team is not None or "{team}" not in o]
        usable = [o for o in usable if all(f"{{{k}}}" not in o or k in moment.data or k == "team"
                                           for k in ("count", "range", "distance", "corner",
                                                     "touchline", "nth", "n", "where",
                                                     "flank", "side", "wide"))]
        if not usable:
            return None
        # Avoid saying the exact same sentence twice in a row for one kind of moment.
        fresh = [o for o in usable if o != self._last.get(kind)] or usable
        template = self.rng.choice(fresh)
        self._last[kind] = template
        text = template.format(team=team, **moment.data)
        return text[0].upper() + text[1:]


def event_data(e: Event) -> Dict[str, object]:
    """The facts of an event that a line may mention."""
    data: Dict[str, object] = {}
    if e.distance_m is not None:
        data["distance"] = int(round(e.distance_m))
    if "range_m" in e.detail:
        data["range"] = int(round(float(e.detail["range_m"])))
    if "corner" in e.detail:
        data["corner"] = str(e.detail["corner"]).replace("-", " ")
    if "touchline" in e.detail:
        data["touchline"] = str(e.detail["touchline"])
    if e.type == "pressure":
        data["count"] = spell_small(int(e.detail.get("players_forward", 5)))
    for key in ("where", "flank", "side"):
        if key in e.detail and e.type in ("on_ball", "flank", "final_third", "box_entry",
                                          "break"):
            data[key] = str(e.detail[key])
    if e.type == "shape_deep":
        data["count"] = spell_small(int(e.detail.get("count", 8)))
    if e.type == "situation":
        data["wide"] = str(e.detail.get("wide", ""))
        data["side"] = str(e.detail.get("side", ""))
    return data


def shot_kind(e: Event) -> str:
    outcome = e.detail.get("outcome")
    return f"shot.{outcome}" if f"shot.{outcome}" in TEMPLATES else "shot"


def moments_from_events(events: Sequence[Event]) -> List[Moment]:
    """Expand events into sayable moments, including the wait before a set piece."""
    out: List[Moment] = []
    for e in events:
        data = event_data(e)
        if e.type in ("throw_in", "long_throw"):
            if not e.detail.get("taken", True) or float(e.detail.get("held_s", 0)) >= 3.0:
                out.append(Moment(float(e.detail.get("setup_t", e.t)) + 0.6, "throw_in.setup",
                                  PRIORITY["throw_in"] - 2, e.team, e, data))
            if e.detail.get("taken", True):
                out.append(Moment(e.t, e.type, PRIORITY[e.type], e.team, e, data))
            continue
        if e.type in ("corner_kick", "free_kick", "kick_off", "goal_kick", "penalty"):
            setup_t = float(e.detail.get("setup_t", e.t))
            taken = bool(e.detail.get("taken"))
            if e.t - setup_t >= 2.5 or not taken:
                kind = f"{e.type}.setup"
                if e.type == "free_kick" and data.get("range", 99) <= 35:
                    kind = "free_kick.setup.near"
                out.append(Moment(setup_t + 0.8, kind, PRIORITY[f"{e.type}.setup"],
                                  e.team, e, data))
            crowd = int(e.detail.get("attackers_in_box", 0))
            if e.type == "corner_kick" and crowd >= 3 and e.t - setup_t >= 5.5:
                out.append(Moment(setup_t + 3.6, "corner_kick.crowd", 40, e.team, e,
                                  {**data, "count": spell_small(crowd).capitalize()}))
            if taken:
                out.append(Moment(e.t, e.type, PRIORITY[e.type], e.team, e, data))
        elif e.type == "shot":
            out.append(Moment(e.t, shot_kind(e), PRIORITY["shot"], e.team, e, data))
        elif e.type == "replay":
            # Only worth a word when the broadcast cuts away right after something big.
            if any(b.type in BIG_MOMENTS and 0.0 <= e.t - b.t <= 12.0 for b in events):
                out.append(Moment(e.t + 0.6, "replay", PRIORITY["replay"], None, e, data))
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


def estimate_duration(text: str, words_per_second: float = 3.2) -> float:
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
        pad = gap_s if m.priority >= 40 else quiet_s
        start = _first_free_slot(placed, m.t, duration, pad, max_delay_s)
        if start is None:
            continue
        if clip_length is not None and start + duration > clip_length + 2.0:
            continue
        placed.append(Line(start, text, m.kind, m.priority, duration, m.speaker,
                           tone_of(m.kind)))
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


class LLMNarrator:
    """Words a moment with a language model, and refuses the result if it changes the facts.

    `complete` is any function that takes a prompt and returns text, so this works with a
    local model, a free API tier, or anything else, without this package depending on it.
    The model only chooses words. If its line mentions the wrong team, adds a number that is
    not in the facts, runs long, or the call fails, the template line is used instead.
    """

    def __init__(self, complete, team_names: Dict[int, str], max_words: int = 16,
                 style: str = "an excited but accurate British football commentator"):
        self.complete = complete
        self.team_names = team_names
        self.max_words = max_words
        self.style = style
        self.fallback = TemplateNarrator(team_names)
        self.history: List[str] = []
        self.fallback_text: Optional[str] = None
        self.accepted = self.rejected = 0

    def facts(self, moment: Moment) -> Dict[str, object]:
        facts: Dict[str, object] = {"moment": moment.kind.replace("_", " ").replace(".", ", ")}
        team = self.team_names.get(moment.team) if moment.team is not None else None
        if team:
            facts["team"] = team
        facts.update(moment.data)
        if moment.event is not None and moment.event.confidence < 0.6:
            facts["certainty"] = "not certain"
        return facts

    def prompt(self, moment: Moment, template: Optional[str] = None) -> str:
        """Ask for a rewording of the line the templates would have said.

        Rewording a sentence is something a small model does reliably; inventing commentary
        from a list of facts is not (it wandered off into nonsense when tried). The template
        line already carries exactly the facts that are true.
        """
        template = template or self.fallback.say(moment) or ""
        recent = " | ".join(self.history[-3:]) or "(nothing yet)"
        return (f"You are {self.style}.\n"
                f"Say the line below in your own words, as you would live on air. Keep its "
                f"meaning. Keep any team name exactly as written. Do not add player names, "
                f"scores or numbers. At most {self.max_words} words. Sound "
                f"{tone_of(moment.kind)}.\n"
                f"Examples:\n"
                f"Line: Corner for Rovers, over on the far right side.\n"
                f"On air: Rovers win a corner, out on the far right.\n"
                f"Line: That's the first real sight of goal for United.\n"
                f"On air: First proper look at goal for United, that.\n"
                f"Do not repeat these recent lines: {recent}\n"
                f"Line: {template}\nOn air:")

    def say(self, moment: Moment) -> Optional[str]:
        template = self.fallback.say(moment)
        self.fallback_text = template
        if template is None:
            return None  # the templates refuse moments that lack the facts to be said at all
        try:
            text = str(self.complete(self.prompt(moment, template))).strip()
            text = text.split("\n")[0].strip().strip('"').strip()
        except Exception:
            text = ""
        if self._faithful(text, moment):
            self.accepted += 1
        else:
            self.rejected += 1
            text = template
        self.history.append(text)
        return text

    def _faithful(self, text: str, moment: Moment) -> bool:
        if not text or len(text.split()) > self.max_words + 4:
            return False
        allowed = {str(v).lower() for v in moment.data.values()}
        if any(tok not in allowed for tok in _digits(text)):
            return False
        said = set(_words(text))
        if any(w in said and w not in allowed and w not in _words(self.fallback_text or "")
               for w in _NUMBER_WORDS):
            return False  # "their fifth corner" when the facts say it is the second
        own = self.team_names.get(moment.team) if moment.team is not None else None
        for name in self.team_names.values():
            if name != own and name.lower() in text.lower():
                return False
        template = self.fallback_text or ""
        if own and own in template and own.lower() not in text.lower():
            return False  # the team was dropped: the line no longer says who did it
        if any(junk in text.lower() for junk in ("line:", "on air", "rewrite", "words")):
            return False  # the model answered the instructions instead of commentating
        # Any name the model introduces is an invention: a capitalised word (other than the
        # one starting a sentence) must already be in the line it was asked to reword.
        known = set(_capitalised(template)) | set(self.team_names.values())
        if any(word not in known for word in _capitalised(text)):
            return False
        return True


_NUMBER_WORDS = ("two three four five six seven eight nine ten eleven twelve second third "
                 "fourth fifth sixth seventh eighth ninth tenth twice thrice hat-trick").split()


def _words(text: str) -> List[str]:
    import re

    return re.findall(r"[a-z-]+", text.lower())


def _capitalised(text: str) -> List[str]:
    """Capitalised words that are not simply starting a sentence."""
    import re

    out = []
    for sentence in re.split(r"[.!?]+\s*", text):
        words = re.findall(r"[A-Za-z'-]+", sentence)
        out += [w for w in words[1:] if w[0].isupper() and w not in ("I", "I'd", "I'm", "I've")]
    return out


def _digits(text: str) -> List[str]:
    import re

    return re.findall(r"\d+", text)


def ollama_complete(model: str, host: str = "http://localhost:11434"):
    """A `complete` function for a local Ollama server. Not exercised by the tests: it needs
    Ollama running on your machine (`ollama run <model>` once to download the model)."""
    import json
    import urllib.request

    def complete(prompt: str) -> str:
        body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0.8, "num_predict": 40}}).encode()
        request = urllib.request.Request(f"{host}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())["response"]

    return complete


def transformers_complete(model_dir: str, max_new_tokens: int = 28):
    """A `complete` function backed by a small local model run with Hugging Face transformers.
    Measured with Qwen3 models on a 4-core CPU; see CLAUDE.md for latency."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    def complete(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:  # Qwen3 thinks out loud unless told not to
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return complete


def make_complete(spec: str):
    """`ollama:NAME` for a local Ollama server, otherwise a path to a transformers model."""
    if spec.startswith("ollama:"):
        return ollama_complete(spec.split(":", 1)[1])
    return transformers_complete(spec)
