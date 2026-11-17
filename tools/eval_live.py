"""How live can the commentary be? Replay SoccerNet GSR ground truth through the live engine.

    python tools/eval_live.py CACHE.pkl [--delay 4]

CACHE.pkl is the clip cache written by tools/eval_gsr.py. Each clip is fed frame by frame; the
rules only ever see the past. Reports, per kind of moment, how long after it happened the
system knew, and how many lines started in sync with a picture delayed by `--delay` seconds.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commentator.commentary import TemplateNarrator  # noqa: E402
from commentator.live import Analyst, LiveCommentator, LiveDirector  # noqa: E402
from commentator.speech import SAMPLE_RATE, Booth  # noqa: E402

EXPECTED = {"Corner": {"corner_kick"}, "Kick-off": {"kick_off"}, "Clearance": {"goal_kick"},
            "Penalty": {"penalty"}, "Direct free-kick": {"free_kick"},
            "Shots on target": {"shot", "goal", "save"}, "Shots off target": {"shot", "goal"},
            "Goal": {"goal"}}


class FakeVoice:
    def synthesize(self, text, voice=None, tone="calm"):
        return np.zeros(int(SAMPLE_RATE * max(0.6, len(text.split()) / 3.0)), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--stride", type=int, default=2, help="use every Nth frame (2 = 12.5 fps)")
    args = ap.parse_args()

    clips = pickle.loads(args.cache.read_bytes())
    names = {0: "White", 1: "Red"}
    knew = collections.defaultdict(list)
    late = collections.defaultdict(list)
    hits, totals = collections.Counter(), collections.Counter()
    cancelled = aired_total = dropped_total = 0
    rule_seconds, rule_runs = 0.0, 0
    for clip in clips:
        frames = clip.frames[::args.stride]
        fps = clip.fps / args.stride
        director = LiveDirector(TemplateNarrator(names), Booth(FakeVoice()),
                                TemplateNarrator(names, seed=3), Analyst(), delay_s=args.delay)
        live = LiveCommentator(director)
        now = 0.0
        for k, frame in enumerate(frames):
            now = frame.t
            if live.due_for_rules(now):
                live.update(frames[:k + 1], fps, now)
            live.speak(now)
        end = now
        while now < end + args.delay + 5.0:
            now += 1.0 / fps
            live.speak(now)
        rule_seconds += live.seconds_in_rules
        rule_runs += live.ticks
        aired_total += len(director.aired)
        dropped_total += len(director.dropped)
        for s in director.aired:
            if s.speaker != "lead":
                continue
            base = s.kind.split(".")[0] if not s.kind.endswith(".setup") else s.kind
            knew[base].append(max(0.0, s.decided_t - s.event_t))
            late[base].append(s.late_by)
            cancelled += s.kind == "goal.cancelled"
        want = EXPECTED.get(clip.action)
        if want:
            totals[clip.action] += 1
            said = {s.kind.split(".")[0] for s in director.aired
                    if abs(s.event_t - clip.action_t) <= 3.0 and not s.kind.endswith(".setup")}
            hits[clip.action] += bool(said & want)

    print(f"{len(clips)} clips, picture delayed {args.delay:g} s, "
          f"analysed at {clips[0].fps / args.stride:g} fps")
    print("\nlabelled action called on air (live)   hits / clips")
    for action in EXPECTED:
        if totals[action]:
            print(f"  {action:18s} {hits[action]:2d} / {totals[action]:2d}")
    print(f"  {'ALL':18s} {sum(hits.values()):2d} / {sum(totals.values()):2d}")
    print("\nmoment               lines   known after (median / worst)   in sync with picture")
    for kind in sorted(knew, key=lambda k: -len(knew[k])):
        k, l = np.array(knew[kind]), np.array(late[kind])
        print(f"  {kind:18s} {len(k):5d}   {np.median(k):5.2f} s / {k.max():5.2f} s"
              f"            {int((l <= 0.3).sum()):3d} of {len(l)}")
    everything = np.concatenate([np.array(v) for v in late.values()]) if late else np.zeros(0)
    print(f"\nlead lines in sync: {int((everything <= 0.3).sum())} of {len(everything)}")
    print(f"lines aired: {aired_total}, dropped as too late: {dropped_total}, "
          f"goal calls taken back on air: {cancelled}")
    print(f"rule engine: {1000 * rule_seconds / max(1, rule_runs):.0f} ms per run, "
          f"run every 0.5 s")


if __name__ == "__main__":
    main()
