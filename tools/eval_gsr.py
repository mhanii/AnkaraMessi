"""Score the event rules against SoccerNet GSR ground truth.

    python tools/eval_gsr.py LABELS_DIR [--verbose] [--cache FILE]

LABELS_DIR holds `SNGS-xxx.json` files (each a clip's `Labels-GameState.json`). Every clip names
one main action and when it happens. A clip counts as a hit when the engine reports the matching
event within the tolerance. Clips usually contain other, unlabelled actions too, so events
outside the labelled moment are listed, not counted as errors.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commentator.events import detect_events  # noqa: E402
from commentator.gsr import load_gamestate  # noqa: E402
from commentator.timeline import Timeline  # noqa: E402

EXPECTED = {
    "Corner": {"corner_kick"},
    "Kick-off": {"kick_off"},
    "Clearance": {"goal_kick"},  # SoccerNet calls goal kicks "Clearance"
    "Penalty": {"penalty"},
    "Direct free-kick": {"free_kick"},
    "Shots on target": {"shot", "goal", "save"},
    "Shots off target": {"shot", "goal"},
    "Goal": {"goal"},
}
RARE = ("penalty", "kick_off", "goal")  # one of these outside its own clip class is suspicious


def load_clips(labels_dir: Path, cache: Path | None):
    if cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    clips = [load_gamestate(p) for p in sorted(labels_dir.glob("*.json"))]
    if cache:
        cache.write_bytes(pickle.dumps(clips))
    return clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir", type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    clips = load_clips(args.labels_dir, args.cache)
    hits = collections.Counter()
    totals = collections.Counter()
    detected = collections.Counter()
    suspicious = []
    misses = []
    for clip in clips:
        events = detect_events(Timeline(clip.frames, clip.fps))
        for e in events:
            detected[e.type] += 1
        want = EXPECTED.get(clip.action)
        near = [e for e in events if abs(e.t - clip.action_t) <= args.tolerance]
        if want is not None:
            totals[clip.action] += 1
            if any(e.type in want for e in near):
                hits[clip.action] += 1
            else:
                misses.append((clip.name, clip.action, clip.action_t,
                               [f"{e.type}@{e.t:.1f}" for e in near]))
        for e in events:
            if e.type in RARE and not (want and e.type in want
                                       and abs(e.t - clip.action_t) <= args.tolerance):
                suspicious.append((clip.name, clip.action, f"{e.type}@{e.t:.1f}"))
        if args.verbose:
            line = ", ".join(f"{e.type}@{e.t:.1f}" for e in events
                             if e.type not in ("pass", "turnover"))
            n_pass = sum(e.type in ("pass", "long_pass") for e in events)
            print(f"{clip.name} {clip.action:17s} t={clip.action_t:5.1f} | passes={n_pass:2d} | {line}")

    print(f"\n{len(clips)} clips from {args.labels_dir}")
    print("labelled action       hits / clips")
    for action in EXPECTED:
        if totals[action]:
            print(f"  {action:18s} {hits[action]:2d} / {totals[action]:2d}")
    scored = sum(totals.values())
    print(f"  {'ALL':18s} {sum(hits.values()):2d} / {scored:2d}")
    print("\nmissed:")
    for m in misses:
        print(f"  {m[0]} {m[1]} at {m[2]:.1f}s; nearby events: {m[3]}")
    print("\nrare events outside their own labelled clip (check by hand):")
    for s in suspicious:
        print(f"  {s[0]} ({s[1]}): {s[2]}")
    print("\nall detected events:", dict(detected))


if __name__ == "__main__":
    main()
