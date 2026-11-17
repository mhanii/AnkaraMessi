"""Command line.

    python -m commentator run CLIP --out out/clip        # video file or folder of frames
    python -m commentator run CLIP --out out/clip --teams "Thun,St. Gallen" --no-audio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="commentator", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse a clip and produce a commentated video")
    run.add_argument("source", type=Path, help="video file, or a folder of numbered frames")
    run.add_argument("--out", type=Path, required=True, help="output folder")
    run.add_argument("--models", type=Path, default=Path("models"))
    run.add_argument("--fps", type=float, default=12.5, help="frames analysed per second")
    run.add_argument("--teams", type=str, default=None,
                     help='two team names, comma separated (default: shirt colours)')
    run.add_argument("--voice", type=str, default="bm_george")
    run.add_argument("--llm", type=str, default=None,
                     help="word the lines with a local Ollama model (a name from `ollama list`)")
    run.add_argument("--max-seconds", type=float, default=None)
    run.add_argument("--no-audio", action="store_true")
    run.add_argument("--no-video", action="store_true")

    live = sub.add_parser("live", help="commentate as if on a live feed, a few seconds behind")
    live.add_argument("source", type=Path)
    live.add_argument("--out", type=Path, required=True)
    live.add_argument("--models", type=Path, default=Path("models"))
    live.add_argument("--delay", type=float, default=4.0,
                      help="seconds the picture runs behind the camera")
    live.add_argument("--fps", type=float, default=12.5)
    live.add_argument("--replay", type=Path, default=None,
                      help="analysis.pkl from a previous run, instead of running the vision models")
    live.add_argument("--llm", type=str, default=None,
                      help="language model for the analyst: ollama:NAME or a local model folder")
    live.add_argument("--teams", type=str, default=None)
    live.add_argument("--pace", choices=("virtual", "real"), default="virtual",
                      help="real = also keep to the wall clock and report falling behind")
    live.add_argument("--device", type=str, default=None, help="e.g. cuda:0")
    live.add_argument("--no-video", action="store_true")
    live.add_argument("--density", choices=("low", "normal", "high"), default="high",
                      help="how talkative the commentary is")
    live.add_argument("--overlay", choices=("none", "map", "full"), default="none",
                      help="none = clean picture; map = mini-map; full = also mark players")

    bench = sub.add_parser("bench", help="can this machine run it live?")
    bench.add_argument("source", type=Path)
    bench.add_argument("--models", type=Path, default=Path("models"))
    bench.add_argument("--device", type=str, default=None, help="e.g. cuda:0")
    bench.add_argument("--frames", type=int, default=30)

    args = parser.parse_args(argv)
    if args.command == "live":
        from .live_run import run_live

        teams = [t.strip() for t in args.teams.split(",")] if args.teams else None
        report = run_live(args.source, args.out, args.models, args.delay, args.fps, args.replay,
                          args.llm, teams, args.pace, args.device, video=not args.no_video,
                          density=args.density, overlay=args.overlay)
        print(json.dumps(report, indent=1))
        print((args.out / "commentary_live.txt").read_text())
        return 0
    if args.command == "bench":
        from .live_run import bench as run_bench

        print(json.dumps(run_bench(args.source, args.models, args.device, args.frames), indent=1))
        return 0
    if args.command == "run":
        from .run import run as run_clip

        def progress(index: int, total: int) -> None:
            if index % 100 == 0:
                print(f"  frame {index}" + (f" / {total}" if total > 0 else ""), flush=True)

        teams = [t.strip() for t in args.teams.split(",")] if args.teams else None
        summary = run_clip(args.source, args.out, args.models, args.fps, teams,
                           audio=not args.no_audio, video=not args.no_video,
                           max_seconds=args.max_seconds, voice=args.voice, llm=args.llm,
                           progress=progress)
        print(json.dumps(summary, indent=1))
        print((args.out / "commentary.txt").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
