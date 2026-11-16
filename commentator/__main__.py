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
    run.add_argument("--max-seconds", type=float, default=None)
    run.add_argument("--no-audio", action="store_true")
    run.add_argument("--no-video", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        from .run import run as run_clip

        def progress(index: int, total: int) -> None:
            if index % 100 == 0:
                print(f"  frame {index}" + (f" / {total}" if total > 0 else ""), flush=True)

        teams = [t.strip() for t in args.teams.split(",")] if args.teams else None
        summary = run_clip(args.source, args.out, args.models, args.fps, teams,
                           audio=not args.no_audio, video=not args.no_video,
                           max_seconds=args.max_seconds, voice=args.voice, progress=progress)
        print(json.dumps(summary, indent=1))
        print((args.out / "commentary.txt").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
