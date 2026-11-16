"""Loader for SoccerNet Game State Reconstruction labels (`Labels-GameState.json`).

Each GSR sequence is a 30-second broadcast clip with hand-checked pitch positions for every
player and the ball, plus the clip's main action and its time. We use it as real ground truth:
event rules are developed and scored against it without any perception error in the way.

GSR pitch coordinates have their origin at the centre spot; ours start at the far-left corner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .core import GOALKEEPER, PLAYER, FrameState, PlayerState
from .pitch import LENGTH, WIDTH

_TEAM = {"left": 0, "right": 1}
_ROLE = {"player": PLAYER, "goalkeeper": GOALKEEPER}


@dataclass
class GsrClip:
    name: str
    action: str  # SoccerNet action class of the clip's main action, e.g. "Corner"
    action_t: float  # seconds from clip start
    fps: float
    frames: List[FrameState]


def load_gamestate(path: Path) -> GsrClip:
    data = json.loads(Path(path).read_text())
    info = data["info"]
    fps = float(info.get("frame_rate", 25))
    index_of = {im["image_id"]: k for k, im in enumerate(data["images"])}
    frames = [FrameState(index=k, t=k / fps, ball=None, players=[], calibrated=False)
              for k in range(len(data["images"]))]

    for ann in data["annotations"]:
        pitch = ann.get("bbox_pitch")
        if not pitch or ann.get("category_id") not in (1, 2, 4):
            continue
        frame = frames[index_of[ann["image_id"]]]
        frame.calibrated = True
        xy = (pitch["x_bottom_middle"] + LENGTH / 2.0, pitch["y_bottom_middle"] + WIDTH / 2.0)
        if ann["category_id"] == 4:
            frame.ball = xy
            continue
        attrs = ann.get("attributes") or {}
        role = _ROLE.get(attrs.get("role"))
        if role is None:
            continue
        team: Optional[int] = _TEAM.get(attrs.get("team"))
        frame.players.append(PlayerState(int(ann["track_id"]), role, team, xy))

    action_t = (int(info["action_position"]) - int(info["clip_start"])) / 1000.0
    return GsrClip(info["name"], info.get("action_class", ""), action_t, fps, frames)
