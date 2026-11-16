"""Scripted scenes with known answers, for testing the event rules without any video.

The scenes imitate what real projected data looks like, including its worst habit: a ball in
the air does not project to where it is. `Scene.fly` therefore sends the ball on a wildly
wrong path between take-off and landing, exactly as SoccerNet ground truth does.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .core import GOALKEEPER, PLAYER, FrameState, PlayerState

XY = Tuple[float, float]


class Scene:
    def __init__(self, players: Dict[int, Tuple[Optional[int], str, XY]], ball: XY,
                 fps: float = 25.0):
        self.fps = fps
        self.players = {pid: [team, role, tuple(xy)] for pid, (team, role, xy) in players.items()}
        self.ball: XY = tuple(ball)
        self._frames: List[FrameState] = []

    # -- building blocks ---------------------------------------------------------------
    def _emit(self, ball: Optional[XY]) -> None:
        k = len(self._frames)
        people = [PlayerState(pid, role, team, xy) for pid, (team, role, xy) in self.players.items()]
        self._frames.append(FrameState(k, k / self.fps, ball, people))

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    def hold(self, seconds: float, at: Optional[XY] = None) -> "Scene":
        """Ball lies still on the ground."""
        if at is not None:
            self.ball = tuple(at)
        for _ in range(self._steps(seconds)):
            self._emit(self.ball)
        return self

    def roll(self, seconds: float, to: XY, carrier: Optional[int] = None) -> "Scene":
        """Ball moves along the ground. With `carrier`, that player moves with it (a dribble)."""
        n = self._steps(seconds)
        x0, y0 = self.ball
        offset = None
        if carrier is not None:
            px, py = self.players[carrier][2]
            offset = (px - x0, py - y0)
        for i in range(1, n + 1):
            u = i / n
            self.ball = (x0 + (to[0] - x0) * u, y0 + (to[1] - y0) * u)
            if offset is not None:
                self.players[carrier][2] = (self.ball[0] + offset[0], self.ball[1] + offset[1])
            self._emit(self.ball)
        return self

    def fly(self, seconds: float, to: XY, error: XY = (8.0, -30.0)) -> "Scene":
        """Ball travels through the air. Its projected position swings far off the true path
        (by up to `error` metres mid-flight) and only returns to the truth on landing."""
        n = self._steps(seconds)
        x0, y0 = self.ball
        for i in range(1, n + 1):
            u = i / n
            bulge = math.sin(math.pi * u)
            self.ball = (x0 + (to[0] - x0) * u, y0 + (to[1] - y0) * u)
            shown = self.ball if i == n else (self.ball[0] + error[0] * bulge,
                                              self.ball[1] + error[1] * bulge)
            self._emit(shown)
        return self

    def hide(self, seconds: float) -> "Scene":
        """Ball not detected (occluded, or the detector missed it)."""
        for _ in range(self._steps(seconds)):
            self._emit(None)
        return self

    def put(self, pid: int, xy: XY) -> "Scene":
        self.players[pid][2] = tuple(xy)
        return self

    def frames(self) -> List[FrameState]:
        return list(self._frames)


def squad(team: int, first_id: int, positions) -> Dict[int, Tuple[int, str, XY]]:
    return {first_id + i: (team, PLAYER, xy) for i, xy in enumerate(positions)}


def keeper(team: int, pid: int, xy: XY) -> Dict[int, Tuple[int, str, XY]]:
    return {pid: (team, GOALKEEPER, xy)}
