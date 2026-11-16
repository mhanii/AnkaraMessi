"""Shared data types. Perception produces pixels; everything downstream works in pitch metres."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

XY = Tuple[float, float]

BALL = "ball"
PLAYER = "player"
GOALKEEPER = "goalkeeper"
REFEREE = "referee"


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame, in image pixels."""

    cls: str
    conf: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    track_id: Optional[int] = None

    @property
    def foot_px(self) -> XY:
        """Ground contact point (bottom-centre of the box). Used for people."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def centre_px(self) -> XY:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class PlayerState:
    track_id: int
    role: str  # PLAYER or GOALKEEPER
    team: Optional[int]  # 0 or 1; None when unknown
    xy: XY  # pitch metres (feet)


@dataclass
class FrameState:
    """What is known about one analysed frame.

    `ball` is the ball's image position projected onto the ground plane. That is only the true
    position while the ball is on the ground: an airborne ball projects metres away from where
    it really is, usually beyond the far side of the pitch. Rules must not trust it mid-flight.
    """

    index: int
    t: float  # seconds from clip start
    ball: Optional[XY]  # None when not seen or when the frame has no calibration
    players: List[PlayerState] = field(default_factory=list)
    calibrated: bool = True  # False on close-ups / replays where the pitch cannot be located


@dataclass
class Event:
    """A football event. Times are seconds from clip start."""

    type: str
    t: float
    end_t: float
    team: Optional[int] = None
    player: Optional[int] = None
    receiver: Optional[int] = None
    xy: Optional[XY] = None
    distance_m: Optional[float] = None
    confidence: float = 1.0
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "t": round(self.t, 2),
            "end_t": round(self.end_t, 2),
            "team": self.team,
            "player": self.player,
            "receiver": self.receiver,
            "xy": None if self.xy is None else [round(self.xy[0], 1), round(self.xy[1], 1)],
            "distance_m": None if self.distance_m is None else round(self.distance_m, 1),
            "confidence": round(self.confidence, 2),
            "detail": self.detail,
        }
