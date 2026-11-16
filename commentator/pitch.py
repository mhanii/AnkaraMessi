"""Pitch geometry in metres.

x runs along the length, 0..105. Goal lines are at x=0 ('left') and x=105 ('right').
y runs across the width, 0..68. y=0 is the 'far' touchline (top of a main-camera picture),
y=68 the 'near' touchline (closest to the main camera).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

XY = Tuple[float, float]

LENGTH = 105.0
WIDTH = 68.0
MID_Y = WIDTH / 2.0
CENTRE: XY = (LENGTH / 2.0, MID_Y)
GOAL_HALF_WIDTH = 3.66
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16
PENALTY_SPOT_DEPTH = 11.0
CIRCLE_RADIUS = 9.15

PENALTY_SPOTS: Tuple[XY, XY] = ((PENALTY_SPOT_DEPTH, MID_Y), (LENGTH - PENALTY_SPOT_DEPTH, MID_Y))
CORNERS: Tuple[XY, ...] = ((0.0, 0.0), (LENGTH, 0.0), (0.0, WIDTH), (LENGTH, WIDTH))
GOAL_CENTRES = {"left": (0.0, MID_Y), "right": (LENGTH, MID_Y)}


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def inside(x: float, y: float, margin: float = 0.0) -> bool:
    """On the pitch, allowing `margin` metres of slack outside the lines (negative = inset)."""
    return -margin <= x <= LENGTH + margin and -margin <= y <= WIDTH + margin


def nearest_goal_side(x: float) -> str:
    return "left" if x < LENGTH / 2.0 else "right"


def goal_line_depth(x: float, side: str) -> float:
    """Distance from the given goal line, positive on the pitch."""
    return x if side == "left" else LENGTH - x


def in_penalty_area(x: float, y: float, side: str, margin: float = 0.0) -> bool:
    depth = goal_line_depth(x, side)
    return -margin <= depth <= PENALTY_AREA_DEPTH + margin and \
        abs(y - MID_Y) <= PENALTY_AREA_HALF_WIDTH + margin


def corner_nearest(xy: Sequence[float]) -> Tuple[XY, float]:
    """Nearest corner flag and its distance in metres."""
    best = min(CORNERS, key=lambda c: dist(xy, c))
    return best, dist(xy, best)


def corner_name(corner: XY) -> str:
    """'far-left', 'near-right', ... as a main-camera viewer would say it."""
    depth = "far" if corner[1] == 0.0 else "near"
    side = "left" if corner[0] == 0.0 else "right"
    return f"{depth}-{side}"


def touchline_name(y: float) -> str:
    return "far" if y < MID_Y else "near"


# Landmarks predicted by the Soccana 29-keypoint pitch model, index -> pitch position.
# Positions were confirmed empirically on SoccerNet GSR ground truth (median error < 1.6 m for
# every landmark seen); the six marked "by symmetry" were not visible in the measured frames.
SOCCANA_LANDMARKS: Tuple[XY, ...] = (
    (0.0, 0.0),                                   # 0  far-left corner
    (0.0, MID_Y - PENALTY_AREA_HALF_WIDTH),       # 1  left penalty area, far, on goal line
    (PENALTY_AREA_DEPTH, MID_Y - PENALTY_AREA_HALF_WIDTH),   # 2  left penalty area, far, front
    (0.0, MID_Y + PENALTY_AREA_HALF_WIDTH),       # 3  (by symmetry)
    (PENALTY_AREA_DEPTH, MID_Y + PENALTY_AREA_HALF_WIDTH),   # 4
    (0.0, MID_Y - GOAL_AREA_HALF_WIDTH),          # 5
    (GOAL_AREA_DEPTH, MID_Y - GOAL_AREA_HALF_WIDTH),         # 6
    (0.0, MID_Y + GOAL_AREA_HALF_WIDTH),          # 7
    (GOAL_AREA_DEPTH, MID_Y + GOAL_AREA_HALF_WIDTH),         # 8
    (0.0, WIDTH),                                 # 9  near-left corner (by symmetry)
    (PENALTY_SPOT_DEPTH + CIRCLE_RADIUS, MID_Y),  # 10 tip of the left penalty arc
    (LENGTH / 2.0, 0.0),                          # 11 halfway line, far
    (LENGTH / 2.0, WIDTH),                        # 12 halfway line, near (by symmetry)
    (LENGTH / 2.0, MID_Y - CIRCLE_RADIUS),        # 13 centre circle, far
    (LENGTH / 2.0, MID_Y + CIRCLE_RADIUS),        # 14 centre circle, near
    (LENGTH / 2.0, MID_Y),                        # 15 centre spot
    (LENGTH, 0.0),                                # 16 far-right corner
    (LENGTH, MID_Y - PENALTY_AREA_HALF_WIDTH),    # 17
    (LENGTH - PENALTY_AREA_DEPTH, MID_Y - PENALTY_AREA_HALF_WIDTH),  # 18
    (LENGTH, MID_Y + PENALTY_AREA_HALF_WIDTH),    # 19 (by symmetry)
    (LENGTH - PENALTY_AREA_DEPTH, MID_Y + PENALTY_AREA_HALF_WIDTH),  # 20
    (LENGTH, MID_Y - GOAL_AREA_HALF_WIDTH),       # 21
    (LENGTH - GOAL_AREA_DEPTH, MID_Y - GOAL_AREA_HALF_WIDTH),        # 22
    (LENGTH, MID_Y + GOAL_AREA_HALF_WIDTH),       # 23 (by symmetry)
    (LENGTH - GOAL_AREA_DEPTH, MID_Y + GOAL_AREA_HALF_WIDTH),        # 24
    (LENGTH, WIDTH),                              # 25 near-right corner (by symmetry)
    (LENGTH - PENALTY_SPOT_DEPTH - CIRCLE_RADIUS, MID_Y),            # 26 tip of the right arc
    (LENGTH / 2.0 - CIRCLE_RADIUS, MID_Y),        # 27 centre circle, left
    (LENGTH / 2.0 + CIRCLE_RADIUS, MID_Y),        # 28 centre circle, right
)


def pitch_polylines(circle_points: int = 48) -> List[List[XY]]:
    """Line markings as polylines in metres, for drawing overlays."""
    lines: List[List[XY]] = [
        [(0, 0), (LENGTH, 0), (LENGTH, WIDTH), (0, WIDTH), (0, 0)],
        [(LENGTH / 2, 0), (LENGTH / 2, WIDTH)],
    ]
    cx, cy = CENTRE
    lines.append([
        (cx + CIRCLE_RADIUS * math.cos(2 * math.pi * k / circle_points),
         cy + CIRCLE_RADIUS * math.sin(2 * math.pi * k / circle_points))
        for k in range(circle_points + 1)
    ])
    for sign, x0 in ((1.0, 0.0), (-1.0, LENGTH)):
        for depth, half in ((PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH),
                            (GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH)):
            lines.append([(x0, cy - half), (x0 + sign * depth, cy - half),
                          (x0 + sign * depth, cy + half), (x0, cy + half)])
    return lines
