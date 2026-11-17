"""Video (or a folder of frames) in, a pitch-metre timeline out.

Per analysed frame: detect and track people and the ball, locate the pitch, and remember shirt
colours. After the last frame: decide roles and teams per track, then project everything onto
the pitch.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ..core import (BALL, GOALKEEPER, PLAYER, REFEREE, Detection, FrameState, PlayerState,
                    holds_ball)
from ..pitch import (CENTRE, CORNERS, LENGTH, PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH,
                     PENALTY_SPOTS, WIDTH, goal_line_depth, inside)
from .calibrate import Homography, PitchCalibrator
from .detector import YoloDetector
from .teams import TeamClassifier, colour_name, shirt_colour

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Where a waiting ball is small and far away, and where the rules most need to see it.
SET_PIECE_SPOTS = list(CORNERS) + list(PENALTY_SPOTS) + [CENTRE]


@dataclass
class FrameRecord:
    index: int  # frame number in the source
    t: float
    detections: List[Detection]
    homography: Optional[Homography]
    ball: Optional[Detection] = None  # the ball detection accepted for this frame
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # people, metres
    ball_xy: Optional[Tuple[float, float]] = None  # ball projected onto the pitch
    colours: Dict[int, np.ndarray] = field(default_factory=dict)  # shirt colour per person


@dataclass
class Analysis:
    frames: List[FrameState]
    fps: float  # analysed frames per second
    source_fps: float
    frame_size: Tuple[int, int]
    records: List[FrameRecord]
    team_of: Dict[int, int]
    roles: Dict[int, str]
    team_colours: Dict[int, np.ndarray]  # BGR
    team_names: Dict[int, str]
    stats: Dict[str, float] = field(default_factory=dict)


def open_source(source: Path, default_fps: float = 25.0) -> Tuple[float, int, Iterator[np.ndarray]]:
    """Returns (fps, frame count or -1, frame iterator) for a video file or a folder of images."""
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no images in {source}")
        return default_fps, len(files), (cv2.imread(str(p)) for p in files)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)

    def frames() -> Iterator[np.ndarray]:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()

    return float(fps), count, frames()


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class _BallPicker:
    """Chooses one ball detection per frame, preferring continuity over raw confidence."""

    def __init__(self, frame_width: int, fps: float):
        self.scale = frame_width / 1920.0
        self.fps = fps
        self.last_xy: Optional[Tuple[float, float]] = None
        self.last_k = -10 ** 6

    def pick(self, k: int, candidates: List[Detection]) -> Optional[Detection]:
        if not candidates:
            return None
        gap = k - self.last_k
        chosen: Optional[Detection] = None
        if self.last_xy is not None and gap / self.fps <= 0.8:
            gate = (90.0 * gap * (12.5 / self.fps) + 60.0) * self.scale
            near = [c for c in candidates
                    if np.hypot(c.centre_px[0] - self.last_xy[0],
                                c.centre_px[1] - self.last_xy[1]) <= gate]
            if near:
                chosen = max(near, key=lambda c: c.conf)
        if chosen is None:
            top = max(candidates, key=lambda c: c.conf)
            if top.conf >= 0.3 or self.last_xy is None:
                chosen = top
        if chosen is not None:
            self.last_xy, self.last_k = chosen.centre_px, k
        return chosen


class Perception:
    """Frame-by-frame perception.

    `step` takes one frame and records what was seen. `snapshot` turns everything recorded so
    far (or just the last N seconds) into a pitch-metre game state, deciding roles and teams
    from all the evidence up to now. That makes the same code serve a finished clip (one
    snapshot at the end) and a live stream (a snapshot every half second).
    """

    def __init__(self, detector: Optional[YoloDetector], calibrator: Optional[PitchCalibrator],
                 fps: float, source_fps: Optional[float] = None, probe_every: int = 1):
        self.detector, self.calibrator = detector, calibrator
        self.fps = fps
        self.source_fps = source_fps or fps
        self.probe_every = probe_every
        self.records: List[FrameRecord] = []
        self.teams = TeamClassifier()
        self.class_votes: Dict[int, Counter] = defaultdict(Counter)
        self.picker: Optional[_BallPicker] = None
        self.frame_size = (0, 0)
        self.timing: Counter = Counter()
        self._team_colours: Dict[int, np.ndarray] = {}

    def step(self, frame: np.ndarray, index: int, t: float) -> FrameRecord:
        k = len(self.records)
        self.frame_size = (frame.shape[1], frame.shape[0])
        if self.picker is None:
            self.picker = _BallPicker(frame.shape[1], self.fps)

        t0 = time.time()
        dets = self.detector.detect(frame)
        self.timing["detect"] += time.time() - t0
        t0 = time.time()
        hom = self.calibrator.calibrate(frame)
        self.timing["calibrate"] += time.time() - t0

        people = [d for d in dets if d.cls in (PLAYER, GOALKEEPER, REFEREE)
                  and d.track_id is not None]
        colours: Dict[int, np.ndarray] = {}
        for d in people:
            if d.cls != REFEREE:
                colour = shirt_colour(frame, d.box)
                if colour is not None:
                    colours[d.track_id] = colour

        balls = [d for d in dets if d.cls == BALL]
        if self.probe_every and k % self.probe_every == 0 and hom is not None \
                and not any(b.conf >= 0.3 for b in balls):
            t0 = time.time()
            found = self.detector.probe_ball(frame, hom.to_image(SET_PIECE_SPOTS))
            self.timing["probe"] += time.time() - t0
            balls += found
            dets = dets + found
        ball = self.picker.pick(k, balls)

        record = FrameRecord(index, t, dets, hom, ball, colours=colours)
        if hom is not None:
            if people:
                for d, (x, y) in zip(people, hom.to_pitch([d.foot_px for d in people])):
                    # People far outside the lines are coaches, substitutes or the crowd.
                    if np.isfinite(x) and np.isfinite(y) and inside(float(x), float(y), 6.0):
                        record.positions[d.track_id] = (float(x), float(y))
            if ball is not None:
                x1, _, x2, y2 = ball.box
                bx, by = hom.to_pitch([((x1 + x2) / 2.0, y2)])[0]
                if np.isfinite(bx) and np.isfinite(by) and abs(bx) < 500 and abs(by) < 500:
                    record.ball_xy = (float(bx), float(by))
        self.ingest(record)
        return record

    def ingest(self, record: FrameRecord) -> None:
        """Take in one frame's observations. `step` calls this; so does a replay of a saved
        analysis, which lets the live pipeline be exercised without re-running the models."""
        for d in record.detections:
            if d.track_id is not None and d.cls in (PLAYER, GOALKEEPER, REFEREE):
                self.class_votes[d.track_id][d.cls] += 1
        for tid, colour in record.colours.items():
            self.teams.add(tid, colour)
        self.records.append(record)

    def snapshot(self, last_seconds: Optional[float] = None) -> Analysis:
        records = self.records
        if last_seconds is not None and records:
            start = records[-1].t - last_seconds
            records = [r for r in records if r.t >= start]

        roles: Dict[int, str] = {}
        for tid, votes in self.class_votes.items():
            cls = votes.most_common(1)[0][0]
            roles[tid] = REFEREE if cls == REFEREE else GOALKEEPER if cls == GOALKEEPER \
                else PLAYER

        split = self.teams.finalize([t for t, r in roles.items() if r != REFEREE])
        team_of, colours = self._stable_labels(split)
        positions = _smooth_positions(records, roles)
        for tid in _find_keepers(positions, roles, split):
            roles[tid] = GOALKEEPER
        _assign_keeper_teams(positions, roles, team_of)

        states: List[FrameState] = []
        for k, rec in enumerate(records):
            if rec.homography is None:
                states.append(FrameState(k, rec.t, None, [], calibrated=False))
                continue
            players = [PlayerState(tid, roles[tid], team_of.get(tid), xy)
                       for tid, xy in positions[k].items()
                       if roles.get(tid) in (PLAYER, GOALKEEPER)]
            holder = None
            if rec.ball is not None:
                holders = [d.track_id for d in rec.detections
                           if d.track_id is not None
                           and roles.get(d.track_id) in (PLAYER, GOALKEEPER)
                           and holds_ball(d.box, rec.ball.centre_px)]
                holder = holders[0] if len(holders) == 1 else None
            states.append(FrameState(k, rec.t, rec.ball_xy, players, calibrated=True,
                                     ball_holder=holder))

        n = max(1, len(records))
        total = max(1, len(self.records))
        stats = {
            "analysed_frames": len(records),
            "calibrated_fraction": sum(r.homography is not None for r in records) / n,
            "ball_seen_fraction": sum(r.ball is not None for r in records) / n,
            "seconds_per_frame": sum(self.timing.values()) / total,
            "keeper_tracks": sum(r == GOALKEEPER for r in roles.values()),
            **{f"seconds_{name}": value for name, value in self.timing.items()},
        }
        names = {k: colour_name(c) for k, c in colours.items()}
        if len(set(names.values())) < len(names):  # both teams got the same colour word
            names = {k: f"{v} ({'AB'[k]})" for k, v in names.items()}
        return Analysis(states, self.fps, self.source_fps, self.frame_size, records, team_of,
                        roles, colours, names, stats)

    def _stable_labels(self, split) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
        """Keep "team 0" meaning the same team from one snapshot to the next. Clustering can
        hand the two labels out in either order; match them to the colours seen last time."""
        team_of, colours = dict(split.team_of), dict(split.colours)
        if len(colours) == 2 and len(self._team_colours) == 2:
            keep = sum(float(np.linalg.norm(colours[k] - self._team_colours[k])) for k in (0, 1))
            swap = sum(float(np.linalg.norm(colours[k] - self._team_colours[1 - k]))
                       for k in (0, 1))
            if swap < keep:
                team_of = {t: 1 - v for t, v in team_of.items()}
                colours = {0: colours[1], 1: colours[0]}
        if len(colours) == 2:
            self._team_colours = colours
        return team_of, colours


def analyze(source: Path, detector: YoloDetector, calibrator: PitchCalibrator,
            analysis_fps: float = 12.5, default_fps: float = 25.0,
            max_seconds: Optional[float] = None, probe_every: int = 1,
            progress: Optional[Callable[[int, int], None]] = None) -> Analysis:
    """Analyse a whole clip: every sampled frame through `Perception`, then one snapshot."""
    src_fps, count, frames = open_source(source, default_fps)
    step = max(1, int(round(src_fps / analysis_fps)))
    perception = Perception(detector, calibrator, src_fps / step, src_fps, probe_every)
    for index, frame in enumerate(frames):
        if max_seconds is not None and index / src_fps > max_seconds:
            break
        if index % step:
            continue
        perception.step(frame, index, index / src_fps)
        if progress:
            progress(index, count)
    return perception.snapshot()


def _smooth_positions(records: List[FrameRecord], roles: Dict[int, str],
                      smooth: int = 5) -> List[Dict[int, Tuple[float, float]]]:
    """Median-smooth each track's pitch position to damp calibration jitter."""
    half = smooth // 2
    out: List[Dict[int, Tuple[float, float]]] = []
    for k, rec in enumerate(records):
        smoothed = {}
        for tid in rec.positions:
            if tid not in roles:
                continue
            window = [records[j].positions[tid]
                      for j in range(max(0, k - half), min(len(records), k + half + 1))
                      if tid in records[j].positions]
            arr = np.asarray(window)
            smoothed[tid] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        out.append(smoothed)
    return out


def _find_keepers(positions, roles: Dict[int, str], split) -> List[int]:
    """Goalkeepers, with no model. A keeper wears a kit that matches neither team (the laws
    require it), stays close to one goal, and is the deepest player at that end.

    All three are needed. Colour alone picks up short, noisy track fragments; "is in the box"
    alone picks up a penalty taker waiting on the spot.
    """
    seen: Counter = Counter()
    deepest: Counter = Counter()
    depths: Dict[int, List[float]] = defaultdict(list)
    for here in positions:
        if not here:
            continue
        for side in ("left", "right"):
            end = {tid: goal_line_depth(x, side) for tid, (x, _) in here.items()
                   if goal_line_depth(x, side) <= LENGTH / 2}
            if end:
                deepest[min(end, key=end.get)] += 1
        for tid, (x, y) in here.items():
            seen[tid] += 1
            if abs(y - WIDTH / 2.0) <= PENALTY_AREA_HALF_WIDTH:
                depths[tid].append(min(x, LENGTH - x))
    threshold = max(18.0, 2.5 * split.typical)
    keepers = []
    for tid, off in split.off_colour.items():
        if roles.get(tid) != PLAYER or off < threshold or seen[tid] < 15:
            continue
        near_goal = len(depths[tid]) >= 0.8 * seen[tid] and float(np.median(depths[tid])) <= 11.0
        if near_goal and deepest[tid] >= 0.7 * seen[tid]:
            keepers.append(tid)
    return keepers


def _assign_keeper_teams(positions, roles: Dict[int, str], team_of: Dict[int, int]) -> None:
    """A goalkeeper wears a third colour, so give each one the team that sits deeper, on
    average, towards the goal they are standing in."""
    depth_sum = {side: Counter() for side in ("left", "right")}
    depth_n = Counter()
    keeper_x: Dict[int, List[float]] = defaultdict(list)
    for here in positions:
        for tid, (x, _) in here.items():
            if roles.get(tid) == GOALKEEPER:
                keeper_x[tid].append(x)
            elif tid in team_of:
                team = team_of[tid]
                depth_n[team] += 1
                for side in ("left", "right"):
                    depth_sum[side][team] += goal_line_depth(x, side)
    if depth_n[0] == 0 or depth_n[1] == 0:
        return
    for tid, xs in keeper_x.items():
        side = "left" if float(np.median(xs)) < LENGTH / 2.0 else "right"
        mean_depth = {team: depth_sum[side][team] / depth_n[team] for team in (0, 1)}
        team_of[tid] = min(mean_depth, key=mean_depth.get)
