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

from ..core import BALL, GOALKEEPER, PLAYER, REFEREE, Detection, FrameState, PlayerState
from ..pitch import (CENTRE, CORNERS, LENGTH, PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH,
                     PENALTY_SPOTS, WIDTH, goal_line_depth, inside)
from .calibrate import Homography, PitchCalibrator
from .detector import YoloDetector
from .teams import TeamClassifier, colour_name

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


def analyze(source: Path, detector: YoloDetector, calibrator: PitchCalibrator,
            analysis_fps: float = 12.5, default_fps: float = 25.0,
            max_seconds: Optional[float] = None, probe_every: int = 2,
            progress: Optional[Callable[[int, int], None]] = None) -> Analysis:
    src_fps, count, frames = open_source(source, default_fps)
    step = max(1, int(round(src_fps / analysis_fps)))
    fps = src_fps / step

    records: List[FrameRecord] = []
    teams = TeamClassifier()
    class_votes: Dict[int, Counter] = defaultdict(Counter)
    picker: Optional[_BallPicker] = None
    frame_size = (0, 0)
    timing = Counter()

    for index, frame in enumerate(frames):
        if max_seconds is not None and index / src_fps > max_seconds:
            break
        if index % step:
            continue
        k = len(records)
        frame_size = (frame.shape[1], frame.shape[0])
        if picker is None:
            picker = _BallPicker(frame.shape[1], fps)

        t0 = time.time()
        dets = detector.detect(frame)
        timing["detect"] += time.time() - t0
        t0 = time.time()
        hom = calibrator.calibrate(frame)
        timing["calibrate"] += time.time() - t0

        people = [d for d in dets if d.cls in (PLAYER, GOALKEEPER, REFEREE)
                  and d.track_id is not None]
        for d in people:
            class_votes[d.track_id][d.cls] += 1
            if d.cls != REFEREE:
                teams.observe(frame, d.track_id, d.box)

        balls = [d for d in dets if d.cls == BALL]
        if probe_every and k % probe_every == 0 and hom is not None \
                and not any(b.conf >= 0.3 for b in balls):
            t0 = time.time()
            found = detector.probe_ball(frame, hom.to_image(SET_PIECE_SPOTS))
            timing["probe"] += time.time() - t0
            balls += found
            dets = dets + found
        ball = picker.pick(k, balls)
        records.append(FrameRecord(index, index / src_fps, dets, hom, ball))
        if progress:
            progress(index, count)

    roles: Dict[int, str] = {}
    for tid, votes in class_votes.items():
        cls = votes.most_common(1)[0][0]
        roles[tid] = REFEREE if cls == REFEREE else GOALKEEPER if cls == GOALKEEPER else PLAYER

    split = teams.finalize([t for t, r in roles.items() if r != REFEREE])
    team_of, colours = dict(split.team_of), split.colours
    positions = _project(records, roles)
    for tid in _find_keepers(positions, roles, split):
        roles[tid] = GOALKEEPER
    _assign_keeper_teams(positions, roles, team_of)

    states: List[FrameState] = []
    for k, rec in enumerate(records):
        if rec.homography is None:
            states.append(FrameState(k, rec.t, None, [], calibrated=False))
            continue
        players = [PlayerState(tid, roles[tid], team_of.get(tid), xy)
                   for tid, xy in positions[k].items() if roles.get(tid) in (PLAYER, GOALKEEPER)]
        ball_xy = None
        if rec.ball is not None:
            x1, _, x2, y2 = rec.ball.box
            bx, by = rec.homography.to_pitch([((x1 + x2) / 2.0, y2)])[0]
            if np.isfinite(bx) and np.isfinite(by) and abs(bx) < 500 and abs(by) < 500:
                ball_xy = (float(bx), float(by))
        states.append(FrameState(k, rec.t, ball_xy, players, calibrated=True))

    n = max(1, len(records))
    stats = {
        "analysed_frames": len(records),
        "calibrated_fraction": sum(r.homography is not None for r in records) / n,
        "ball_seen_fraction": sum(r.ball is not None for r in records) / n,
        "seconds_per_frame": sum(timing.values()) / n,
        "keeper_tracks": sum(r == GOALKEEPER for r in roles.values()),
        **{f"seconds_{name}": value for name, value in timing.items()},
    }
    names = {k: colour_name(c) for k, c in colours.items()}
    if len(set(names.values())) < len(names):  # both teams got the same colour word
        names = {k: f"{v} ({'AB'[k]})" for k, v in names.items()}
    return Analysis(states, fps, src_fps, frame_size, records, team_of, roles, colours,
                    names, stats)


def _project(records: List[FrameRecord], roles: Dict[int, str],
             smooth: int = 5) -> List[Dict[int, Tuple[float, float]]]:
    """Project people onto the pitch, then median-smooth each track to damp calibration jitter."""
    raw: List[Dict[int, Tuple[float, float]]] = []
    for rec in records:
        here: Dict[int, Tuple[float, float]] = {}
        people = [d for d in rec.detections if d.track_id is not None and d.track_id in roles]
        if rec.homography is not None and people:
            pts = rec.homography.to_pitch([d.foot_px for d in people])
            for d, (x, y) in zip(people, pts):
                # People far outside the lines are coaches, substitutes or the crowd.
                if np.isfinite(x) and np.isfinite(y) and inside(float(x), float(y), 6.0):
                    here[d.track_id] = (float(x), float(y))
        raw.append(here)

    half = smooth // 2
    out: List[Dict[int, Tuple[float, float]]] = []
    for k, here in enumerate(raw):
        smoothed = {}
        for tid in here:
            window = [raw[j][tid] for j in range(max(0, k - half), min(len(raw), k + half + 1))
                      if tid in raw[j]]
            arr = np.asarray(window)
            smoothed[tid] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        out.append(smoothed)
    return out


def _find_keepers(positions, roles: Dict[int, str], split) -> List[int]:
    """Goalkeepers, with no model: a track whose shirt matches neither team and who stays
    inside a penalty area. The laws require the keeper's kit to differ from both teams, and
    nobody else lives in the box."""
    in_box: Counter = Counter()
    seen: Counter = Counter()
    for here in positions:
        for tid, (x, y) in here.items():
            seen[tid] += 1
            depth = min(x, LENGTH - x)
            if depth <= PENALTY_AREA_DEPTH + 1.5 and abs(y - WIDTH / 2.0) <= PENALTY_AREA_HALF_WIDTH:
                in_box[tid] += 1
    threshold = max(18.0, 2.5 * split.typical)
    return [tid for tid, off in split.off_colour.items()
            if roles.get(tid) == PLAYER and off >= threshold
            and seen[tid] >= 5 and in_box[tid] >= 0.7 * seen[tid]]


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
