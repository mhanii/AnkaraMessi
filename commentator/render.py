"""Draw what the system sees and says onto the video, so a person can check it at a glance.

On every frame: a coloured ring under each tracked player (team colour), a marker on the ball,
a top-down mini-map of where everybody is on the pitch, the current event as a tag, and the
commentary line as a caption.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .commentary import Line
from .core import BALL, GOALKEEPER, REFEREE, Event, FrameState
from .perception.pipeline import Analysis, open_source
from .pitch import LENGTH, WIDTH, pitch_polylines

FONT = cv2.FONT_HERSHEY_SIMPLEX
EVENT_LABELS = {
    "goal": "GOAL", "shot": "SHOT", "save": "SAVE", "penalty": "PENALTY",
    "corner_kick": "CORNER KICK", "free_kick": "FREE KICK", "kick_off": "KICK-OFF",
    "goal_kick": "GOAL KICK", "hand_throw": "KEEPER THROW", "throw_in": "THROW-IN",
    "long_throw": "LONG THROW", "long_pass": "LONG PASS", "pass": "PASS",
    "turnover": "TURNOVER",
}
_UNKNOWN = (200, 200, 200)


def _colour(team: Optional[int], colours: Dict[int, np.ndarray]) -> Tuple[int, int, int]:
    if team is None or team not in colours:
        return _UNKNOWN
    return tuple(int(v) for v in colours[team])


def draw_minimap(state: Optional[FrameState], colours: Dict[int, np.ndarray],
                 width: int = 360) -> np.ndarray:
    scale = width / (LENGTH + 6.0)
    height = int(round((WIDTH + 6.0) * scale))
    img = np.full((height, width, 3), (40, 110, 50), np.uint8)

    def px(x: float, y: float) -> Tuple[int, int]:
        return int(round((x + 3.0) * scale)), int(round((y + 3.0) * scale))

    for line in pitch_polylines():
        pts = np.array([px(x, y) for x, y in line], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (235, 235, 235), 1, cv2.LINE_AA)
    if state is None or not state.calibrated:
        cv2.putText(img, "no pitch view", (10, height // 2), FONT, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return img
    for p in state.players:
        c = _colour(p.team, colours)
        cv2.circle(img, px(*p.xy), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, px(*p.xy), 4, (0, 220, 255) if p.role == GOALKEEPER else c, -1,
                   cv2.LINE_AA)
    if state.ball is not None and -3 <= state.ball[0] <= LENGTH + 3 \
            and -3 <= state.ball[1] <= WIDTH + 3:
        cv2.circle(img, px(*state.ball), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, px(*state.ball), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def _caption(frame: np.ndarray, text: str, speaker: str = "lead") -> None:
    h, w = frame.shape[:2]
    scale = w / 1280.0 * 0.85
    thick = max(1, int(round(2 * w / 1280.0)))
    words, rows, row = text.split(), [], ""
    for word in words:
        trial = (row + " " + word).strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] > 0.8 * w and row:
            rows.append(row)
            row = word
        else:
            row = trial
    rows.append(row)
    line_h = int(cv2.getTextSize("Ag", FONT, scale, thick)[0][1] * 1.9)
    box_h = line_h * len(rows) + int(0.6 * line_h)
    y0 = h - box_h - int(0.04 * h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (int(0.07 * w), y0), (int(0.93 * w), y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    analyst = speaker == "analyst"
    colour = (150, 235, 255) if analyst else (255, 255, 255)  # the analyst speaks in amber
    label = "ANALYST" if analyst else "COMMENTARY"
    cv2.putText(frame, label, (int(0.07 * w) + int(0.01 * w), y0 - int(0.012 * w)), FONT,
                scale * 0.55, colour, max(1, thick - 1), cv2.LINE_AA)
    for i, r in enumerate(rows):
        tw = cv2.getTextSize(r, FONT, scale, thick)[0][0]
        y = y0 + int(0.3 * line_h) + (i + 1) * line_h - int(0.35 * line_h)
        cv2.putText(frame, r, ((w - tw) // 2, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _tag(frame: np.ndarray, text: str) -> None:
    w = frame.shape[1]
    scale = w / 1280.0 * 0.8
    thick = max(1, int(round(2 * w / 1280.0)))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y, pad = int(0.025 * w), int(0.045 * w), int(0.012 * w)
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (30, 30, 200), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _badge(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    scale = w / 1280.0 * 0.5
    thick = max(1, int(round(w / 1280.0)))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(0.025 * w), h - int(0.02 * w)
    cv2.circle(frame, (x + th // 2, y - th // 2), th // 2, (40, 40, 230), -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + int(1.6 * th), y), FONT, scale, (255, 255, 255), thick,
                cv2.LINE_AA)


def render(source: Path, analysis: Analysis, events: Sequence[Event], lines: Sequence[Line],
           out_path: Path, audio_path: Optional[Path] = None, out_width: int = 1280,
           default_fps: float = 25.0, badge: Optional[str] = None,
           overlay: str = "none") -> Path:
    """Write the video with captions (and the commentary audio, if given) to `out_path`.

    `overlay` controls what else is drawn: "none" leaves the picture clean, "map" adds the
    top-down mini-map, "full" also marks every tracked player and the ball (a debug view).
    """
    src_fps, _, frames = open_source(source, default_fps)
    index_to_record = {rec.index: k for k, rec in enumerate(analysis.records)}
    last_index = analysis.records[-1].index if analysis.records else -1
    tags = [e for e in events if e.type in EVENT_LABELS and e.type not in ("pass", "turnover")]
    tmp = Path(out_path).with_suffix(".silent.mp4")
    writer = None
    k = 0

    for index, frame in enumerate(frames):
        if index > last_index:
            break
        k = index_to_record.get(index, k)
        rec, state = analysis.records[k], analysis.frames[k]
        t = index / src_fps
        h, w = frame.shape[:2]
        unit = w / 1920.0

        if overlay == "full":
            for d in rec.detections:
                if d.cls == BALL or d.track_id is None:
                    continue
                role = analysis.roles.get(d.track_id)
                if role == REFEREE:
                    continue
                x1, _, x2, y2 = d.box
                colour = (0, 220, 255) if role == GOALKEEPER else \
                    _colour(analysis.team_of.get(d.track_id), analysis.team_colours)
                centre = (int((x1 + x2) / 2), int(y2))
                axes = (max(8, int((x2 - x1) * 0.55)), max(4, int((x2 - x1) * 0.2)))
                cv2.ellipse(frame, centre, axes, 0, -30, 210, (0, 0, 0), int(6 * unit) + 1, cv2.LINE_AA)
                cv2.ellipse(frame, centre, axes, 0, -30, 210, colour, int(3 * unit) + 1, cv2.LINE_AA)
            if rec.ball is not None:
                bx, by = rec.ball.centre_px
                s = int(14 * unit)
                tri = np.array([[bx, by - s], [bx - s, by - 2.6 * s], [bx + s, by - 2.6 * s]], np.int32)
                cv2.fillPoly(frame, [tri], (255, 255, 255), cv2.LINE_AA)
                cv2.polylines(frame, [tri], True, (0, 0, 0), max(1, int(2 * unit)), cv2.LINE_AA)

        if overlay in ("map", "full"):
            mini = draw_minimap(state, analysis.team_colours, width=int(430 * unit))
            mh, mw = mini.shape[:2]
            x0, y0 = w - mw - int(24 * unit), int(24 * unit)
            frame[y0:y0 + mh, x0:x0 + mw] = cv2.addWeighted(frame[y0:y0 + mh, x0:x0 + mw], 0.25,
                                                            mini, 0.75, 0)

        active = [e for e in tags if e.t <= t <= e.t + 2.5]
        if active and overlay != "none":
            _tag(frame, EVENT_LABELS[active[-1].type])
        spoken = [ln for ln in lines if ln.t <= t <= ln.end + 0.5]
        if spoken:
            _caption(frame, spoken[-1].text, spoken[-1].speaker)
        if badge:
            _badge(frame, badge)

        out_h = int(round(h * out_width / w / 2) * 2)
        small = cv2.resize(frame, (out_width, out_h), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                     (out_width, out_h))
        writer.write(small)
    if writer is None:
        raise RuntimeError("no frames rendered")
    writer.release()

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp)]
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    if audio_path is not None:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    tmp.unlink(missing_ok=True)
    return Path(out_path)
