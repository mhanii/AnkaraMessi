"""End to end: a clip in, a commentated video (and the data behind it) out."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .commentary import (Line, LLMNarrator, TemplateNarrator, direct, format_script,
                         make_complete, moments_from_events, retime)
from .core import Event
from .events import detect_events
from .timeline import Timeline

DETECTOR = "soccana_det.pt"
KEYPOINTS = "soccana_keypoints.pt"
TTS_MODEL = "tts/kokoro-v1.0.onnx"
TTS_VOICES = "tts/voices-v1.0.bin"


def analyse_clip(source: Path, models: Path, analysis_fps: float = 12.5,
                 max_seconds: Optional[float] = None, cache: Optional[Path] = None,
                 progress: Optional[Callable[[int, int], None]] = None):
    """Perception only. Cached to `cache` because it is by far the slowest step."""
    if cache is not None and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    from .perception.calibrate import PitchCalibrator
    from .perception.detector import YoloDetector
    from .perception.pipeline import analyze

    analysis = analyze(
        source,
        YoloDetector(models / DETECTOR, frame_rate=analysis_fps),
        PitchCalibrator(models / KEYPOINTS),
        analysis_fps=analysis_fps, max_seconds=max_seconds, progress=progress)
    if cache is not None:
        Path(cache).write_bytes(pickle.dumps(analysis))
    return analysis


def commentate(events: List[Event], team_names: Dict[int, str], clip_length: float,
               llm: Optional[str] = None) -> List[Line]:
    """`llm` is `ollama:NAME` or a path to a local transformers model that words the lines;
    without it the built-in templates are used."""
    if llm:
        narrator = LLMNarrator(make_complete(llm), team_names)
    else:
        narrator = TemplateNarrator(team_names)
    return direct(moments_from_events(events), narrator, clip_length)


def run(source: Path, out_dir: Path, models: Path, analysis_fps: float = 12.5,
        team_names: Optional[List[str]] = None, audio: bool = True, video: bool = True,
        max_seconds: Optional[float] = None, voice: str = "bm_george",
        llm: Optional[str] = None,
        progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    t0 = time.time()
    analysis = analyse_clip(source, models, analysis_fps, max_seconds,
                            cache=out_dir / "analysis.pkl", progress=progress)
    timings["perception_s"] = round(time.time() - t0, 1)

    timeline = Timeline(analysis.frames, analysis.fps)
    events = detect_events(timeline)
    clip_length = float(timeline.t[-1]) if len(timeline) else 0.0

    names = {k: v.title() for k, v in analysis.team_names.items()}
    if team_names:
        names = {k: n for k, n in enumerate(team_names[:2])}
    lines = commentate(events, names, clip_length, llm)

    audio_path = None
    if audio and lines:
        from .speech import SAMPLE_RATE, Booth, KokoroSynthesizer, mix, write_wav

        t0 = time.time()
        booth = Booth(KokoroSynthesizer(models / TTS_MODEL, models / TTS_VOICES, voice=voice),
                      cache_dir=models / "tts" / "cache", voices={"lead": voice})
        clips = [booth.say(ln.text, ln.speaker, ln.tone)[0] for ln in lines]
        kept = retime(lines, [len(c) / SAMPLE_RATE for c in clips])
        clips = [c for ln, c in zip(lines, clips) if any(ln is k for k in kept)]
        lines = kept
        audio_path = out_dir / "commentary.wav"
        write_wav(audio_path, mix([ln.t for ln in lines], clips, clip_length))
        timings["speech_s"] = round(time.time() - t0, 1)

    (out_dir / "events.json").write_text(json.dumps([e.to_dict() for e in events], indent=1))
    (out_dir / "commentary.txt").write_text(format_script(lines) + "\n")
    (out_dir / "commentary.json").write_text(json.dumps(
        [{"t": round(ln.t, 2), "duration": round(ln.duration, 2), "text": ln.text,
          "kind": ln.kind} for ln in lines], indent=1))

    video_path = None
    if video:
        from .render import render

        t0 = time.time()
        video_path = render(source, analysis, events, lines, out_dir / "commentated.mp4",
                            audio_path=audio_path)
        timings["render_s"] = round(time.time() - t0, 1)

    summary = {
        "clip_seconds": round(clip_length, 1),
        "teams": names,
        "events": len(events),
        "lines": len(lines),
        "stats": {k: round(float(v), 3) for k, v in analysis.stats.items()},
        "timings": timings,
        "video": str(video_path) if video_path else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
