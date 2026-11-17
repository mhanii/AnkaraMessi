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
                 progress: Optional[Callable[[int, int], None]] = None,
                 device: Optional[str] = None):
    """Perception only. Cached to `cache` because it is by far the slowest step."""
    if cache is not None and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    from .perception.calibrate import PitchCalibrator
    from .perception.detector import YoloDetector
    from .perception.pipeline import analyze

    analysis = analyze(
        source,
        YoloDetector(models / DETECTOR, frame_rate=analysis_fps, device=device),
        PitchCalibrator(models / KEYPOINTS, device=device),
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
        team_names: Optional[List[str]] = None, video: bool = True,
        max_seconds: Optional[float] = None, llm: Optional[str] = None,
        delay_s: float = 4.0, density: str = "high", overlay: str = "none",
        device: Optional[str] = None,
        progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, object]:
    """A finished clip in, a commentated video out.

    The vision models run over the whole clip first (the slow part, cached in the output
    folder). The commentary is then produced by the same live engine a stream would use, fed
    the saved observations in order, so a clip and a live feed sound the same.
    """
    from .live_run import run_live

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "analysis.pkl"
    started = time.time()
    analysis = analyse_clip(source, models, analysis_fps, max_seconds, cache=cache,
                            progress=progress, device=device)
    report = run_live(source, out_dir, models, delay_s=delay_s, analysis_fps=analysis_fps,
                      replay=cache, llm=llm, team_names=team_names, video=video,
                      density=density, overlay=overlay, verbose=False)
    report["vision"] = {"seconds": round(time.time() - started, 1),
                        **{k: round(float(v), 3) for k, v in analysis.stats.items()}}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1))
    return report
