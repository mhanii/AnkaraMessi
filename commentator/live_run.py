"""Run the commentator as if on a live feed, and measure how well the machine keeps up.

`run_live` feeds a source frame by frame. With `pace="virtual"` the clock is the video's own
time, which answers "is the commentary causal and how soon is each thing known?" on any
machine. With `pace="real"` the loop also sleeps to wall-clock time and reports whether the
machine fell behind, which is the real test on a GPU.

`replay` takes a saved analysis instead of running the vision models, so everything after
perception can be exercised at true speed on a machine too slow to do the vision live.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .commentary import Line, LLMNarrator, TemplateNarrator, make_complete
from .live import Analyst, LiveCommentator, LiveDirector, Spoken, latency_report
from .run import DETECTOR, KEYPOINTS, TTS_MODEL, TTS_VOICES

ANALYST_STYLE = ("the co-commentator beside the lead commentator at a football match: warm, "
                 "sharp, a former player giving a quick opinion on what just happened")


def stock_lines() -> List[tuple]:
    """Every line that mentions no team and no number: (text, speaker, tone)."""
    from .commentary import TEMPLATES, tone_of

    out = []
    for kind, options in TEMPLATES.items():
        speaker = "analyst" if kind.startswith("colour") else "lead"
        for text in options:
            if "{" not in text:
                out.append((text, speaker, tone_of(kind)))
    return out


def _stream(source: Path, perception, replay, analysis_fps: float) -> Iterator[float]:
    """Advance perception one analysed frame at a time; yields the stream time."""
    if replay is not None:
        for record in replay.records:
            perception.ingest(record)
            yield record.t
        return
    from .perception.pipeline import open_source

    src_fps, _, frames = open_source(source)
    step = max(1, int(round(src_fps / analysis_fps)))
    for index, frame in enumerate(frames):
        if index % step == 0:
            perception.step(frame, index, index / src_fps)
            yield index / src_fps


def run_live(source: Path, out_dir: Path, models: Path, delay_s: float = 4.0,
             analysis_fps: float = 12.5, replay: Optional[Path] = None,
             llm: Optional[str] = None, team_names: Optional[List[str]] = None,
             pace: str = "virtual", device: Optional[str] = None, video: bool = True,
             window_s: float = 45.0, verbose: bool = True, density: str = "high",
             overlay: str = "none") -> Dict[str, object]:
    from .perception.pipeline import Perception
    from .speech import Booth, KokoroSynthesizer, mix, write_wav

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = pickle.loads(Path(replay).read_bytes()) if replay else None
    if saved is not None:
        perception = Perception(None, None, saved.fps, saved.source_fps)
    else:
        from .perception.calibrate import PitchCalibrator
        from .perception.detector import YoloDetector
        from .perception.pipeline import open_source

        src_fps = open_source(source)[0]
        step = max(1, int(round(src_fps / analysis_fps)))
        perception = Perception(
            YoloDetector(models / DETECTOR, frame_rate=src_fps / step, device=device),
            PitchCalibrator(models / KEYPOINTS, device=device), src_fps / step, src_fps)

    names: Dict[int, str] = dict(enumerate(team_names[:2])) if team_names else {}
    lead = TemplateNarrator(names)
    if llm:
        analyst_voice = LLMNarrator(make_complete(llm), names, style=ANALYST_STYLE)
    else:
        analyst_voice = TemplateNarrator(names, seed=11)
    booth = Booth(KokoroSynthesizer(models / TTS_MODEL, models / TTS_VOICES),
                  cache_dir=models / "tts" / "cache")
    director = LiveDirector(lead, booth, analyst_voice, Analyst(), delay_s, density=density)
    # Before kick-off: record every line that needs no match facts, so they cost nothing live.
    warm_started = time.time()
    warmed = booth.warm(stock_lines())
    warm_seconds = time.time() - warm_started
    live = LiveCommentator(director)

    def announce(line: Spoken) -> None:
        if verbose:
            who = "ANALYST" if line.speaker == "analyst" else "LEAD   "
            print(f"  on air {line.start:5.1f}s  {who} [{line.tone:7s}] {line.text}", flush=True)

    wall_start = time.time()
    worst_behind = 0.0
    now = 0.0
    seconds_snapshot = 0.0
    for now in _stream(source, perception, saved, analysis_fps):
        if pace == "real":
            ahead = now - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)
            worst_behind = max(worst_behind, -ahead)
        if live.due_for_rules(now):
            started = time.time()
            snap = perception.snapshot(last_seconds=window_s)
            seconds_snapshot += time.time() - started
            # Name each team once, from its shirts, and then stick to it. Re-deriving the
            # colour word every half second makes one team change name as the light changes.
            for k, v in snap.team_names.items():
                if k not in names and sum(t == k for t in snap.team_of.values()) >= 5:
                    names[k] = v.title()
            live.update(snap.frames, snap.fps, now)
        for line in live.speak(now):
            announce(line)

    # The feed has ended; the delayed picture still has `delay_s` seconds left to show.
    clip_length = now
    director.closing = True
    tick = 1.0 / max(perception.fps, 1.0)
    while now < clip_length + delay_s + 5.0:
        now += tick
        if pace == "real":
            time.sleep(max(0.0, now - (time.time() - wall_start)))
        for line in live.speak(now):
            announce(line)

    aired = director.aired
    shift = [max(0.0, s.start - delay_s) for s in aired]  # onto the delayed picture's clock
    lines = [Line(t, s.text, s.kind, s.priority, s.end - s.start, s.speaker, s.tone)
             for t, s in zip(shift, aired)]
    audio_path = out_dir / "commentary_live.wav"
    write_wav(audio_path, mix(shift, [s.audio for s in aired], clip_length,
                              cuts=[None if s.cut_at is None else s.cut_at - delay_s
                                    for s in aired]))

    script = []
    for t, s in zip(shift, aired):
        who = "ANALYST" if s.speaker == "analyst" else "LEAD"
        timing = "in sync" if s.late_by <= 0.3 else f"{s.late_by:.1f}s late"
        script.append(f"[{int(t // 60):02d}:{t % 60:04.1f}] {who:7s} ({s.tone}) {s.text}"
                      f"   <- happened {s.event_t:.1f}s, known {s.decided_t:.1f}s, {timing}"
                      + (", cut off" if s.cut_at is not None else ""))
    (out_dir / "commentary_live.txt").write_text("\n".join(script) + "\n")

    frames_done = max(1, len(perception.records))
    report: Dict[str, object] = {
        "clip_seconds": round(clip_length, 1),
        "pace": pace,
        "vision": "replayed from a saved analysis" if saved is not None else "run live",
        "latency": latency_report(aired, delay_s),
        "rules": {"window_s": window_s, "runs": live.ticks,
                  "seconds_per_run": round(live.seconds_in_rules / max(1, live.ticks), 3),
                  "snapshot_seconds_per_run": round(seconds_snapshot / max(1, live.ticks), 3)},
        "speech": {"stock_lines_recorded_before_start": warmed,
                   "seconds_recording_them": round(warm_seconds, 1),
                   "lines_from_cache": booth.hits, "lines_synthesised": booth.misses,
                   "seconds_synthesising": round(booth.seconds_synthesising, 1)},
        "dropped": [f"{kind}: {text} ({why})" for kind, text, why in director.dropped],
    }
    if isinstance(analyst_voice, LLMNarrator):
        report["language_model"] = {"model": llm, "lines_accepted": analyst_voice.accepted,
                                    "lines_rejected_for_facts": analyst_voice.rejected,
                                    "seconds_wording": round(director.seconds_wording, 1)}
    if saved is None:
        per_frame = sum(perception.timing.values()) / frames_done
        report["vision_speed"] = {
            "seconds_per_analysed_frame": round(per_frame, 3),
            "budget_for_real_time": round(1.0 / perception.fps, 3),
            "real_time_factor": round(per_frame * perception.fps, 2),
            "keeps_up": per_frame * perception.fps <= 1.0,
        }
    if pace == "real":
        report["fell_behind_by_s"] = round(worst_behind, 2)

    if video:
        from .render import render

        final = perception.snapshot()
        report["video"] = str(render(source, final, live.events, lines,
                                     out_dir / "commentated_live.mp4", audio_path=audio_path,
                                     badge=f"LIVE  +{delay_s:g}s delay", overlay=overlay))
    (out_dir / "live_report.json").write_text(json.dumps(report, indent=1))
    return report


def bench(source: Path, models: Path, device: Optional[str] = None, frames: int = 30,
          analysis_fps: float = 12.5) -> Dict[str, object]:
    """Time each stage on this machine and say whether it can run live."""
    from .perception.calibrate import PitchCalibrator
    from .perception.detector import YoloDetector
    from .perception.pipeline import SET_PIECE_SPOTS, open_source
    from .speech import SAMPLE_RATE, KokoroSynthesizer

    detector = YoloDetector(models / DETECTOR, device=device)
    calibrator = PitchCalibrator(models / KEYPOINTS, device=device)
    src_fps, _, stream = open_source(source)
    step = max(1, int(round(src_fps / analysis_fps)))
    spent = {"detect": 0.0, "calibrate": 0.0, "probe": 0.0}
    done = 0
    for index, frame in enumerate(stream):
        if index % step:
            continue
        if done == 0:  # first call loads and warms the models; do not count it
            detector.detect(frame)
            calibrator.calibrate(frame)
        t0 = time.time()
        detector.detect(frame)
        spent["detect"] += time.time() - t0
        t0 = time.time()
        hom = calibrator.calibrate(frame)
        spent["calibrate"] += time.time() - t0
        if hom is not None:
            t0 = time.time()
            detector.probe_ball(frame, hom.to_image(SET_PIECE_SPOTS))
            spent["probe"] += time.time() - t0
        done += 1
        if done >= frames:
            break
    per_frame = {k: v / max(1, done) for k, v in spent.items()}
    total = sum(per_frame.values())

    synth = KokoroSynthesizer(models / TTS_MODEL, models / TTS_VOICES)
    synth.synthesize("Warm up.")
    t0 = time.time()
    audio = synth.synthesize("Corner for the home side, swung in towards the near post!",
                             tone="excited")
    tts_seconds = time.time() - t0

    return {
        "device": device or "cpu",
        "frames_timed": done,
        "milliseconds_per_analysed_frame": {k: round(v * 1000, 1) for k, v in per_frame.items()},
        "total_ms_per_frame": round(total * 1000, 1),
        "fastest_analysis_rate_fps": round(1.0 / total, 1) if total else None,
        "live_at_12.5_fps": total <= 1 / 12.5,
        "live_at_6.25_fps": total <= 1 / 6.25,
        "speech": {"seconds_of_audio": round(len(audio) / SAMPLE_RATE, 2),
                   "seconds_to_make": round(tts_seconds, 2),
                   "faster_than_real_time": tts_seconds < len(audio) / SAMPLE_RATE},
    }
