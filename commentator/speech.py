"""Text to speech, and mixing spoken lines into one track.

`Synthesizer` is the only thing the rest of the code knows about. `KokoroSynthesizer` is the
default: Kokoro-82M through onnxruntime, about real-time on a 4-core laptop CPU, no GPU and no
paid API. Any other engine (for example tiny-voca) only has to implement `synthesize`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Protocol, Sequence, Tuple

import numpy as np

from .commentary import Line

SAMPLE_RATE = 24000

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell_number(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens - 2] + ("-" + _ONES[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + (" and " + spell_number(rest) if rest else "")
    return str(n)


def speakable(text: str) -> str:
    """Make a line safe for any TTS engine: digits become words."""
    return re.sub(r"\d+", lambda m: spell_number(int(m.group())), text)


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> np.ndarray:
        """Mono float32 samples at SAMPLE_RATE."""
        ...


class KokoroSynthesizer:
    def __init__(self, model: Path, voices: Path, voice: str = "bm_george", speed: float = 1.12):
        from kokoro_onnx import Kokoro  # imported lazily: optional dependency

        self.engine = Kokoro(str(model), str(voices))
        self.voice, self.speed = voice, speed
        self.lang = "en-gb" if voice.startswith("b") else "en-us"

    def synthesize(self, text: str) -> np.ndarray:
        samples, rate = self.engine.create(speakable(text), voice=self.voice, speed=self.speed,
                                           lang=self.lang)
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {rate}")
        samples = np.asarray(samples, dtype=np.float32)
        return _trim_silence(samples)


def _trim_silence(samples: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples
    pad = int(0.05 * SAMPLE_RATE)
    return samples[max(0, loud[0] - pad):loud[-1] + pad]


def synthesize_lines(lines: Sequence[Line], synth: Synthesizer) -> List[np.ndarray]:
    return [synth.synthesize(ln.text) for ln in lines]


def mix(lines: Sequence[Line], clips: Sequence[np.ndarray], length_s: float) -> np.ndarray:
    """Lay each spoken line at its start time on one mono track."""
    end = max([length_s] + [ln.t + len(c) / SAMPLE_RATE for ln, c in zip(lines, clips)])
    track = np.zeros(int(np.ceil(end * SAMPLE_RATE)) + 1, dtype=np.float32)
    for ln, clip in zip(lines, clips):
        start = int(round(ln.t * SAMPLE_RATE))
        track[start:start + len(clip)] += clip
    peak = float(np.abs(track).max())
    if peak > 0.98:
        track *= 0.98 / peak
    return track


def write_wav(path: Path, track: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), track, SAMPLE_RATE)
