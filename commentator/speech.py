"""Text to speech, tones, and mixing spoken lines into one track.

`Synthesizer` is the only thing the rest of the code knows about. `KokoroSynthesizer` is the
default: Kokoro-82M through onnxruntime, about real-time on a 4-core laptop CPU, no GPU and no
paid API. Any other engine only has to implement `synthesize`.

Kokoro has no emotion control, so tone is made by hand: how fast the line is spoken, a small
pitch shift, and loudness. An excited line is quicker, higher and louder; a deflated one is
slower, lower and quieter.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

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


@dataclass(frozen=True)
class Delivery:
    """How a tone is voiced. `pitch` > 1 raises the voice; `pace` > 1 speaks faster."""

    pace: float
    pitch: float
    gain: float


DELIVERY: Dict[str, Delivery] = {
    "excited": Delivery(pace=1.22, pitch=1.05, gain=1.0),
    "tense": Delivery(pace=1.06, pitch=1.01, gain=0.85),
    "calm": Delivery(pace=1.10, pitch=1.00, gain=0.75),
    "down": Delivery(pace=0.98, pitch=0.97, gain=0.7),
}
VOICES = {"lead": "bm_george", "analyst": "bf_emma"}


class Synthesizer(Protocol):
    def synthesize(self, text: str, voice: Optional[str] = None,
                   tone: str = "calm") -> np.ndarray:
        """Mono float32 samples at SAMPLE_RATE."""
        ...


class KokoroSynthesizer:
    def __init__(self, model: Path, voices: Path, voice: str = "bm_george"):
        from kokoro_onnx import Kokoro  # imported lazily: optional dependency

        self.engine = Kokoro(str(model), str(voices))
        self.available = set(self.engine.get_voices())
        self.voice = voice

    def synthesize(self, text: str, voice: Optional[str] = None,
                   tone: str = "calm") -> np.ndarray:
        voice = voice if voice in self.available else self.voice
        delivery = DELIVERY.get(tone, DELIVERY["calm"])
        # Pitch is shifted by resampling, which also shortens the audio, so the engine is
        # asked to speak that much slower to keep the overall pace as intended.
        samples, rate = self.engine.create(
            speakable(text), voice=voice, speed=delivery.pace / delivery.pitch,
            lang="en-gb" if voice.startswith("b") else "en-us")
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {rate}")
        samples = _trim_silence(np.asarray(samples, dtype=np.float32))
        return shape_tone(samples, delivery)


def shape_tone(samples: np.ndarray, delivery: Delivery) -> np.ndarray:
    """Apply pitch shift (by resampling) and loudness."""
    if abs(delivery.pitch - 1.0) > 1e-3 and len(samples) > 1:
        n = max(1, int(round(len(samples) / delivery.pitch)))
        samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)),
                            samples).astype(np.float32)
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    if peak > 0:
        samples = samples / peak * 0.95 * delivery.gain
    return samples.astype(np.float32)


def _trim_silence(samples: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples
    pad = int(0.05 * SAMPLE_RATE)
    return samples[max(0, loud[0] - pad):loud[-1] + pad]


class Booth:
    """The commentary box: turns (text, speaker, tone) into audio and remembers every line.

    Lines are cached on disk by voice, tone and text. Commentary repeats itself a lot
    ("Corner for White..."), so after a short while most lines cost nothing to speak, which
    is what makes instant reactions possible live.
    """

    def __init__(self, synth: Synthesizer, cache_dir: Optional[Path] = None,
                 voices: Optional[Dict[str, str]] = None):
        self.synth = synth
        self.voices = dict(VOICES, **(voices or {}))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory: Dict[str, np.ndarray] = {}
        self.hits = self.misses = 0
        self.seconds_synthesising = 0.0

    def _key(self, text: str, speaker: str, tone: str) -> str:
        raw = f"{self.voices.get(speaker, speaker)}|{tone}|{text}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def say(self, text: str, speaker: str = "lead",
            tone: str = "calm") -> Tuple[np.ndarray, float]:
        """Returns (audio, seconds it took to produce). Cached lines take 0 seconds."""
        key = self._key(text, speaker, tone)
        if key in self.memory:
            self.hits += 1
            return self.memory[key], 0.0
        path = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if path is not None and path.exists():
            self.hits += 1
            self.memory[key] = np.load(path)
            return self.memory[key], 0.0
        start = time.time()
        audio = self.synth.synthesize(text, self.voices.get(speaker), tone)
        spent = time.time() - start
        self.misses += 1
        self.seconds_synthesising += spent
        self.memory[key] = audio
        if path is not None:
            np.save(path, audio)
        return audio, spent

    def warm(self, lines: Sequence[Tuple[str, str, str]]) -> int:
        """Pre-render (text, speaker, tone) lines, e.g. before kick-off. Returns how many
        were new."""
        before = self.misses
        for text, speaker, tone in lines:
            self.say(text, speaker, tone)
        return self.misses - before


def mix(starts: Sequence[float], clips: Sequence[np.ndarray], length_s: float,
        cuts: Optional[Sequence[Optional[float]]] = None) -> np.ndarray:
    """Lay each spoken clip at its start time on one mono track. `cuts` gives, per clip, a
    time at which it was interrupted (None = played to the end)."""
    cuts = cuts or [None] * len(clips)
    ends = [s + len(c) / SAMPLE_RATE for s, c in zip(starts, clips)]
    track = np.zeros(int(np.ceil(max([length_s] + ends) * SAMPLE_RATE)) + 1, dtype=np.float32)
    fade = int(0.08 * SAMPLE_RATE)
    for start, clip, cut in zip(starts, clips, cuts):
        if cut is not None:
            keep = max(0, int(round((cut - start) * SAMPLE_RATE)))
            clip = clip[:keep].copy()
            if len(clip) > fade:
                clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        begin = max(0, int(round(start * SAMPLE_RATE)))
        track[begin:begin + len(clip)] += clip
    peak = float(np.abs(track).max())
    if peak > 0.98:
        track *= 0.98 / peak
    return track


def write_wav(path: Path, track: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), track, SAMPLE_RATE)
