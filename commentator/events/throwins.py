"""Throw-ins. Placeholder until the rule is designed against real examples (see engine)."""

from __future__ import annotations

from typing import List, Sequence

from ..config import RuleConfig
from ..core import Event
from ..possession import Possession
from ..timeline import Timeline


def detect_throw_ins(tl: Timeline, segments: Sequence[Possession],
                     cfg: RuleConfig) -> List[Event]:
    return []
