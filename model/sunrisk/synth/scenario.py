"""Day scripts: what the wearer does, where, and when."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import ACTIVITIES, ENVIRONMENTS


@dataclass
class Segment:
    minutes: float
    environment: str                 # sun / shade / indoor  ("cloud" comes from the sky, not the script)
    activity: str
    posture: str | None = None       # "supine" / "prone" for lying
    sunscreen: tuple[float, float, tuple[str, ...] | None] | None = None  # (SPF, mg/cm^2, parts) applied at start

    def __post_init__(self):
        if self.environment not in ("sun", "shade", "indoor"):
            raise ValueError(f"scripted environment must be sun/shade/indoor, got {self.environment!r}")
        if self.activity not in ACTIVITIES:
            raise ValueError(f"unknown activity {self.activity!r}")


@dataclass
class Scenario:
    start_local_min: float           # minutes after local (standard time) midnight
    segments: list[Segment]
    surface: str = "sand"            # albedo key in config.ALBEDO
    name: str = ""

    @property
    def total_minutes(self) -> float:
        return sum(s.minutes for s in self.segments)


def beach_day() -> Scenario:
    """The reference beach day used for the figure and regression tests."""
    S = Segment
    face_body = ("face", "neck", "shoulders", "upper_back", "chest", "thighs", "shins")
    return Scenario(9 * 60, [
        S(30, "indoor", "sitting"),
        S(15, "sun", "walking", sunscreen=(30, 0.8, None)),
        S(40, "sun", "lying", "supine"),
        S(35, "sun", "lying", "prone"),
        S(20, "sun", "swimming"),
        S(70, "shade", "sitting"),
        S(45, "indoor", "sitting"),
        S(45, "sun", "vigorous", sunscreen=(30, 0.8, face_body)),
        S(20, "sun", "swimming"),
        S(60, "sun", "lying", "supine"),
        S(20, "sun", "walking"),
        S(30, "indoor", "standing"),
    ], surface="sand", name="beach_day")


# Markov day generator for event-detection training/evaluation (synthetic only)
_STATES = [
    ("indoor", "sitting"), ("indoor", "standing"), ("indoor", "light_activity"),
    ("sun", "walking"), ("sun", "standing"), ("sun", "sitting"), ("sun", "lying"),
    ("sun", "vigorous"), ("sun", "swimming"), ("sun", "light_activity"),
    ("shade", "sitting"), ("shade", "standing"), ("shade", "lying"), ("shade", "walking"),
]
_MEDIAN_MIN = {"sitting": 20, "standing": 8, "light_activity": 12, "walking": 10,
               "lying": 25, "vigorous": 15, "swimming": 12}


def random_scenario(rng: np.random.Generator, hours: float = 8.0, start_local_min: float | None = None) -> Scenario:
    segs: list[Segment] = []
    total = 0.0
    prev = None
    while total < hours * 60:
        env, act = _STATES[rng.integers(len(_STATES))]
        if (env, act) == prev:
            continue
        minutes = float(np.clip(rng.lognormal(np.log(_MEDIAN_MIN[act]), 0.5), 2, 90))
        posture = rng.choice(["supine", "prone"]) if act == "lying" else None
        segs.append(Segment(minutes, env, act, posture))
        total += minutes
        prev = (env, act)
    start = start_local_min if start_local_min is not None else float(rng.uniform(8, 11) * 60)
    surface = str(rng.choice(["sand", "grass"]))
    return Scenario(start, segs, surface, name="random")


ENV_CODE = {e: i for i, e in enumerate(ENVIRONMENTS)}
ACT_CODE = {a: i for i, a in enumerate(ACTIVITIES)}
