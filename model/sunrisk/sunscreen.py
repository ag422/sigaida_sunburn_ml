"""Sunscreen protection: linear in applied amount, exponential wear-off, swim loss.

Protection from several applications adds linearly (consistent with the linear amount
model): SPF_eff = 1 + sum_i (SPF_i - 1) * A_i(t) / A_lab, where A_i(t) is the amount still
on the skin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

A_LAB = config.LAB_SUNSCREEN_AMOUNT.value


@dataclass
class Layer:
    spf_label: float
    amount_mg_cm2: float  # amount currently remaining on skin


@dataclass
class SunscreenState:
    layers: list[Layer] = field(default_factory=list)

    def apply(self, spf_label: float,
              amount_mg_cm2: float = config.TYPICAL_APPLIED_AMOUNT.value) -> None:
        if spf_label < 1:
            raise ValueError("SPF must be >= 1")
        self.layers.append(Layer(spf_label, amount_mg_cm2))

    def effective_spf(self) -> float:
        return 1.0 + sum((l.spf_label - 1.0) * l.amount_mg_cm2 / A_LAB for l in self.layers)

    def wear(self, dt_h: float, active: bool = False) -> None:
        tau = (config.SUNSCREEN_WEAR_TAU_ACTIVE_H if active else config.SUNSCREEN_WEAR_TAU_H).value
        k = np.exp(-dt_h / tau)
        for l in self.layers:
            l.amount_mg_cm2 *= k

    def swim(self) -> None:
        for l in self.layers:
            l.amount_mg_cm2 *= config.SWIM_RETENTION.value


def effective_spf(spf_label, amount_mg_cm2):
    """Vectorised single-layer SPF (used by the Monte Carlo)."""
    return 1.0 + (np.asarray(spf_label, dtype=float) - 1.0) * np.asarray(amount_mg_cm2, dtype=float) / A_LAB
