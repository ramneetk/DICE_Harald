"""Piecewise-constant asynchronous calendars."""

from __future__ import annotations

import numpy as np

from .actions import action_power
from .config import VPPConfig, DEFAULT_CONFIG


class Calendar:
    """Agent calendar: E intervals × |Ai| probability simplex."""

    def __init__(
        self,
        probs: np.ndarray,
        config: VPPConfig = DEFAULT_CONFIG,
    ):
        self.config = config
        self.probs = np.asarray(probs, dtype=float)
        if self.probs.shape != (config.E, config.num_actions):
            raise ValueError(
                f"Expected shape ({config.E}, {config.num_actions}), got {self.probs.shape}"
            )
        self._normalize()

    def _normalize(self) -> None:
        row_sums = self.probs.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        self.probs = self.probs / row_sums

    def copy(self) -> "Calendar":
        return Calendar(self.probs.copy(), self.config)

    def power_profile_kw(self) -> np.ndarray:
        """Expected power per interval, shape (E,)."""
        gamma = np.asarray(self.config.action_power_kw)
        return self.probs @ gamma

    def aggregate_power_timeseries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, power) at interval midpoints."""
        dt = self.config.dwell_seconds
        times = (np.arange(self.config.E) + 0.5) * dt
        return times, self.power_profile_kw()

    def memory_bytes(self) -> int:
        """Rough per-node storage for calendar + gradient buffers."""
        e, a = self.probs.shape
        # calendar, gradient, published copy, execution copy
        return int(e * a * 8 * 4)

    @classmethod
    def idle(cls, config: VPPConfig = DEFAULT_CONFIG) -> "Calendar":
        probs = np.zeros((config.E, config.num_actions))
        probs[:, 0] = 1.0
        return cls(probs, config)

    @classmethod
    def from_hard_schedule(
        cls,
        action_indices: np.ndarray,
        config: VPPConfig = DEFAULT_CONFIG,
    ) -> "Calendar":
        E = config.E
        probs = np.zeros((E, config.num_actions))
        for k, a in enumerate(action_indices):
            probs[k, int(a)] = 1.0
        return cls(probs, config)


def swarm_aggregate_power(calendars: list[Calendar]) -> np.ndarray:
    """Total expected kW per interval across swarm."""
    if not calendars:
        return np.zeros(DEFAULT_CONFIG.E)
    stack = np.stack([c.power_profile_kw() for c in calendars], axis=0)
    return stack.sum(axis=0)


def l2_tracking_error(
    aggregate_kw: np.ndarray,
    target_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    dt = config.dwell_seconds
    diff = aggregate_kw - target_kw
    return float(np.sqrt(dt * np.sum(diff ** 2)))
