"""Physical and algorithmic constants from Section 10 of the paper."""

from dataclasses import dataclass
from enum import Enum


class GraphonTopology(str, Enum):
    FLAT = "flat"
    HIERARCHICAL = "hierarchical"
    SCALE_FREE = "scale_free"
    NESTED_GRID = "nested_grid"


@dataclass(frozen=True)
class VPPConfig:
    # Planning horizon
    T_hours: float = 24.0
    dwell_minutes: float = 15.0

    # Action space Ai = {Idle, ChargeStandard, ChargeFast, DischargeV2G}
    action_names: tuple = ("Idle", "ChargeStandard", "ChargeFast", "DischargeV2G")
    # kW: positive = injection (V2G), negative = consumption
    action_power_kw: tuple = (0.0, -7.2, -22.0, 10.0)

    # Reward / optimization
    epsilon: float = 0.05
    discount_gamma: float = 0.01
    tolerance_tau_kw: float = 50.0  # matching tolerance relative to swarm scale
    sync_beta: float = 2.0  # heat-kernel temperature for sync probability

    # Graphon / network
    graphon_decay: float = 0.5
    ws_k: int = 4
    ws_p: float = 0.1
    neighborhood_size: int = 20

    # Guardrail (Algorithm 1)
    bisection_tol: float = 1e-6
    cooperative_floor_fraction: float = 0.5

    # Smart contract slashing (Eq. 53)
    lambda_reward: float = 1.0
    lambda_penalty: float = 10.0

    # Coordination
    max_iterations: int = 100
    convergence_tol: float = 1e-4
    c0_coherence: float = 2.0

    @property
    def T_seconds(self) -> float:
        return self.T_hours * 3600.0

    @property
    def dwell_seconds(self) -> float:
        return self.dwell_minutes * 60.0

    @property
    def E(self) -> int:
        return int(self.T_seconds / self.dwell_seconds)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)


DEFAULT_CONFIG = VPPConfig()

# Reduced horizon for fast smoke tests (--quick)
QUICK_CONFIG = VPPConfig(dwell_minutes=60.0, max_iterations=20)
