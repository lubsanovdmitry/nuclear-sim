"""Diesel generator state machine.

States: standby → starting → running (or failed).
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from physics.constants import DIESEL_START_DELAY_MIN, DIESEL_START_DELAY_MAX

DIESEL_STATES = frozenset({"standby", "starting", "running", "failed"})


@dataclass
class DieselState:
    state: str = "standby"       # one of DIESEL_STATES
    start_timer: float = 0.0     # elapsed time since start signal (s)
    start_delay: float = field(  # time until RUNNING; fixed at midpoint by default
        default_factory=lambda: (DIESEL_START_DELAY_MIN + DIESEL_START_DELAY_MAX) / 2.0
    )

    def __post_init__(self) -> None:
        if self.state not in DIESEL_STATES:
            raise ValueError(f"Invalid diesel state: {self.state!r}")


def step_diesels(
    states: list[DieselState],
    start_signals: list[bool],
    dt: float,
    rng: np.random.Generator | None = None,
) -> list[DieselState]:
    """Advance diesel generator states by dt seconds.

    Args:
        states: current state for each diesel generator
        start_signals: True if a start command is active for that diesel
        dt: timestep (s)

    Returns:
        New list of DieselState objects (originals are not mutated).
    """
    new_states: list[DieselState] = []
    for s, signal in zip(states, start_signals):
        ns = DieselState(
            state=s.state,
            start_timer=s.start_timer,
            start_delay=s.start_delay,
        )
        if ns.state == "standby" and signal:
            ns.state = "starting"
            ns.start_timer = 0.0
            if rng is not None:
                ns.start_delay = float(rng.uniform(DIESEL_START_DELAY_MIN, DIESEL_START_DELAY_MAX))
        elif ns.state == "starting":
            ns.start_timer += dt
            if ns.start_timer >= ns.start_delay:
                ns.state = "running"
        new_states.append(ns)
    return new_states
