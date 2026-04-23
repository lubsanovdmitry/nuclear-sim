"""Coolant pump coastdown model — 4 pumps, each contributing 25% of nominal flow."""

import numpy as np
from physics.constants import (
    DIESEL_PUMP_RAMP_RATE,
    DIESEL_PUMP_SPEED_MAX,
    NUM_PUMPS,
    PUMP_COASTDOWN_TAU,
)


def step_pumps(
    speeds: np.ndarray,
    powered: list[bool],
    offsite_power: bool,
    dt: float,
    diesel_powered: list[bool] | None = None,
) -> np.ndarray:
    """Advance pump speeds by dt seconds.

    A pump maintains speed=1 only when offsite_power is True AND its powered flag is True.
    On diesel power, a pump ramps toward DIESEL_PUMP_SPEED_MAX at
    DIESEL_PUMP_RAMP_RATE. Otherwise it coasts down with exponential decay
    (tau=PUMP_COASTDOWN_TAU).

    Args:
        speeds: current normalised speed per pump, shape (NUM_PUMPS,), range [0, 1]
        powered: operator ON/OFF command per pump
        offsite_power: True if AC grid is available
        dt: timestep (s)
        diesel_powered: optional per-pump diesel-bus availability. If None,
            all entries are treated as False for backward compatibility.

    Returns:
        New speeds array, same shape.
    """
    new_speeds = speeds.copy()
    decay = np.exp(-dt / PUMP_COASTDOWN_TAU)
    diesel_powered_flags = diesel_powered if diesel_powered is not None else [False] * NUM_PUMPS
    for i in range(NUM_PUMPS):
        if offsite_power and powered[i]:
            new_speeds[i] = 1.0
        elif diesel_powered_flags[i] and powered[i]:
            new_speeds[i] = min(DIESEL_PUMP_SPEED_MAX, speeds[i] + DIESEL_PUMP_RAMP_RATE * dt)
        else:
            new_speeds[i] = speeds[i] * decay
    return new_speeds


def total_flow_fraction(pump_speeds: np.ndarray, delta_T: float = 0.0) -> float:
    """Return total coolant flow as fraction of nominal (0–1).

    Each of the NUM_PUMPS pumps contributes an equal share of nominal flow.
    When pump-driven flow drops below 3%, natural circulation (thermosiphon) is added.
    It scales with buoyancy head (delta_T / 30 K) and floors at 3% of nominal.

    Args:
        pump_speeds: normalised speed per pump, shape (NUM_PUMPS,), range [0, 1]
        delta_T: T_fuel - T_cool (K) — drives buoyancy head
    """
    pump_flow = float(np.sum(pump_speeds)) / NUM_PUMPS
    if pump_flow < 0.03:
        natural_circ = 0.03 * min(1.0, delta_T / 30.0)
        return max(pump_flow, natural_circ)
    return pump_flow
