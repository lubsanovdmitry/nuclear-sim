"""Pressurizer pressure and liquid-level models for the RCS.

Simple first-order model: pressure tracks a temperature-dependent setpoint.
Pressure rises when coolant overheats (loss of cooling) and falls when
coolant cools (LOCA depressurization).

SCRAM setpoints (from constants):
  high: > 170 bar (1.70e7 Pa)
  low:  < 120 bar (1.20e7 Pa)
"""
import numpy as np

from physics.constants import (
    PRESSURE_NOMINAL,
    PRESSURE_SCRAM_HIGH,
    PRESSURE_SCRAM_LOW,
    PRESSURE_TAU,
    PRESSURE_TEMP_COEFF,
    T_REF_COOLANT,
    PRZR_LEVEL_NOMINAL,
    PRZR_LEVEL_TEMP_COEFF,
    PRZR_LEVEL_TAU,
    PRZR_DRAIN_COEFF,
    PRZR_MAKEUP_COEFF,
    PRZR_HEATER_DPDT_MAX,
    PRZR_SPRAY_DPDT_MAX,
    PORV_FLOW_DEFICIT,
)


def step_pressurizer(
    pressure: float,
    t_cool: float,
    flow_fraction: float,
    dt: float,
    heater_fraction: float = 0.0,
    spray_fraction: float = 0.0,
    porv_open: bool = False,
) -> float:
    """Advance RCS pressure by dt seconds.

    Target pressure shifts with coolant temperature relative to nominal.
    A first-order lag (tau = PRESSURE_TAU) governs the response.
    Flow loss and an open PORV lower the target via equivalent flow-deficit terms,
    so the lag actively drives pressure down rather than fighting external drains.
    Operator heaters raise pressure; spray lowers it.

    Args:
        pressure: current RCS pressure (Pa)
        t_cool: current coolant temperature (K)
        flow_fraction: current coolant flow as fraction of nominal (0–1)
        dt: timestep (s)
        heater_fraction: operator heater demand 0–1 (raises pressure)
        spray_fraction: operator spray demand 0–1 (lowers pressure)
        porv_open: True when PORV is open (adds equivalent flow deficit to p_target)

    Returns:
        New RCS pressure (Pa).
    """
    p_target = PRESSURE_NOMINAL + PRESSURE_TEMP_COEFF * (t_cool - T_REF_COOLANT)
    # Flow loss (pump coastdown / LOCA) and open PORV both lower p_target.
    # Treating them as equivalent flow deficits means the first-order lag drives
    # pressure downward — the physically correct direction.
    flow_deficit = (1.0 - flow_fraction) + (PORV_FLOW_DEFICIT if porv_open else 0.0)
    p_target -= PRESSURE_TEMP_COEFF * 50.0 * flow_deficit  # 50 K equivalent per unit flow loss
    # Cold ECCS injection at 293 K drives p_target deeply negative via the linear
    # temperature coefficient; clamp to atmospheric so the lag can't pull RCS sub-zero.
    p_target = max(p_target, 1e5)
    dp_dt = (p_target - pressure) / PRESSURE_TAU
    dp_dt += heater_fraction * PRZR_HEATER_DPDT_MAX
    dp_dt -= spray_fraction * PRZR_SPRAY_DPDT_MAX
    return max(pressure + dp_dt * dt, 1e5)


def step_pressurizer_level(
    level: float,
    t_cool: float,
    pressure: float,
    eccs_flow: float,
    dt: float,
) -> float:
    """Advance pressurizer liquid level by dt seconds.

    Level tracks thermal expansion of the RCS water inventory.  Inventory is
    lost through a break (LOCA) in proportion to how far pressure has fallen
    below nominal; ECCS injection provides makeup flow.

    Args:
        level: current pressurizer level (fraction 0–1)
        t_cool: coolant temperature (K)
        pressure: RCS pressure (Pa)
        eccs_flow: ECCS injection_flow_fraction (0–1+ as fraction of nominal flow)
        dt: timestep (s)

    Returns:
        New pressurizer level (fraction, clamped to [0, 1]).
    """
    l_eq = PRZR_LEVEL_NOMINAL + PRZR_LEVEL_TEMP_COEFF * (t_cool - T_REF_COOLANT)
    dl_thermal = (l_eq - level) / PRZR_LEVEL_TAU
    pressure_deficit_frac = max(0.0, (PRESSURE_NOMINAL - pressure) / PRESSURE_NOMINAL)
    dl_drain = -PRZR_DRAIN_COEFF * pressure_deficit_frac
    dl_makeup = PRZR_MAKEUP_COEFF * eccs_flow
    return float(np.clip(level + (dl_thermal + dl_drain + dl_makeup) * dt, 0.0, 1.0))


def check_scram_pressure(pressure: float) -> bool:
    """Return True if pressure is outside safe operating limits.

    Triggers SCRAM at > 170 bar (1.70e7 Pa) or < 120 bar (1.20e7 Pa).

    Args:
        pressure: RCS pressure (Pa)

    Returns:
        True if a pressure SCRAM condition exists.
    """
    return pressure > PRESSURE_SCRAM_HIGH or pressure < PRESSURE_SCRAM_LOW
