"""Lumped and nodal fuel/coolant thermal models — pure functions, numpy only."""

import numpy as np

from physics.axial import axial_coolant_temp, flat_power_shape
from physics.constants import (
    A_FUEL_NODE,
    H_TRANSFER_A,
    M_COOL_CP_COOL,
    M_DOT_NOM_CP_COOL,
    M_FUEL_CP_FUEL,
    T_COOLANT_INLET,
)


def t_sat_celsius(pressure_pa: float) -> float:
    """Saturation temperature (°C) from RCS pressure — linear fit to steam tables.

    Accurate to ±3°C in the 100–200 bar PWR operating range.
    At 155 bar nominal: ~341°C; actual subcooling at 300°C coolant ≈ 41°C.
    """
    p_bar = pressure_pa / 1.0e5
    return 311.0 + 0.55 * (p_bar - 100.0)


def step_thermal(
    t_fuel: float,
    t_cool: float,
    power: float,
    flow_fraction: float,
    decay_heat: float,
    dt: float,
    t_in: float = T_COOLANT_INLET,
) -> tuple[float, float]:
    """Advance fuel and coolant temperatures by one explicit-Euler timestep.

    Thermal time constants (~5 s fuel, ~2 s coolant) are large relative to
    the expected 0.1 s timestep, so explicit Euler is stable and accurate here.

    Decay heat is placed in the fuel node (physically: fission-product decay
    occurs in the fuel rods before heat transfers to the coolant).

    Args:
        t_fuel: fuel temperature (K)
        t_cool: coolant temperature (K)
        power: fission power (W)
        flow_fraction: coolant flow as fraction of nominal (0–1)
        decay_heat: decay heat power (W)
        t_in: cold-leg inlet temperature (K); defaults to nominal T_COOLANT_INLET.
            Pass the dynamic state value to model secondary heat-sink degradation.
        dt: timestep (s)

    Returns:
        (new_t_fuel, new_t_cool) in Kelvin
    """
    q_transfer = H_TRANSFER_A * (t_fuel - t_cool)
    q_removed = flow_fraction * M_DOT_NOM_CP_COOL * (t_cool - t_in)

    d_t_fuel = (power + decay_heat - q_transfer) / M_FUEL_CP_FUEL
    d_t_cool = (q_transfer - q_removed) / M_COOL_CP_COOL

    return t_fuel + d_t_fuel * dt, t_cool + d_t_cool * dt


def step_thermal_nodal(
    t_fuel: np.ndarray,
    t_cool: np.ndarray,
    void_fraction: np.ndarray,
    power_shape: np.ndarray,
    total_power: float,
    flow_fraction: float,
    decay_heat: float,
    pressure: float,
    t_in: float,
    dt: float,
    htc: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit-Euler nodal thermal step.  N = len(t_fuel).

    Fuel: integrated via explicit Euler.
    Coolant: quasi-steady via axial_coolant_temp() each tick
             (τ_cool ~2 s >> dt = 0.1 s, so quasi-steady is valid).

    htc: W/m²K per node, from two_phase.heat_transfer_coefficient().
         If None, stubbed as 30 000 W/m²K until two_phase.py exists.

    Returns (new_t_fuel, new_t_cool), each shape (N,).
    """
    N = len(t_fuel)
    if htc is None:
        htc = np.full(N, 30_000.0)

    m_fuel_cp_node = M_FUEL_CP_FUEL / N

    new_t_fuel = t_fuel.copy()
    for n in range(N):
        q_fiss = power_shape[n] * total_power / N
        q_decay = decay_heat / N
        q_trans = htc[n] * A_FUEL_NODE * (t_fuel[n] - t_cool[n])
        d_t_fuel = (q_fiss + q_decay - q_trans) * dt / m_fuel_cp_node
        new_t_fuel[n] = t_fuel[n] + d_t_fuel

    new_t_cool = axial_coolant_temp(t_in, power_shape, total_power + decay_heat, flow_fraction, N)
    return new_t_fuel, new_t_cool
