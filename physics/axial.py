"""Axial nodalization for PWR core — pure functions, numpy only.

Node 0 = bottom (coolant inlet), node N-1 = top (hot leg outlet).
All temperatures in Kelvin, power in Watts, flow as fraction of nominal.
"""

import numpy as np

from physics.constants import A_FUEL_NODE, CP_COOL, NOMINAL_FLOW_RATE


def cosine_power_shape(N: int = 10) -> np.ndarray:
    """Chopped-cosine axial power distribution, normalized so mean == 1.0.

    q[n] = cos(pi * (n - (N-1)/2) / (N * 1.2))
    Center nodes are highest (~1.3×); edge nodes are lowest.
    """
    n = np.arange(N, dtype=float)
    q = np.cos(np.pi * (n - (N - 1) / 2.0) / (N * 1.2))
    return q / q.mean()


def flat_power_shape(N: int = 10) -> np.ndarray:
    """Uniform axial power — returns np.ones(N)."""
    return np.ones(N)


def axial_coolant_temp(
    t_in: float,
    power_shape: np.ndarray,
    total_power: float,
    flow_fraction: float,
    N: int = 10,
) -> np.ndarray:
    """Coolant temperature profile from bottom to top via enthalpy rise.

    dT[n] = power_shape[n] * total_power / (N * m_dot * CP_COOL)
    T[0]  = t_in + dT[0]
    T[n]  = T[n-1] + dT[n]

    Guard: if m_dot < 1 kg/s, clamp each dT to 50 K to prevent blowup.
    Returns array of shape (N,) in Kelvin.
    """
    m_dot = flow_fraction * NOMINAL_FLOW_RATE
    if m_dot < 1.0:
        denom = N * max(m_dot, 1e-12) * CP_COOL
        dT = np.minimum(power_shape * total_power / denom, 50.0)
    else:
        dT = power_shape * total_power / (N * m_dot * CP_COOL)

    T = np.empty(N)
    T[0] = t_in + dT[0]
    for i in range(1, N):
        T[i] = T[i - 1] + dT[i]
    return T


def axial_fuel_temp(
    t_cool: np.ndarray,
    power_shape: np.ndarray,
    total_power: float,
    htc: np.ndarray,
    N: int = 10,
) -> np.ndarray:
    """Steady-state fuel centerline temperature at each axial node.

    T_fuel[n] = t_cool[n] + power_shape[n] * total_power / (N * htc[n] * A_FUEL_NODE)

    htc should come from two_phase.heat_transfer_coefficient(); during Phase 2-S1
    the caller stubs it with np.full(N, 30000.0).
    Returns array of shape (N,) in Kelvin.
    """
    return t_cool + power_shape * total_power / (N * htc * A_FUEL_NODE)
