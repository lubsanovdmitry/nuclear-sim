"""Point kinetics equations (PKE) solver — RK4, pure functions, numpy only."""

import numpy as np
from numpy.typing import NDArray

from physics.constants import BETA_GROUPS, LAMBDA_GROUPS, BETA_TOTAL, PROMPT_NEUTRON_LIFETIME


def steady_state_initial_conditions(n0: float = 1.0) -> NDArray[np.float64]:
    """Return PKE state vector [n, C1..C6] at steady state for given power level.

    At steady state with rho=0: C_i = beta_i * n / (Lambda * lambda_i)
    """
    C0 = BETA_GROUPS * n0 / (PROMPT_NEUTRON_LIFETIME * LAMBDA_GROUPS)
    return np.concatenate([[n0], C0])


def pke_derivatives(
    state: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """Compute d/dt of PKE state vector.

    Args:
        state: [n, C1, C2, C3, C4, C5, C6]
        rho: total reactivity (dimensionless)

    Returns:
        derivatives: [dn/dt, dC1/dt, ..., dC6/dt]
    """
    n = state[0]
    C = state[1:7]

    dn_dt = ((rho - BETA_TOTAL) / PROMPT_NEUTRON_LIFETIME) * n + np.dot(LAMBDA_GROUPS, C)
    dC_dt = (BETA_GROUPS / PROMPT_NEUTRON_LIFETIME) * n - LAMBDA_GROUPS * C

    return np.concatenate([[dn_dt], dC_dt])


def rk4_step(
    state: NDArray[np.float64],
    rho: float,
    dt: float,
) -> NDArray[np.float64]:
    """Advance PKE state by one RK4 step.

    Args:
        state: current [n, C1..C6]
        rho: reactivity (held constant over step)
        dt: timestep in seconds

    Returns:
        new state [n, C1..C6]
    """
    k1 = pke_derivatives(state, rho)
    k2 = pke_derivatives(state + 0.5 * dt * k1, rho)
    k3 = pke_derivatives(state + 0.5 * dt * k2, rho)
    k4 = pke_derivatives(state + dt * k3, rho)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_pke(
    state0: NDArray[np.float64],
    rho: float,
    t_end: float,
    dt: float = 0.001,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate PKE from t=0 to t_end with constant reactivity.

    Args:
        state0: initial [n, C1..C6]
        rho: constant reactivity
        t_end: end time in seconds
        dt: timestep in seconds

    Returns:
        (times, states) where states shape is (n_steps+1, 7)
    """
    n_steps = int(round(t_end / dt))
    times = np.linspace(0.0, t_end, n_steps + 1)
    states = np.empty((n_steps + 1, 7))
    states[0] = state0

    state = state0.copy()
    for i in range(n_steps):
        state = rk4_step(state, rho, dt)
        states[i + 1] = state

    return times, states
