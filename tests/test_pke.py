"""PKE solver tests."""

import numpy as np
import pytest

from physics.pke_solver import steady_state_initial_conditions, solve_pke
from physics.constants import BETA_TOTAL


def test_steady_state_rho_zero_stable_60s():
    """Steady state at n=1, rho=0 must remain within 0.1% over 60 seconds."""
    state0 = steady_state_initial_conditions(n0=1.0)
    times, states = solve_pke(state0, rho=0.0, t_end=60.0)

    power = states[:, 0]
    assert power.min() > 0.999, f"Power dropped below 0.999: min={power.min():.6f}"
    assert power.max() < 1.001, f"Power rose above 1.001: max={power.max():.6f}"


def test_large_negative_reactivity_drops_power_below_1pct():
    """rho=-0.01 must drive power below 1% within 200s.

    Prompt drop takes n to ~39% instantly. The longest-lived precursor group
    (group 1, T½≈56s) sustains power for several minutes; 1% is crossed ~140s.
    """
    state0 = steady_state_initial_conditions(n0=1.0)
    times, states = solve_pke(state0, rho=-0.01, t_end=200.0)

    final_power = states[-1, 0]
    assert final_power < 0.01, (
        f"Power did not drop below 1% in 200s: final power = {final_power:.4f}"
    )


def test_small_positive_reactivity_raises_power():
    """rho=+0.001 (sub-prompt-critical) must raise power above initial within 60s
    and remain physically bounded (no prompt-critical divergence)."""
    state0 = steady_state_initial_conditions(n0=1.0)
    times, states = solve_pke(state0, rho=0.001, t_end=60.0)

    power = states[:, 0]
    final_power = power[-1]

    # Power must rise
    assert final_power > 1.0, f"Power did not increase: final={final_power:.4f}"

    # rho < beta so growth is on delayed timescale — power stays bounded
    assert final_power < 1000.0, f"Power diverged unphysically: final={final_power:.4f}"

    # Confirm rho < beta (prompt sub-critical regime, finite stable period)
    assert 0.001 < BETA_TOTAL, "Test assumption violated: rho must be below beta_total"
