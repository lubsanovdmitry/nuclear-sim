"""Tests for Xe-135/I-135 kinetics and ANS-5.1 decay heat model."""

import numpy as np
import pytest

from physics.xenon import (
    xenon_equilibrium,
    xenon_derivatives,
    xenon_reactivity,
    rk4_step_xenon,
)
from physics.decay_heat import decay_heat_fraction, decay_heat_power
from physics.constants import NOMINAL_POWER_W


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_shutdown(t_hours: float, dt_s: float = 300.0) -> tuple[list[float], list[float]]:
    """Integrate I/Xe kinetics from full-power equilibrium after shutdown (n=0).

    Returns (Xe_values, t_values) sampled every dt_s seconds.
    """
    I, Xe = xenon_equilibrium(n=1.0)
    t_end = t_hours * 3600.0
    t = 0.0
    t_values: list[float] = [t]
    Xe_values: list[float] = [Xe]

    while t < t_end - dt_s / 2.0:
        I, Xe = rk4_step_xenon(I, Xe, n=0.0, dt=dt_s)
        t += dt_s
        t_values.append(t)
        Xe_values.append(Xe)

    return Xe_values, t_values


# ---------------------------------------------------------------------------
# Xenon kinetics tests
# ---------------------------------------------------------------------------

class TestXenonEquilibrium:
    def test_equilibrium_zero_at_zero_power(self):
        I, Xe = xenon_equilibrium(n=0.0)
        assert I == 0.0
        assert Xe == 0.0

    def test_equilibrium_positive_at_full_power(self):
        I, Xe = xenon_equilibrium(n=1.0)
        assert I > 0.0
        assert Xe > 0.0

    def test_equilibrium_scales_with_power(self):
        """Iodine equilibrium is proportional to power (no burnout)."""
        I_full, _ = xenon_equilibrium(n=1.0)
        I_half, _ = xenon_equilibrium(n=0.5)
        assert abs(I_half / I_full - 0.5) < 1e-6

    def test_equilibrium_xenon_reactivity_range(self):
        """Equilibrium xenon worth at full power: −2 to −4 % Δk/k."""
        _, Xe_eq = xenon_equilibrium(n=1.0)
        rho = xenon_reactivity(Xe_eq)
        assert -0.04 < rho < -0.015, (
            f"Equilibrium xenon reactivity {rho:.4f} outside expected range"
        )


class TestXenonBuildupAfterShutdown:
    def test_xenon_increases_in_first_six_hours(self):
        """Xenon concentration is higher at 6 h than at shutdown — buildup confirmed."""
        Xe_values, _ = _simulate_shutdown(t_hours=6.0)
        assert Xe_values[-1] > Xe_values[0], (
            "Xenon must be higher at t=6 h than at shutdown"
        )

    def test_xenon_peak_time(self):
        """Xenon peaks between 8 and 14 hours after full-power shutdown.

        Analytical result: t_peak = ln(λ_I/λ_Xe)/(λ_I−λ_Xe) ≈ 11.3 h
        for the SPEC decay constants (λ_I=2.87e-5, λ_Xe=2.09e-5 /s).
        The SPEC description of '3–6 hr' is an approximation; the physics
        gives ~11 h for a full-power shutdown with these lambda values.
        """
        Xe_values, t_values = _simulate_shutdown(t_hours=20.0)
        peak_idx = int(np.argmax(Xe_values))
        peak_hours = t_values[peak_idx] / 3600.0
        assert 8.0 <= peak_hours <= 14.0, (
            f"Xenon peak at {peak_hours:.1f} h, expected 8–14 h"
        )

    def test_peak_xenon_exceeds_equilibrium(self):
        """Peak xenon concentration exceeds full-power equilibrium value."""
        _, Xe_eq = xenon_equilibrium(n=1.0)
        Xe_values, _ = _simulate_shutdown(t_hours=20.0)
        assert max(Xe_values) > Xe_eq, "Peak xenon must exceed equilibrium value"

    def test_xenon_clears_at_40_hours(self):
        """Xenon returns toward equilibrium within 40 h (xenon pit clears)."""
        _, Xe_eq = xenon_equilibrium(n=1.0)
        Xe_values, _ = _simulate_shutdown(t_hours=40.0)
        # With phi=0, Xe decays to 0 (no ongoing production from flux).
        # At 40 h, Xe should be below the full-power equilibrium value.
        assert Xe_values[-1] < Xe_eq, (
            "At 40 h, xenon should be falling back below full-power equilibrium"
        )


class TestXenonReactivity:
    def test_reactivity_zero_at_zero_xenon(self):
        assert xenon_reactivity(0.0) == 0.0

    def test_reactivity_negative_during_buildup(self):
        """xenon_reactivity is negative and becomes more negative as Xe rises."""
        Xe_values, t_values = _simulate_shutdown(t_hours=12.0)

        # Sample at 2 h, 6 h, 10 h — all during the rising phase (~peak at 11 h)
        idx_2h = next(i for i, t in enumerate(t_values) if t >= 2 * 3600)
        idx_6h = next(i for i, t in enumerate(t_values) if t >= 6 * 3600)
        idx_10h = next(i for i, t in enumerate(t_values) if t >= 10 * 3600)

        rho_2h = xenon_reactivity(Xe_values[idx_2h])
        rho_6h = xenon_reactivity(Xe_values[idx_6h])
        rho_10h = xenon_reactivity(Xe_values[idx_10h])

        assert rho_2h < 0.0, "Xenon reactivity must be negative"
        assert rho_6h < rho_2h, "Reactivity must become more negative from 2 h to 6 h"
        assert rho_10h < rho_6h, "Reactivity must become more negative from 6 h to 10 h"

    def test_reactivity_proportional_to_xenon(self):
        """xenon_reactivity is linear in Xe."""
        Xe = 1.0e21
        assert abs(xenon_reactivity(2 * Xe) / xenon_reactivity(Xe) - 2.0) < 1e-9


class TestXenonDerivatives:
    def test_derivatives_at_equilibrium_zero(self):
        """At full-power equilibrium, dI/dt = dXe/dt = 0."""
        I_eq, Xe_eq = xenon_equilibrium(n=1.0)
        dI, dXe = xenon_derivatives(I_eq, Xe_eq, n=1.0)
        assert abs(dI) < 1.0, f"dI/dt at equilibrium = {dI:.2e}, expected ~0"
        assert abs(dXe) < 1.0, f"dXe/dt at equilibrium = {dXe:.2e}, expected ~0"

    def test_xe_derivative_positive_just_after_shutdown(self):
        """Immediately after shutdown, dXe/dt > 0 — Xe starts building up."""
        I_eq, Xe_eq = xenon_equilibrium(n=1.0)
        _, dXe = xenon_derivatives(I_eq, Xe_eq, n=0.0)
        assert dXe > 0.0, "dXe/dt must be positive immediately after shutdown"


# ---------------------------------------------------------------------------
# Decay heat tests
# ---------------------------------------------------------------------------

class TestDecayHeat:
    def test_initial_fraction_approximately_seven_percent(self):
        """Decay heat at t=0 is in the 5–9% range (ANS-5.1 gives ~6.3%)."""
        fraction = decay_heat_fraction(t=0.0)
        assert 0.05 < fraction < 0.09, (
            f"Initial decay heat fraction = {fraction:.4f}, expected ~6–7%"
        )

    def test_decay_heat_drops_substantially_in_one_hour(self):
        """Decay heat drops from ~6.3% to ~1.1% within 1 hour.

        Note: ANS-5.1 gives ~1.1% at 3600 s, not <1%.  The slow-decaying
        fission-product groups (t½ ≈ hours) maintain ~1% of nominal power
        throughout the first hour.  The test uses <2% as the physical bound.
        """
        fraction = decay_heat_fraction(t=3600.0)
        assert fraction < 0.020, (
            f"Decay heat at 1 h = {fraction:.4f} (expected < 2%)"
        )

    def test_decay_heat_is_monotonically_decreasing(self):
        """Decay heat decreases strictly after shutdown."""
        times = [0, 30, 120, 600, 1800, 3600, 7200, 14400]
        fractions = [decay_heat_fraction(t) for t in times]
        for i in range(len(fractions) - 1):
            assert fractions[i + 1] < fractions[i], (
                f"Non-monotonic: f({times[i]}) = {fractions[i]:.5f}, "
                f"f({times[i+1]}) = {fractions[i+1]:.5f}"
            )

    def test_decay_heat_approaches_zero_at_long_times(self):
        """Decay heat fraction < 0.1% after ~3 years."""
        fraction = decay_heat_fraction(t=1e8)
        assert fraction < 1e-3

    def test_decay_heat_power_scales_with_nominal(self):
        """decay_heat_power returns fraction × nominal power."""
        frac = decay_heat_fraction(0.0)
        power = decay_heat_power(0.0, NOMINAL_POWER_W)
        assert abs(power - frac * NOMINAL_POWER_W) < 1.0

    def test_decay_heat_at_one_hour_in_megawatts(self):
        """At 1 h, decay heat power is in the ~30–60 MW range for a 3000 MWt reactor."""
        power_mw = decay_heat_power(3600.0, NOMINAL_POWER_W) / 1e6
        assert 25.0 < power_mw < 70.0, (
            f"Decay heat at 1 h = {power_mw:.1f} MW, expected 25–70 MW"
        )
