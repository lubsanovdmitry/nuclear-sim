"""Tests for physics/thermal.py — lumped fuel/coolant thermal model."""

import numpy as np
import pytest

from physics.constants import (
    NOMINAL_POWER_W,
    T_REF_COOLANT,
    T_REF_FUEL,
    T_COOLANT_INLET,
)
from physics.thermal import step_thermal


def run_steps(
    t_fuel: float,
    t_cool: float,
    power: float,
    flow_fraction: float,
    decay_heat: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float]:
    for _ in range(n_steps):
        t_fuel, t_cool = step_thermal(t_fuel, t_cool, power, flow_fraction, decay_heat, dt)
    return t_fuel, t_cool


class TestNominalSteadyState:
    """At nominal power + full flow the operating point should be stable."""

    def test_reference_point_is_fixed(self):
        # Starting exactly at (T_REF_FUEL, T_REF_COOLANT) must be a fixed point.
        t_fuel, t_cool = run_steps(
            T_REF_FUEL, T_REF_COOLANT,
            NOMINAL_POWER_W, 1.0, 0.0,
            dt=0.1, n_steps=1000,
        )
        assert abs(t_fuel - T_REF_FUEL) < 0.1
        assert abs(t_cool - T_REF_COOLANT) < 0.1

    def test_converges_from_cold(self):
        # Starting cold should converge toward reference temperatures.
        t_fuel, t_cool = run_steps(
            T_COOLANT_INLET, T_COOLANT_INLET,
            NOMINAL_POWER_W, 1.0, 0.0,
            dt=0.1, n_steps=3000,  # 300 s >> time constants (2 s, 5 s)
        )
        assert abs(t_fuel - T_REF_FUEL) < 5.0
        assert abs(t_cool - T_REF_COOLANT) < 5.0


class TestFlowLoss:
    """On flow = 0, fission power cannot be removed — temperatures must climb."""

    def test_fuel_rises_monotonically(self):
        dt = 0.1
        t_fuel, t_cool = T_REF_FUEL, T_REF_COOLANT
        history_fuel = [t_fuel]

        for _ in range(500):  # 50 s
            t_fuel, t_cool = step_thermal(
                t_fuel, t_cool, NOMINAL_POWER_W, 0.0, 0.0, dt
            )
            history_fuel.append(t_fuel)

        diffs = np.diff(history_fuel)
        # Fuel temp must be monotonically non-decreasing (flat start then rising)
        assert np.all(diffs >= -1e-9), "Fuel temperature decreased during flow loss"
        # And must end clearly above initial
        assert history_fuel[-1] > T_REF_FUEL + 10.0


class TestDecayHeat:
    """Decay heat alone (power=0, flow=0) must still heat the core."""

    def test_both_temps_rise(self):
        # Start fully equilibrated so fission-product decay in fuel is the only
        # heat source.  Coolant eventually heats via fuel-to-coolant transfer.
        decay = 1.0e7  # 10 MW — realistic post-shutdown decay heat

        t_fuel, t_cool = run_steps(
            T_COOLANT_INLET, T_COOLANT_INLET,
            0.0, 0.0, decay,
            dt=0.1, n_steps=500,  # 50 s
        )
        assert t_fuel > T_COOLANT_INLET + 1.0, "Fuel did not heat up from decay heat"
        assert t_cool > T_COOLANT_INLET + 1.0, "Coolant did not heat up from decay heat"

    def test_higher_decay_heat_gives_higher_temp(self):
        decay_low = 5.0e6   # 5 MW
        decay_high = 5.0e7  # 50 MW

        tf_low, tc_low = run_steps(
            T_COOLANT_INLET, T_COOLANT_INLET, 0.0, 0.0, decay_low, 0.1, 500
        )
        tf_high, tc_high = run_steps(
            T_COOLANT_INLET, T_COOLANT_INLET, 0.0, 0.0, decay_high, 0.1, 500
        )
        assert tf_high > tf_low
        assert tc_high > tc_low
