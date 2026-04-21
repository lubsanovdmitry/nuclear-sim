"""Tests for physics/axial.py and Phase 2 additions to physics/thermal.py."""

import numpy as np
import pytest

from physics.axial import (
    axial_coolant_temp,
    axial_fuel_temp,
    cosine_power_shape,
    flat_power_shape,
)
from physics.constants import (
    NOMINAL_FLOW_RATE,
    NOMINAL_POWER_W,
    T_COOLANT_INLET,
    T_REF_COOLANT,
    T_REF_FUEL,
)
from physics.thermal import step_thermal


# ── 1. cosine_power_shape ──────────────────────────────────────────────────────

class TestCosinePowerShape:
    def test_mean_is_one(self):
        q = cosine_power_shape(10)
        assert abs(q.mean() - 1.0) < 1e-12

    def test_center_is_max(self):
        q = cosine_power_shape(10)
        center = q[4:6].max()
        assert center == q.max(), "Peak must be at central nodes (4 or 5)"

    def test_edges_are_min(self):
        q = cosine_power_shape(10)
        edge_max = max(q[0], q[-1])
        assert edge_max == q.min(), "Edge nodes must be the minimum"

    def test_symmetric(self):
        q = cosine_power_shape(10)
        np.testing.assert_allclose(q, q[::-1], atol=1e-12)

    def test_flat_is_ones(self):
        q = flat_power_shape(10)
        np.testing.assert_array_equal(q, np.ones(10))


# ── 2. axial_coolant_temp — monotonicity ──────────────────────────────────────

class TestAxialCoolantTempMonotonic:
    @pytest.mark.parametrize("flow", [0.2, 0.5, 1.0])
    def test_monotonically_increasing(self, flow: float):
        shape = cosine_power_shape(10)
        T = axial_coolant_temp(T_COOLANT_INLET, shape, NOMINAL_POWER_W, flow)
        diffs = np.diff(T)
        assert np.all(diffs >= 0), f"Coolant temp not monotonic at flow={flow}"

    def test_flat_shape_also_monotonic(self):
        shape = flat_power_shape(10)
        T = axial_coolant_temp(T_COOLANT_INLET, shape, NOMINAL_POWER_W, 1.0)
        assert np.all(np.diff(T) >= 0)

    def test_inlet_bound(self):
        shape = cosine_power_shape(10)
        T = axial_coolant_temp(T_COOLANT_INLET, shape, NOMINAL_POWER_W, 1.0)
        assert T[0] > T_COOLANT_INLET, "Bottom node must be hotter than inlet"


# ── 3. axial_coolant_temp — nominal temperatures ──────────────────────────────

class TestAxialCoolantTempNominal:
    """At 100 % power, full flow:
       inlet  ≈ 543 K (270 °C),
       outlet ≈ 598 K (325 °C).
    Both values follow from NOMINAL_FLOW_RATE * CP_COOL ≈ 54.6 MW/K.
    """

    def test_inlet_near_543K(self):
        shape = flat_power_shape(10)
        T = axial_coolant_temp(T_COOLANT_INLET, shape, NOMINAL_POWER_W, 1.0)
        # Bottom node = t_in + dT[0]; for flat shape dT[0] = ΔT_total/10
        assert T[0] > T_COOLANT_INLET
        assert T[0] < T_COOLANT_INLET + 10.0, "Bottom node too hot for flat shape"

    def test_outlet_near_598K(self):
        shape = flat_power_shape(10)
        T = axial_coolant_temp(T_COOLANT_INLET, shape, NOMINAL_POWER_W, 1.0)
        outlet = T[-1]
        # Total rise = NOMINAL_POWER_W / (NOMINAL_FLOW_RATE * CP_COOL) ≈ 54.9 K
        expected_rise = NOMINAL_POWER_W / (NOMINAL_FLOW_RATE * 5900.0)
        expected_outlet = T_COOLANT_INLET + expected_rise
        assert abs(outlet - expected_outlet) < 1.0, (
            f"Outlet {outlet:.1f} K, expected ≈ {expected_outlet:.1f} K"
        )
        # Physical sanity: hot-leg should be around 598 K (325 °C)
        assert 590.0 < outlet < 610.0, f"Outlet {outlet:.1f} K out of physical range"

    def test_zero_power_gives_uniform_tin(self):
        shape = cosine_power_shape(10)
        T = axial_coolant_temp(T_COOLANT_INLET, shape, 0.0, 1.0)
        np.testing.assert_allclose(T, T_COOLANT_INLET, atol=1e-10)


# ── 4. axial_fuel_temp — fuel always hotter than coolant ─────────────────────

class TestAxialFuelTemp:
    def test_fuel_hotter_than_coolant_every_node(self):
        shape = cosine_power_shape(10)
        t_cool = axial_coolant_temp(T_COOLANT_INLET, shape, NOMINAL_POWER_W, 1.0)
        htc = np.full(10, 30_000.0)
        t_fuel = axial_fuel_temp(t_cool, shape, NOMINAL_POWER_W, htc)
        assert np.all(t_fuel > t_cool), "All fuel nodes must exceed local coolant temp"

    def test_fuel_hotter_at_center(self):
        shape = cosine_power_shape(10)
        t_cool = axial_coolant_temp(T_COOLANT_INLET, shape, NOMINAL_POWER_W, 1.0)
        htc = np.full(10, 30_000.0)
        t_fuel = axial_fuel_temp(t_cool, shape, NOMINAL_POWER_W, htc)
        margin = t_fuel - t_cool
        # Center nodes have highest power_shape → largest margin
        assert margin[4:6].mean() > margin[0], "Center fuel-coolant margin must exceed edge"

    def test_zero_power_fuel_equals_coolant(self):
        shape = flat_power_shape(10)
        t_cool = np.linspace(543.0, 598.0, 10)
        htc = np.full(10, 30_000.0)
        t_fuel = axial_fuel_temp(t_cool, shape, 0.0, htc)
        np.testing.assert_allclose(t_fuel, t_cool, atol=1e-10)


# ── 5. step_thermal() backward compatibility ──────────────────────────────────

class TestStepThermalBackwardCompat:
    """Verify that Phase 1 behavioral guarantees still hold for step_thermal()."""

    def _run(self, t_fuel, t_cool, power, flow, decay, dt, n):
        for _ in range(n):
            t_fuel, t_cool = step_thermal(t_fuel, t_cool, power, flow, decay, dt)
        return t_fuel, t_cool

    def test_fixed_point_at_nominal(self):
        tf, tc = self._run(T_REF_FUEL, T_REF_COOLANT, NOMINAL_POWER_W, 1.0, 0.0, 0.1, 1000)
        assert abs(tf - T_REF_FUEL) < 0.1
        assert abs(tc - T_REF_COOLANT) < 0.1

    def test_converges_from_cold(self):
        tf, tc = self._run(T_COOLANT_INLET, T_COOLANT_INLET, NOMINAL_POWER_W, 1.0, 0.0, 0.1, 3000)
        assert abs(tf - T_REF_FUEL) < 5.0
        assert abs(tc - T_REF_COOLANT) < 5.0

    def test_fuel_rises_at_zero_flow(self):
        dt = 0.1
        tf, tc = T_REF_FUEL, T_REF_COOLANT
        history = [tf]
        for _ in range(500):
            tf, tc = step_thermal(tf, tc, NOMINAL_POWER_W, 0.0, 0.0, dt)
            history.append(tf)
        assert np.all(np.diff(history) >= -1e-9), "Fuel must not decrease at zero flow"
        assert history[-1] > T_REF_FUEL + 10.0

    def test_decay_heat_heats_both(self):
        tf, tc = self._run(T_COOLANT_INLET, T_COOLANT_INLET, 0.0, 0.0, 1e7, 0.1, 500)
        assert tf > T_COOLANT_INLET + 1.0
        assert tc > T_COOLANT_INLET + 1.0

    def test_higher_decay_gives_higher_temps(self):
        tf_lo, tc_lo = self._run(T_COOLANT_INLET, T_COOLANT_INLET, 0.0, 0.0, 5e6, 0.1, 500)
        tf_hi, tc_hi = self._run(T_COOLANT_INLET, T_COOLANT_INLET, 0.0, 0.0, 5e7, 0.1, 500)
        assert tf_hi > tf_lo
        assert tc_hi > tc_lo
