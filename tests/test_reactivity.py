"""Tests for reactivity component model."""

import numpy as np
import pytest

from physics.reactivity import (
    rod_reactivity,
    doppler_reactivity,
    moderator_reactivity,
    boron_reactivity,
    void_reactivity,
    compute_reactivity,
)
from physics.xenon import xenon_equilibrium, xenon_reactivity, rk4_step_xenon
from physics.constants import (
    ROD_WORTH_MIN, ROD_WORTH_MAX,
    T_REF_FUEL, T_REF_COOLANT,
)
from api.state import PlantState


# ---------------------------------------------------------------------------
# rod_reactivity
# ---------------------------------------------------------------------------

class TestRodReactivity:
    def test_all_rods_inserted_equals_worth_min(self):
        """All rods at 0% (fully inserted) → ROD_WORTH_MIN (-0.02)."""
        rho = rod_reactivity([0.0, 0.0, 0.0, 0.0])
        assert abs(rho - ROD_WORTH_MIN) < 1e-10

    def test_all_rods_withdrawn_equals_worth_max(self):
        """All rods at 100% (fully withdrawn) → ROD_WORTH_MAX."""
        rho = rod_reactivity([100.0, 100.0, 100.0, 100.0])
        assert abs(rho - ROD_WORTH_MAX) < 1e-10

    def test_half_withdrawn(self):
        """50% average position → midpoint worth."""
        rho = rod_reactivity([50.0, 50.0, 50.0, 50.0])
        expected = ROD_WORTH_MIN + 0.5 * (ROD_WORTH_MAX - ROD_WORTH_MIN)
        assert abs(rho - expected) < 1e-10

    def test_large_negative_when_fully_inserted(self):
        """Fully inserted rods produce clearly negative reactivity."""
        rho = rod_reactivity([0.0, 0.0, 0.0, 0.0])
        assert rho < -0.01

    def test_uses_average_of_banks(self):
        """Mixed positions → average of bank positions used."""
        rho_mixed = rod_reactivity([0.0, 100.0])
        rho_mid = rod_reactivity([50.0, 50.0])
        assert abs(rho_mixed - rho_mid) < 1e-10

    def test_numpy_array_input(self):
        """Accepts numpy arrays in addition to lists."""
        rho = rod_reactivity(np.array([0.0, 0.0, 0.0, 0.0]))
        assert abs(rho - ROD_WORTH_MIN) < 1e-10


# ---------------------------------------------------------------------------
# doppler_reactivity
# ---------------------------------------------------------------------------

class TestDopplerReactivity:
    def test_zero_at_reference_temperature(self):
        """Doppler feedback is zero at T_REF_FUEL."""
        assert doppler_reactivity(T_REF_FUEL) == 0.0

    def test_higher_fuel_temp_gives_more_negative_rho(self):
        """Increasing fuel temperature → more negative reactivity (negative feedback)."""
        rho_ref = doppler_reactivity(T_REF_FUEL)
        rho_hot = doppler_reactivity(T_REF_FUEL + 100.0)
        assert rho_hot < rho_ref

    def test_lower_fuel_temp_gives_positive_rho(self):
        """Fuel below reference temperature gives positive reactivity."""
        rho = doppler_reactivity(T_REF_FUEL - 100.0)
        assert rho > 0.0

    def test_linearity(self):
        """Doppler feedback is linear: doubling ΔT doubles Δrho."""
        delta = 200.0
        rho_delta = doppler_reactivity(T_REF_FUEL + delta)
        rho_2delta = doppler_reactivity(T_REF_FUEL + 2 * delta)
        assert abs(rho_2delta / rho_delta - 2.0) < 1e-9

    def test_magnitude_reasonable(self):
        """100 K rise gives ~0.0025 dk/k of negative feedback."""
        rho = doppler_reactivity(T_REF_FUEL + 100.0)
        assert -0.005 < rho < -0.001


# ---------------------------------------------------------------------------
# moderator_reactivity
# ---------------------------------------------------------------------------

class TestModeratorReactivity:
    def test_zero_at_reference_temperature(self):
        """Moderator feedback is zero at T_REF_COOLANT."""
        assert moderator_reactivity(T_REF_COOLANT) == 0.0

    def test_higher_coolant_temp_gives_more_negative_rho(self):
        """Increasing coolant temperature → more negative reactivity."""
        rho_ref = moderator_reactivity(T_REF_COOLANT)
        rho_hot = moderator_reactivity(T_REF_COOLANT + 50.0)
        assert rho_hot < rho_ref

    def test_lower_coolant_temp_gives_positive_rho(self):
        """Coolant below reference gives positive moderator reactivity."""
        rho = moderator_reactivity(T_REF_COOLANT - 50.0)
        assert rho > 0.0

    def test_magnitude_reasonable(self):
        """50 K rise gives ~0.0075 dk/k of negative feedback (ALPHA_MODERATOR = -1.5e-4)."""
        rho = moderator_reactivity(T_REF_COOLANT + 50.0)
        assert -0.015 < rho < -0.005


# ---------------------------------------------------------------------------
# boron_reactivity
# ---------------------------------------------------------------------------

class TestBoronReactivity:
    def test_zero_at_zero_ppm(self):
        assert boron_reactivity(0.0) == 0.0

    def test_negative_for_positive_boron(self):
        assert boron_reactivity(500.0) < 0.0

    def test_linear_scaling(self):
        assert abs(boron_reactivity(1000.0) / boron_reactivity(500.0) - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_reactivity (integration test using PlantState)
# ---------------------------------------------------------------------------

class TestComputeReactivity:
    def test_nominal_state_near_zero(self):
        """At normal operating conditions (75% rods, equilibrium xenon, 1000 ppm boron),
        total reactivity is near zero — ROD_WORTH_MAX chosen to make this exact."""
        from physics.xenon import xenon_equilibrium
        from physics.constants import INITIAL_BORON_PPM
        _, Xe_eq = xenon_equilibrium(1.0)
        state = PlantState(
            rod_positions=[75.0, 75.0, 75.0, 75.0],
            t_fuel=T_REF_FUEL,
            t_cool=T_REF_COOLANT,
            xenon=Xe_eq,
            boron_ppm=INITIAL_BORON_PPM,
        )
        rho = compute_reactivity(state)
        assert abs(rho) < 1e-3, f"Expected near-zero reactivity at steady state, got {rho:.6f}"

    def test_fresh_core_small_positive_reactivity(self):
        """Fresh core (no xenon) at 100% rods has small positive reactivity = ROD_WORTH_MAX."""
        state = PlantState(
            rod_positions=[100.0, 100.0, 100.0, 100.0],
            t_fuel=T_REF_FUEL,
            t_cool=T_REF_COOLANT,
            t_cool_axial=np.full(10, T_REF_COOLANT),  # zero moderator feedback
            xenon=0.0,
            boron_ppm=0.0,
        )
        rho = compute_reactivity(state)
        assert abs(rho - ROD_WORTH_MAX) < 1e-10

    def test_all_rods_inserted_large_negative(self):
        """All rods inserted → total reactivity is large and negative."""
        state = PlantState(
            rod_positions=[0.0, 0.0, 0.0, 0.0],
            t_fuel=T_REF_FUEL,
            t_cool=T_REF_COOLANT,
            xenon=0.0,
            boron_ppm=0.0,
        )
        rho = compute_reactivity(state)
        assert rho < -0.015, f"Expected large negative rho, got {rho:.4f}"

    def test_doppler_feedback_increases_with_temperature(self):
        """Higher fuel temperature → more negative total reactivity."""
        base = PlantState(t_fuel=T_REF_FUEL, rod_positions=[100.0]*4)
        hot = PlantState(t_fuel=T_REF_FUEL + 200.0, rod_positions=[100.0]*4)
        assert compute_reactivity(hot) < compute_reactivity(base)

    def test_moderator_feedback_increases_with_temperature(self):
        """Higher coolant temperature → more negative total reactivity."""
        base = PlantState(
            t_cool=T_REF_COOLANT,
            t_cool_axial=np.full(10, T_REF_COOLANT),
            rod_positions=[100.0]*4,
        )
        hot = PlantState(
            t_cool=T_REF_COOLANT + 50.0,
            t_cool_axial=np.full(10, T_REF_COOLANT + 50.0),
            rod_positions=[100.0]*4,
        )
        assert compute_reactivity(hot) < compute_reactivity(base)

    def test_xenon_peak_reactivity_below_threshold(self):
        """Xenon at post-shutdown peak gives rho_xenon < -0.002."""
        I, Xe = xenon_equilibrium(n=1.0)
        # Integrate 12 hours post-shutdown to reach near-peak
        dt = 300.0
        for _ in range(int(12 * 3600 / dt)):
            I, Xe = rk4_step_xenon(I, Xe, n=0.0, dt=dt)

        rho_xe = xenon_reactivity(Xe)
        assert rho_xe < -0.002, (
            f"Peak xenon reactivity {rho_xe:.4f} not below -0.002"
        )

        from physics.constants import INITIAL_BORON_PPM
        state = PlantState(
            rod_positions=[75.0]*4,
            t_fuel=T_REF_FUEL,
            t_cool=T_REF_COOLANT,
            xenon=Xe,
            boron_ppm=INITIAL_BORON_PPM,
        )
        rho_total = compute_reactivity(state)
        assert rho_total < -0.005, (
            f"Peak xenon + nominal boron should keep normal position subcritical, got {rho_total:.4f}"
        )

    def test_boron_adds_negative_reactivity(self):
        """Adding boron reduces total reactivity."""
        no_boron = PlantState(boron_ppm=0.0, rod_positions=[100.0]*4)
        with_boron = PlantState(boron_ppm=1000.0, rod_positions=[100.0]*4)
        assert compute_reactivity(with_boron) < compute_reactivity(no_boron)

    def test_components_sum_correctly(self):
        """compute_reactivity equals manual sum of components."""
        from physics.reactivity import (
            rod_reactivity, doppler_reactivity, moderator_reactivity,
            boron_reactivity, void_reactivity,
        )
        state = PlantState(
            rod_positions=[75.0, 75.0, 75.0, 75.0],
            t_fuel=T_REF_FUEL + 50.0,
            t_cool=T_REF_COOLANT + 10.0,
            xenon=1.0e21,
            boron_ppm=200.0,
        )
        t_cool_eff = float(np.average(state.t_cool_axial, weights=state.axial_power_shape))
        expected = (
            rod_reactivity(state.rod_positions)
            + doppler_reactivity(state.t_fuel)
            + moderator_reactivity(t_cool_eff)
            + xenon_reactivity(state.xenon)
            + boron_reactivity(state.boron_ppm)
            + void_reactivity(state.void_fraction, state.axial_power_shape)
        )
        assert abs(compute_reactivity(state) - expected) < 1e-15


# ---------------------------------------------------------------------------
# void_reactivity (Phase 2)
# ---------------------------------------------------------------------------

class TestVoidReactivity:
    def test_zero_void_gives_zero_rho(self):
        """PWR nominal — all zeros → no void feedback."""
        from physics.axial import cosine_power_shape
        shape = cosine_power_shape(10)
        rho = void_reactivity(np.zeros(10), shape, reactor_type='PWR')
        assert rho == 0.0

    def test_bwr_40pct_void_approx_minus_006(self):
        """BWR at uniform 40% void → alpha_avg = 0.4 → rho ≈ -0.06."""
        shape = np.ones(10)
        rho = void_reactivity(np.full(10, 0.4), shape, reactor_type='BWR')
        assert abs(rho - (-0.06)) < 1e-10

    def test_pwr_void_smaller_than_bwr(self):
        """PWR void coeff is smaller in magnitude than BWR."""
        shape = np.ones(10)
        alpha = np.full(10, 0.3)
        rho_pwr = void_reactivity(alpha, shape, reactor_type='PWR')
        rho_bwr = void_reactivity(alpha, shape, reactor_type='BWR')
        assert abs(rho_pwr) < abs(rho_bwr)

    def test_power_weighted_average_used(self):
        """Higher void in high-power nodes counts more than edge nodes."""
        from physics.axial import cosine_power_shape
        shape = cosine_power_shape(10)
        # Void concentrated at center (high-power) nodes
        void_center = np.zeros(10)
        void_center[4:6] = 0.5
        # Void concentrated at edge (low-power) nodes
        void_edge = np.zeros(10)
        void_edge[0] = 0.5
        void_edge[-1] = 0.5
        rho_center = void_reactivity(void_center, shape)
        rho_edge = void_reactivity(void_edge, shape)
        # Center nodes are higher power → larger magnitude feedback
        assert abs(rho_center) > abs(rho_edge)


# ---------------------------------------------------------------------------
# PlantState Phase 2 fields
# ---------------------------------------------------------------------------

class TestPlantStatePhase2Fields:
    def test_default_state_has_all_phase2_fields(self):
        """default_state() returns PlantState with all Phase 2 arrays and scalars."""
        from api.state import default_state
        s = default_state()
        assert hasattr(s, 't_fuel_axial')
        assert s.t_fuel_axial.shape == (10,)
        assert hasattr(s, 't_cool_axial')
        assert s.t_cool_axial.shape == (10,)
        assert hasattr(s, 'void_fraction')
        assert s.void_fraction.shape == (10,)
        assert hasattr(s, 'quality')
        assert s.quality.shape == (10,)
        assert hasattr(s, 'heat_flux')
        assert s.heat_flux.shape == (10,)
        assert hasattr(s, 'htc')
        assert s.htc.shape == (10,)
        assert hasattr(s, 'boiling_regime')
        assert len(s.boiling_regime) == 10
        assert hasattr(s, 'axial_power_shape')
        assert s.axial_power_shape.shape == (10,)
        assert hasattr(s, 'dnbr')
        assert s.dnbr == 2.5
        assert hasattr(s, 'chf')
        assert s.chf == 1.5e6
        assert hasattr(s, 'peak_heat_flux_node')
        assert s.peak_heat_flux_node == 5
        assert hasattr(s, 'fuel_damage')
        assert s.fuel_damage is False
        assert hasattr(s, 'film_boiling_nodes')
        assert s.film_boiling_nodes == []
        assert hasattr(s, 'loca_active')
        assert s.loca_active is False
        assert hasattr(s, 'loca_break_size')
        assert s.loca_break_size == 0.0
        assert hasattr(s, 'dnbr_low_timer')
        assert s.dnbr_low_timer == 0.0

    def test_default_void_fraction_zeros(self):
        """PWR nominal: all nodes subcooled, void = 0."""
        from api.state import default_state
        s = default_state()
        np.testing.assert_array_equal(s.void_fraction, np.zeros(10))

    def test_default_axial_power_shape_cosine(self):
        """Default axial power shape is cosine-normalized (mean=1)."""
        from api.state import default_state
        s = default_state()
        assert abs(s.axial_power_shape.mean() - 1.0) < 1e-12

    def test_compute_reactivity_near_zero_at_equilibrium(self):
        """compute_reactivity returns near-zero for default_state equilibrium."""
        from api.state import default_state
        s = default_state()
        rho = compute_reactivity(s)
        assert abs(rho) < 1e-3, f"Equilibrium rho = {rho:.6f}, expected |rho| < 1e-3"
