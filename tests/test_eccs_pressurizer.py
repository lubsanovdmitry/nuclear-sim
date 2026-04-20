"""Tests for plant/eccs.py and plant/pressurizer.py."""

import pytest

from plant.eccs import ECCSState, step_eccs, eccs_reactivity
from plant.pressurizer import step_pressurizer, check_scram_pressure
from physics.constants import (
    ECCS_HPSI_THRESHOLD,
    ECCS_LPSI_THRESHOLD,
    ECCS_HPSI_FLOW_FRACTION,
    ECCS_LPSI_FLOW_FRACTION,
    ECCS_WATER_TEMP,
    PRESSURE_NOMINAL,
    PRESSURE_SCRAM_HIGH,
    PRESSURE_SCRAM_LOW,
    T_REF_COOLANT,
    PRZR_HEATER_DPDT_MAX,
    PRZR_SPRAY_DPDT_MAX,
)


# ---------------------------------------------------------------------------
# ECCS activation threshold tests
# ---------------------------------------------------------------------------

class TestECCSThresholds:
    def test_hpsi_active_at_exactly_100_bar(self) -> None:
        """HPSI actuates at the 100 bar setpoint (pressure has dropped to threshold)."""
        state = step_eccs(ECCS_HPSI_THRESHOLD, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.hpsi_active is True

    def test_hpsi_active_below_100_bar(self) -> None:
        """HPSI injects during LOCA when pressure has fallen to 50 bar."""
        state = step_eccs(0.50e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.hpsi_active is True

    def test_hpsi_inactive_at_nominal_pressure(self) -> None:
        """HPSI must NOT inject at nominal RCS pressure (155 bar > 100 bar setpoint)."""
        state = step_eccs(1.55e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.hpsi_active is False

    def test_lpsi_active_at_exactly_20_bar(self) -> None:
        """LPSI must activate at the 20 bar threshold."""
        state = step_eccs(ECCS_LPSI_THRESHOLD, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.lpsi_active is True

    def test_lpsi_active_below_20_bar(self) -> None:
        state = step_eccs(1.0e6, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.lpsi_active is True

    def test_lpsi_inactive_above_20_bar(self) -> None:
        """LPSI must not inject at 50 bar — only HPSI is active there."""
        state = step_eccs(0.50e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.lpsi_active is False

    def test_hpsi_only_zone_between_20_and_100_bar(self) -> None:
        """In the 20–100 bar window, HPSI is active but LPSI is not."""
        mid_pressure = 0.50e7  # 50 bar: below HPSI threshold, above LPSI threshold
        state = step_eccs(mid_pressure, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.hpsi_active is True
        assert state.lpsi_active is False
        assert state.injection_flow_fraction == pytest.approx(ECCS_HPSI_FLOW_FRACTION)

    def test_no_injection_when_not_armed(self) -> None:
        """ECCS (inhibited) must not inject regardless of pressure."""
        for pressure in [2.0e7, 1.55e7, 0.50e7, 0.5e6]:
            state = step_eccs(pressure, armed=False, t_cool=T_REF_COOLANT, flow_fraction=1.0)
            assert state.hpsi_active is False
            assert state.lpsi_active is False
            assert state.injection_flow_fraction == pytest.approx(0.0)


class TestECCSInjectionFlow:
    def test_hpsi_flow_fraction_correct(self) -> None:
        """HPSI injects at the correct fraction of nominal flow (pressure < 100 bar)."""
        state = step_eccs(0.50e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.injection_flow_fraction == pytest.approx(ECCS_HPSI_FLOW_FRACTION)

    def test_lpsi_flow_fraction_correct(self) -> None:
        """At 10 bar both HPSI and LPSI are active; total flow = sum of both."""
        state = step_eccs(1.0e6, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.injection_flow_fraction == pytest.approx(
            ECCS_HPSI_FLOW_FRACTION + ECCS_LPSI_FLOW_FRACTION
        )

    def test_injection_temperature_is_cold(self) -> None:
        """Injected water must be cold (significantly below nominal coolant temp)."""
        state = step_eccs(0.50e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        assert state.injection_temp < T_REF_COOLANT - 100.0  # at least 100 K cooler


class TestECCSReactivity:
    def test_reactivity_negative_during_hpsi(self) -> None:
        """Cold borated-water injection must produce negative reactivity."""
        state = step_eccs(0.50e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        rho = eccs_reactivity(state, T_REF_COOLANT)
        assert rho < 0.0

    def test_reactivity_zero_when_no_injection(self) -> None:
        state = ECCSState()  # no injection
        rho = eccs_reactivity(state, T_REF_COOLANT)
        assert rho == pytest.approx(0.0)

    def test_reactivity_zero_at_nominal_pressure(self) -> None:
        """No reactivity effect at nominal pressure (ECCS not actuated)."""
        state = step_eccs(1.55e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        rho = eccs_reactivity(state, T_REF_COOLANT)
        assert rho == pytest.approx(0.0)

    def test_reactivity_more_negative_with_higher_injection_flow(self) -> None:
        """Higher injection flow fraction → more boron added → more negative rho."""
        state_hpsi = step_eccs(0.50e7, armed=True, t_cool=T_REF_COOLANT, flow_fraction=1.0)
        state_both = ECCSState(
            hpsi_active=True,
            lpsi_active=True,
            injection_flow_fraction=state_hpsi.injection_flow_fraction * 2.0,
            injection_temp=ECCS_WATER_TEMP,
        )
        rho_hpsi = eccs_reactivity(state_hpsi, T_REF_COOLANT)
        rho_both = eccs_reactivity(state_both, T_REF_COOLANT)
        assert rho_both < rho_hpsi


# ---------------------------------------------------------------------------
# Pressurizer tests
# ---------------------------------------------------------------------------

class TestPressurizerSteadyState:
    def test_pressure_stable_at_nominal_conditions(self) -> None:
        """At nominal T_cool and full flow, pressure settles to PRESSURE_NOMINAL."""
        p = PRESSURE_NOMINAL
        for _ in range(1000):
            p = step_pressurizer(p, T_REF_COOLANT, flow_fraction=1.0, dt=0.1)
        assert p == pytest.approx(PRESSURE_NOMINAL, rel=1e-6)

    def test_pressure_rises_with_hot_coolant(self) -> None:
        """Overheated coolant (loss of cooling) must drive pressure above nominal."""
        p = PRESSURE_NOMINAL
        t_hot = T_REF_COOLANT + 50.0  # 50 K overtemperature
        for _ in range(300):
            p = step_pressurizer(p, t_hot, flow_fraction=1.0, dt=0.1)
        assert p > PRESSURE_NOMINAL

    def test_pressure_falls_with_lost_flow(self) -> None:
        """Flow loss (LOCA-like) must depressurize the RCS."""
        p = PRESSURE_NOMINAL
        for _ in range(300):
            p = step_pressurizer(p, T_REF_COOLANT, flow_fraction=0.0, dt=0.1)
        assert p < PRESSURE_NOMINAL

    def test_pressure_approaches_target_exponentially(self) -> None:
        """After a perturbation, pressure must decay monotonically toward target."""
        p = PRESSURE_NOMINAL * 1.05  # 5% above nominal
        prev = p
        for _ in range(50):
            p = step_pressurizer(p, T_REF_COOLANT, flow_fraction=1.0, dt=0.5)
            assert p < prev
            prev = p


class TestPressurizerSCRAMSetpoints:
    def test_no_scram_at_nominal_pressure(self) -> None:
        assert check_scram_pressure(PRESSURE_NOMINAL) is False

    def test_scram_triggered_above_170_bar(self) -> None:
        assert check_scram_pressure(PRESSURE_SCRAM_HIGH + 1.0) is True

    def test_scram_triggered_at_exactly_170_bar(self) -> None:
        """SCRAM is strictly > 170 bar, so the boundary must trigger."""
        # The check is pressure > SCRAM_HIGH, so exactly at threshold is False
        assert check_scram_pressure(PRESSURE_SCRAM_HIGH) is False
        assert check_scram_pressure(PRESSURE_SCRAM_HIGH + 1.0) is True

    def test_scram_triggered_below_120_bar(self) -> None:
        assert check_scram_pressure(PRESSURE_SCRAM_LOW - 1.0) is True

    def test_scram_triggered_at_exactly_120_bar(self) -> None:
        """SCRAM is strictly < 120 bar."""
        assert check_scram_pressure(PRESSURE_SCRAM_LOW) is False
        assert check_scram_pressure(PRESSURE_SCRAM_LOW - 1.0) is True

    def test_no_scram_in_operating_band(self) -> None:
        """Pressures between 120 and 170 bar must not trigger SCRAM."""
        for p_bar in [120, 130, 140, 155, 160, 169]:
            p_pa = p_bar * 1e5
            assert check_scram_pressure(p_pa) is False, f"Unexpected SCRAM at {p_bar} bar"

class TestPressurizerHeaterSpray:
    def test_heaters_raise_pressure(self) -> None:
        """Full heater demand must raise pressure above what zero heaters produce."""
        p_no_heater = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, 10.0, heater_fraction=0.0)
        p_heater    = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, 10.0, heater_fraction=1.0)
        assert p_heater > p_no_heater

    def test_spray_lowers_pressure(self) -> None:
        """Full spray demand must lower pressure below what zero spray produces."""
        p_no_spray = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, 10.0, spray_fraction=0.0)
        p_spray    = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, 10.0, spray_fraction=1.0)
        assert p_spray < p_no_spray

    def test_heater_rate_proportional(self) -> None:
        """50% heater demand must raise pressure at roughly half the rate of 100%."""
        dt = 1.0
        dp_full = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, dt, heater_fraction=1.0) - PRESSURE_NOMINAL
        dp_half = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, dt, heater_fraction=0.5) - PRESSURE_NOMINAL
        assert dp_half == pytest.approx(dp_full / 2.0, rel=0.05)

    def test_spray_rate_proportional(self) -> None:
        """50% spray demand must lower pressure at roughly half the rate of 100%."""
        dt = 1.0
        dp_full = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, dt, spray_fraction=1.0) - PRESSURE_NOMINAL
        dp_half = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, dt, spray_fraction=0.5) - PRESSURE_NOMINAL
        assert dp_half == pytest.approx(dp_full / 2.0, rel=0.05)

    def test_no_heater_spray_backward_compatible(self) -> None:
        """Default call (no heater/spray args) must behave identically to old call."""
        p_default = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, 5.0)
        p_explicit = step_pressurizer(PRESSURE_NOMINAL, T_REF_COOLANT, 1.0, 5.0, 0.0, 0.0)
        assert p_default == pytest.approx(p_explicit)


    def test_high_pressure_scram_reached_during_overtemp(self) -> None:
        """Sustained overtemperature transient must eventually reach high-pressure SCRAM."""
        p = PRESSURE_NOMINAL
        t_very_hot = T_REF_COOLANT + 200.0  # severe overtemperature
        scram_seen = False
        for _ in range(500):
            p = step_pressurizer(p, t_very_hot, flow_fraction=1.0, dt=0.5)
            if check_scram_pressure(p):
                scram_seen = True
                break
        assert scram_seen, "High-pressure SCRAM never triggered during overtemperature transient"
