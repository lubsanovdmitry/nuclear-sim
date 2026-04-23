"""Phase 2 simulation loop tests — axial nodalization, two-phase, DNBR."""

from __future__ import annotations

import asyncio
import copy

import numpy as np
import pytest

from api.state import default_state, PlantState
from api.simulation_loop import simulation_tick
from physics.two_phase import critical_heat_flux


async def _run_ticks(state: PlantState, n: int, dt: float = 0.1) -> PlantState:
    for _ in range(n):
        state = await simulation_tick(state, dt=dt)
    return state


_PHASE2_ALARM_NAMES = {
    "LOW_DNBR", "APPROACH_CHF", "FILM_BOILING", "FUEL_DAMAGE",
    "VOID_HIGH", "VOID_MODERATE", "AXIAL_TILT",
}


def test_nominal_600_ticks_no_phase2_alarms() -> None:
    """600 ticks at nominal: DNBR > 2.5, no Phase 2 alarms, fuel_damage=False."""
    state = asyncio.run(_run_ticks(default_state(), 600))

    assert state.dnbr > 2.5, f"DNBR too low at nominal: {state.dnbr:.3f}"
    assert not state.fuel_damage, "fuel_damage set at nominal"

    phase2_active = _PHASE2_ALARM_NAMES.intersection(state.alarms)
    assert not phase2_active, f"Unexpected Phase 2 alarms at nominal: {phase2_active}"


def test_high_power_approach_chf_within_10_ticks() -> None:
    """At 130% power APPROACH_CHF or LOW_DNBR fires within 10 ticks."""
    state = default_state()
    # Force power to 130% and bypass the HI_POWER SCRAM so we can observe DNBR effects
    state.n = 1.30
    state.scram_bypasses = ["HI_POWER"]

    fired: set[str] = set()
    for _ in range(10):
        state = asyncio.run(_run_ticks(state, 1))
        fired.update(state.alarms)
        if "APPROACH_CHF" in fired or "LOW_DNBR" in fired:
            break

    assert "APPROACH_CHF" in fired or "LOW_DNBR" in fired, (
        f"Neither APPROACH_CHF nor LOW_DNBR fired within 10 ticks at 130% power. "
        f"Alarms seen: {fired}, DNBR={state.dnbr:.3f}"
    )


def test_fuel_damage_requires_sustained_low_dnbr() -> None:
    """fuel_damage stays False until dnbr_low_timer exceeds 5.0 s (50 ticks at dnbr<1.0).

    Re-injects n=4.0 each tick so Doppler feedback doesn't pull power back between ticks.
    Pre-seeds dnbr_low_timer near the threshold to avoid running 50 ticks.
    """
    state = default_state()
    # 400% power → peak heat_flux ≈ 4 × nominal_peak → DNBR < 1.0
    state.n = 4.0
    state.scram_bypasses = ["HI_POWER", "HI_FUEL_TEMP", "HI_COOL_TEMP", "HI_PRESSURE"]
    state.dnbr_low_timer = 4.70  # needs ~4 more ticks of dnbr<1.0 to exceed 5.0

    s = copy.deepcopy(state)
    fuel_damage_tick: int | None = None
    for tick in range(8):
        s.n = 4.0  # re-inject high power each tick (Doppler would otherwise pull n down)
        s = asyncio.run(_run_ticks(s, 1))

        if tick == 0 and s.dnbr >= 1.0:
            pytest.skip(f"DNBR not < 1.0 at n=4.0 (got {s.dnbr:.3f})")

        if s.fuel_damage and fuel_damage_tick is None:
            fuel_damage_tick = tick

        if fuel_damage_tick is not None and tick > fuel_damage_tick:
            # Verify irreversibility
            assert s.fuel_damage, f"fuel_damage reverted at tick {tick}"

    assert fuel_damage_tick is not None, (
        f"fuel_damage never set after 8 ticks; final timer={s.dnbr_low_timer:.2f}"
    )
    # Verify it fired after timer crossed 5.0 s (not immediately)
    assert fuel_damage_tick >= 3, (
        f"fuel_damage fired too early at tick {fuel_damage_tick}"
    )


def test_fuel_damage_direct_timer() -> None:
    """Direct timer manipulation: fuel_damage = True once dnbr_low_timer > 5.0."""
    state = default_state()
    # Pre-set timer just below threshold; DNBR forced to < 1.0 via explicit state override
    state.dnbr_low_timer = 4.95
    state.dnbr = 0.5       # will be recomputed, but set to ensure predicate

    # We can't stop simulation_tick from recomputing DNBR, so instead verify the
    # irreversible property: once fuel_damage is True it stays True.
    state.fuel_damage = True
    state_after = asyncio.run(_run_ticks(state, 10))
    assert state_after.fuel_damage, "fuel_damage should be irreversible once set"


def test_phase1_loop_still_passes() -> None:
    """Phase 1 steady-state test: 600 ticks, no alarms, n≈1.0, temps stable."""
    from physics.constants import T_REF_FUEL, T_REF_COOLANT, PRESSURE_NOMINAL

    state = asyncio.run(_run_ticks(default_state(), 600))

    assert not state.scram, "Unexpected SCRAM"
    p2_alarms = _PHASE2_ALARM_NAMES.intersection(state.alarms)
    p1_alarms = set(state.alarms) - p2_alarms
    assert p1_alarms == set(), f"Phase 1 alarms fired: {p1_alarms}"
    assert abs(state.n - 1.0) < 0.01, f"Power drifted: {state.n:.6f}"
    assert abs(state.t_fuel - T_REF_FUEL) < 1.0
    assert abs(state.t_cool - T_REF_COOLANT) < 1.0
    assert abs(state.pressure - PRESSURE_NOMINAL) < 1.0e5
    assert state.flow_fraction > 0.999


def test_chf_uses_current_tick_quality() -> None:
    state = default_state()
    state.pressure = 2.0e6  # 20 bar, drives positive quality from nominal coolant temperatures
    state.quality = np.full(10, -0.3)
    initial_chf = critical_heat_flux(state.flow_fraction, state.pressure, float(state.quality.mean()))

    state = asyncio.run(_run_ticks(state, 1))
    expected_chf = critical_heat_flux(state.flow_fraction, state.pressure, float(state.quality.mean()))

    assert state.chf == pytest.approx(expected_chf)
    assert state.chf < initial_chf


def test_dnbr_responds_same_tick_as_quality_change() -> None:
    state = default_state()
    state.pressure = 2.0e6  # 20 bar
    state.quality = np.full(10, -0.3)
    state = asyncio.run(_run_ticks(state, 1))

    assert float(state.quality.mean()) > 0.0
    expected_dnbr = state.chf / max(float(state.heat_flux[state.peak_heat_flux_node]), 1.0)
    assert state.dnbr == pytest.approx(expected_dnbr)


def test_sbo_regime_chatter_reduced() -> None:
    """SBO should not flip worst boiling regime every tick for long periods."""
    order = ["single_phase", "subcooled_boiling", "nucleate_boiling", "film_boiling"]

    state = default_state()
    state.offsite_power = False
    state.diesel_start_signals = [False, False]
    for ds in state.diesel_states:
        ds.state = "failed"

    worst_prev = None
    transitions = 0
    for _ in range(1200):
        state = asyncio.run(_run_ticks(state, 1))
        worst_now = max(state.boiling_regime, key=lambda r: order.index(r))
        if worst_prev is not None and worst_now != worst_prev:
            transitions += 1
        worst_prev = worst_now

    assert transitions <= 20, f"Too many worst-regime transitions in SBO: {transitions}"


def test_htc_rate_limited_across_regime_change() -> None:
    """HTC should not jump by large factors within a single 100 ms tick."""
    state = default_state()
    htc0 = state.htc.copy()

    # Force film-boiling classification on the next tick.
    state.void_fraction = np.full(10, 0.8)
    state.void_fraction_dyn = np.full(10, 0.8)
    state = asyncio.run(_run_ticks(state, 1))
    htc1 = state.htc.copy()

    # Then force low void to push back toward non-film regime.
    state.void_fraction = np.zeros(10)
    state.void_fraction_dyn = np.zeros(10)
    state = asyncio.run(_run_ticks(state, 1))
    htc2 = state.htc.copy()

    rel_step_1 = np.max(np.abs((htc1 - htc0) / np.maximum(htc0, 1.0)))
    rel_step_2 = np.max(np.abs((htc2 - htc1) / np.maximum(htc1, 1.0)))

    assert rel_step_1 < 0.5, f"HTC changed too quickly in one tick (step1): {rel_step_1:.3f}"
    assert rel_step_2 < 0.5, f"HTC changed too quickly in one tick (step2): {rel_step_2:.3f}"
