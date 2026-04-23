"""Tests for accident scenario trigger functions."""

import asyncio

import pytest

from api.state import default_state, PlantState
from api.simulation_loop import simulation_tick
from physics.constants import (
    ECCS_HPSI_THRESHOLD,
    PRESSURE_SCRAM_LOW,
    PRESSURE_NOMINAL,
)
from physics.xenon import xenon_equilibrium, xenon_reactivity
from accidents.loop import trigger_loop
from accidents.sbo import trigger_sbo
from accidents.loca import trigger_loca, update_loca
from accidents.rod_ejection import trigger_rod_ejection
from accidents.xenon_pit import trigger_xenon_pit


async def _run(state: PlantState, n: int, dt: float = 0.1) -> PlantState:
    for _ in range(n):
        state = await simulation_tick(state, dt=dt)
    return state


# ---------------------------------------------------------------------------
# LOOP
# ---------------------------------------------------------------------------

def test_loop_offsite_power_false() -> None:
    state = trigger_loop(default_state())
    assert state.offsite_power is False


def test_loop_diesel_start_signals() -> None:
    state = trigger_loop(default_state())
    assert state.diesel_start_signals == [True, True]


def test_loop_alarm_added() -> None:
    state = trigger_loop(default_state())
    assert "LOOP" in state.alarms


def test_loop_alarm_not_duplicated() -> None:
    s = default_state()
    s.alarms = ["LOOP"]
    state = trigger_loop(s)
    assert state.alarms.count("LOOP") == 1


def test_loop_pumps_still_on() -> None:
    """Pumps remain logically on; coastdown is driven by the simulation loop."""
    state = trigger_loop(default_state())
    assert state.pumps == [True, True, True, True]


# ---------------------------------------------------------------------------
# SBO
# ---------------------------------------------------------------------------

def test_sbo_offsite_power_false() -> None:
    state = trigger_sbo(default_state())
    assert state.offsite_power is False


def test_sbo_diesels_failed() -> None:
    state = trigger_sbo(default_state())
    assert all(d.state == "failed" for d in state.diesel_states)


def test_sbo_no_diesel_start_signals() -> None:
    state = trigger_sbo(default_state())
    assert state.diesel_start_signals == [False, False]


def test_sbo_alarm_added() -> None:
    state = trigger_sbo(default_state())
    assert "SBO" in state.alarms


# ---------------------------------------------------------------------------
# LOCA
# ---------------------------------------------------------------------------

def test_loca_pressure_below_scram_low() -> None:
    """Depressurization puts RCS below low-pressure SCRAM setpoint."""
    state = trigger_loca(default_state())
    assert state.pressure < PRESSURE_SCRAM_LOW


def test_loca_pressure_above_hpsi_threshold() -> None:
    """Initial pressure still above HPSI threshold so injection activates immediately."""
    state = trigger_loca(default_state())
    assert state.pressure >= ECCS_HPSI_THRESHOLD


def test_loca_eccs_armed() -> None:
    s = default_state()
    s.eccs_armed = False
    state = trigger_loca(s)
    assert state.eccs_armed is True


def test_loca_alarm_added() -> None:
    state = trigger_loca(default_state())
    assert "LOCA" in state.alarms


def test_loca_pressure_reduced_from_nominal() -> None:
    state = trigger_loca(default_state())
    assert state.pressure < PRESSURE_NOMINAL


def test_loca_pressure_scales_with_break_size() -> None:
    s = default_state()
    state = trigger_loca(s, break_size=0.5)
    assert state.pressure == pytest.approx(130.0e5)


# ---------------------------------------------------------------------------
# Rod Ejection
# ---------------------------------------------------------------------------

def test_rod_ejection_bank_a_fully_withdrawn() -> None:
    s = default_state()
    s.rod_positions = [50.0, 100.0, 100.0, 100.0]   # bank A partially inserted
    state = trigger_rod_ejection(s)
    assert state.rod_positions[0] == pytest.approx(100.0)


def test_rod_ejection_other_banks_unchanged() -> None:
    s = default_state()
    s.rod_positions = [50.0, 75.0, 80.0, 90.0]
    state = trigger_rod_ejection(s)
    assert state.rod_positions[1:] == [75.0, 80.0, 90.0]


def test_rod_ejection_alarm_added() -> None:
    state = trigger_rod_ejection(default_state())
    assert "ROD_EJECTION" in state.alarms


def test_rod_ejection_sets_target_position() -> None:
    state = trigger_rod_ejection(default_state())
    assert state.rod_target_positions[0] == pytest.approx(100.0)


def test_rod_ejection_sets_ejection_rho() -> None:
    s = default_state()
    s.rod_positions = [50.0, 100.0, 100.0, 100.0]
    state = trigger_rod_ejection(s)
    assert state.ejection_rho == pytest.approx(0.005)


def test_rod_ejection_at_100pct_has_zero_rho() -> None:
    s = default_state()
    s.rod_positions = [100.0, 100.0, 100.0, 100.0]
    state = trigger_rod_ejection(s)
    assert state.ejection_rho == pytest.approx(0.0)


def test_rod_ejection_inserts_positive_reactivity() -> None:
    """Ejecting a partially-inserted bank from 50% to 100% removes negative rod worth."""
    from physics.reactivity import compute_reactivity

    s_before = default_state()
    s_before.rod_positions = [50.0, 100.0, 100.0, 100.0]
    rho_before = compute_reactivity(s_before)

    s_after = trigger_rod_ejection(s_before)
    rho_after = compute_reactivity(s_after)

    assert s_after.ejection_rho > 0.0
    assert rho_after > rho_before, (
        f"Reactivity should increase after rod ejection: {rho_before:.6f} -> {rho_after:.6f}"
    )


# ---------------------------------------------------------------------------
# Xenon Pit
# ---------------------------------------------------------------------------

def test_xenon_pit_scram() -> None:
    state = trigger_xenon_pit(default_state())
    assert state.scram is True


def test_xenon_pit_zero_power() -> None:
    state = trigger_xenon_pit(default_state())
    assert state.n == pytest.approx(0.0)


def test_xenon_pit_rods_inserted() -> None:
    state = trigger_xenon_pit(default_state())
    assert state.rod_positions == [0.0, 0.0, 0.0, 0.0]


def test_xenon_pit_t_since_scram() -> None:
    state = trigger_xenon_pit(default_state())
    assert state.t_since_scram == pytest.approx(3.0 * 3600.0)


def test_xenon_pit_xenon_above_equilibrium() -> None:
    """Xe at 3h post-shutdown exceeds full-power equilibrium (classic Xe peak)."""
    _, Xe_eq = xenon_equilibrium(1.0)
    state = trigger_xenon_pit(default_state())
    assert state.xenon > Xe_eq, (
        f"Xe at 3h ({state.xenon:.3e}) should exceed full-power eq ({Xe_eq:.3e})"
    )


def test_xenon_pit_iodine_decayed() -> None:
    """I-135 decays after shutdown — should be less than full-power equilibrium."""
    I_eq, _ = xenon_equilibrium(1.0)
    state = trigger_xenon_pit(default_state())
    assert state.iodine < I_eq, (
        f"Iodine at 3h ({state.iodine:.3e}) should be below full-power eq ({I_eq:.3e})"
    )


def test_xenon_pit_large_negative_reactivity() -> None:
    """Xe + post-shutdown boron prevent restart even at 100% rod withdrawal."""
    from physics.reactivity import rod_reactivity, boron_reactivity
    state = trigger_xenon_pit(default_state())
    rho_xe = xenon_reactivity(state.xenon)
    rho_rods_max = rod_reactivity([100.0, 100.0, 100.0, 100.0])
    rho_b = boron_reactivity(state.boron_ppm)
    total = rho_xe + rho_rods_max + rho_b
    assert total < 0, (
        f"Xe ({rho_xe:.5f}) + rods@100% ({rho_rods_max:.5f}) + boron ({rho_b:.5f}) = {total:.5f}; should be negative"
    )


def test_xenon_pit_alarm_added() -> None:
    state = trigger_xenon_pit(default_state())
    assert "XENON_PIT" in state.alarms


def test_xenon_pit_boron_injected() -> None:
    """Post-shutdown boration is applied to maintain shutdown margin during cooldown."""
    state = trigger_xenon_pit(default_state())
    assert state.boron_ppm > 0.0


def test_xenon_pit_subcritical_at_cold_temps() -> None:
    """Reactor stays subcritical at cold-shutdown temperatures despite xenon clearing.

    At T_cool=400K the cold moderator feedback turns positive; boron must provide
    enough static margin so the xenon pit cannot be overwhelmed by temperature alone.
    """
    from physics.reactivity import compute_reactivity

    state = trigger_xenon_pit(default_state())
    # Simulate operator attempting restart at cold conditions, rods 75% out
    state.rod_positions = [75.0, 75.0, 75.0, 75.0]
    state.t_cool = 400.0   # K — significant cooldown after 3 h
    state.t_fuel = 600.0   # K
    rho = compute_reactivity(state)
    assert rho < 0.0, (
        f"Reactor should remain subcritical in cold xenon pit (rho={rho:.4f})"
    )


# ---------------------------------------------------------------------------
# LOCA — Phase 2 two-phase tests
# ---------------------------------------------------------------------------

def test_loca_large_break_eccs_armed_no_fuel_damage() -> None:
    """Large break (1.0), ECCS armed: fuel_damage=False after 120 s (1200 ticks).

    HPSI activates within ~3 ticks (pressure drops below 100 bar quickly).
    loca_flow_fraction recovers faster than it decays once HPSI is on.
    With SCRAM reducing fission power and ECCS providing cooling,
    DNBR stays above 1.0 and fuel_damage never sets.
    """
    state = trigger_loca(default_state(), break_size=1.0)
    state = asyncio.run(_run(state, 1200))
    assert not state.fuel_damage, (
        f"fuel_damage set unexpectedly after 120 s with ECCS armed "
        f"(dnbr_low_timer={state.dnbr_low_timer:.2f})"
    )


def test_loca_large_break_no_eccs_fuel_damage() -> None:
    """Large break, eccs_armed=False, SCRAM bypassed (ATWS): fuel_damage=True within 45 s.

    With full fission power maintained and no ECCS cooling, loca_flow_fraction
    decays to zero (~12.5 s), CHF collapses, DNBR < 1.0 for 5 s → fuel_damage.
    SCRAM is bypassed to hold power; this represents the worst-case ATWS+LOCA scenario.
    """
    state = trigger_loca(default_state(), break_size=1.0)
    state.eccs_armed = False
    state.scram_bypasses = ["LO_PRESSURE", "LO_FLOW", "HI_FUEL_TEMP",
                            "HI_COOL_TEMP", "HI_PRESSURE", "HI_POWER"]

    for tick in range(450):
        state.n = 1.0  # maintain power (ATWS: SCRAM bypassed, rods stay out)
        state = asyncio.run(_run(state, 1))
        if state.fuel_damage:
            break

    assert state.fuel_damage, (
        f"fuel_damage not set within 45 s with no ECCS and full power "
        f"(dnbr_low_timer={state.dnbr_low_timer:.2f}, dnbr={state.dnbr:.3f})"
    )


def test_loca_small_break_slow_pressure_decay() -> None:
    """Small break (0.05): pressure decays slowly, still above 100 bar after 60 s.

    The pressurizer counteracts the slow blowdown; combined with the small
    break_size, pressure equilibrates well above the HPSI threshold at 60 s.
    No fuel damage expected.
    """
    state = trigger_loca(default_state(), break_size=0.05)
    state = asyncio.run(_run(state, 600))   # 60 s

    assert state.pressure > ECCS_HPSI_THRESHOLD, (
        f"Pressure reached HPSI threshold too quickly for small break: "
        f"{state.pressure/1e5:.1f} bar at t=60 s"
    )
    assert not state.fuel_damage, "fuel_damage set during small-break LOCA with ECCS"


def test_loca_void_spike_within_5s() -> None:
    """Large break: void_fraction.mean() > 0.2 within first 5 s (50 ticks).

    Rapid depressurization drops T_sat below coolant temperature; all nodes
    flash to steam within a few ticks, giving a core-average void well above 20%.
    """
    state = trigger_loca(default_state(), break_size=1.0)
    max_void_mean = 0.0
    for _ in range(50):
        state = asyncio.run(_run(state, 1))
        max_void_mean = max(max_void_mean, float(state.void_fraction.mean()))

    assert max_void_mean > 0.2, (
        f"Void fraction mean did not exceed 0.2 within 5 s of large break "
        f"(max mean void = {max_void_mean:.3f})"
    )


def test_loca_scram_fires_within_5s() -> None:
    """Large break: SCRAM fires within 5 s (pressure drops below 120 bar immediately).

    trigger_loca sets pressure to 105 bar (<120 bar SCRAM_LOW), so the
    end-of-tick SCRAM check fires on the very first simulation tick.
    """
    state = trigger_loca(default_state(), break_size=1.0)
    scram_fired = False
    for _ in range(50):    # 50 ticks = 5 s
        state = asyncio.run(_run(state, 1))
        if state.scram:
            scram_fired = True
            assert state.t <= 5.0 + 0.15, (
                f"SCRAM fired too late at t={state.t:.2f} s"
            )
            break

    assert scram_fired, "SCRAM did not fire within 5 s of large-break LOCA"
