"""Replay/control tests for diesel RNG seeding and debug telemetry."""

import asyncio

from api.server import ControlInput, _state_to_ws, control
from api.simulation_loop import get_sim_rng_debug, set_sim_rng_seed, simulation_tick
from api.state import PlantState, default_state
from physics.constants import DIESEL_START_DELAY_MAX, DIESEL_START_DELAY_MIN, T_COOLANT_INLET


def _run_ticks(state: PlantState, n: int, dt: float = 0.1) -> PlantState:
    """Advance state for n ticks using the production simulation loop."""
    async def _run() -> PlantState:
        s = state
        for _ in range(n):
            s = await simulation_tick(s, dt=dt)
        return s

    return asyncio.run(_run())


def _collect_diesel_start_signature(seed: int) -> tuple[list[float], list[int]]:
    """Return sampled start delays and run-transition ticks for both diesels."""
    set_sim_rng_seed(seed)
    state = default_state()
    state.offsite_power = False
    state.diesel_start_signals = [True, True]

    first_running_tick = [-1, -1]
    for tick in range(1, 260):
        state = asyncio.run(simulation_tick(state, dt=0.1))
        for i, ds in enumerate(state.diesel_states):
            if ds.state == "running" and first_running_tick[i] < 0:
                first_running_tick[i] = tick

    delays = [float(d.start_delay) for d in state.diesel_states]
    return delays, first_running_tick


def test_control_reseeds_and_omitted_field_preserves_current_seed() -> None:
    """Explicit seed reseeds RNG; omitted field leaves seed untouched."""
    set_sim_rng_seed(None)

    asyncio.run(control(ControlInput(diesel_rng_seed=1234)))
    debug = get_sim_rng_debug()
    assert debug["diesel_rng_seed"] == 1234
    assert debug["diesel_rng_seeded"] is True

    asyncio.run(control(ControlInput()))
    debug_after = get_sim_rng_debug()
    assert debug_after["diesel_rng_seed"] == 1234
    assert debug_after["diesel_rng_seeded"] is True


def test_control_null_seed_restores_unseeded_mode() -> None:
    """Explicit null from /control switches back to unseeded RNG."""
    asyncio.run(control(ControlInput(diesel_rng_seed=77)))
    assert get_sim_rng_debug()["diesel_rng_seeded"] is True

    asyncio.run(control(ControlInput(diesel_rng_seed=None)))
    debug = get_sim_rng_debug()
    assert debug["diesel_rng_seed"] is None
    assert debug["diesel_rng_seeded"] is False


def test_deterministic_replay_same_seed_same_delays_and_transitions() -> None:
    """Given identical seed and timeline, diesel timing must replay exactly."""
    delays_a, run_ticks_a = _collect_diesel_start_signature(seed=2026)
    delays_b, run_ticks_b = _collect_diesel_start_signature(seed=2026)

    assert delays_a == delays_b
    assert run_ticks_a == run_ticks_b
    assert all(DIESEL_START_DELAY_MIN <= d <= DIESEL_START_DELAY_MAX for d in delays_a)


def test_nonseeded_behavior_still_valid_within_spec_bounds() -> None:
    """Without explicit seed, diesel startup still respects spec delay bounds."""
    set_sim_rng_seed(None)
    state = default_state()
    state.offsite_power = False
    state.diesel_start_signals = [True, True]

    state = _run_ticks(state, 2)
    sampled_delays = [float(d.start_delay) for d in state.diesel_states]
    assert all(DIESEL_START_DELAY_MIN <= d <= DIESEL_START_DELAY_MAX for d in sampled_delays)


def test_ws_payload_includes_replay_and_pump_power_debug_fields() -> None:
    """Websocket payload exposes seed/state debug and diesel/pump diagnostics."""
    set_sim_rng_seed(314159)
    state = default_state()
    state.offsite_power = False
    state.diesel_start_signals = [True, False]
    state.pumps = [True, True, True, True]
    state = _run_ticks(state, 1)

    payload = _state_to_ws(state)

    assert "diesel_rng_seed" in payload
    assert "diesel_rng_seeded" in payload
    assert "diesel_rng_bit_generator" in payload
    assert "diesel_rng_state" in payload
    assert "diesel_start_signals" in payload
    assert "diesel_start_timer_s" in payload
    assert "diesel_start_delay_s" in payload
    assert "pump_power_source" in payload
    assert "pump_speeds" in payload

    assert payload["diesel_rng_seed"] == 314159
    assert payload["diesel_rng_seeded"] is True
    assert len(payload["diesel_start_delay_s"]) == 2
    assert len(payload["pump_power_source"]) == 4


def test_t_in_recovers_toward_nominal_with_diesel_running() -> None:
    """With LOOP + diesel recovery, T_in trends cooler than full SBO behavior."""
    set_sim_rng_seed(2026)
    diesel_state = default_state()
    diesel_state.offsite_power = False
    diesel_state.diesel_start_signals = [True, True]

    sbo_state = default_state()
    sbo_state.offsite_power = False
    sbo_state.diesel_start_signals = [False, False]
    for ds in sbo_state.diesel_states:
        ds.state = "failed"

    diesel_state = _run_ticks(diesel_state, 600)  # 60 s
    sbo_state = _run_ticks(sbo_state, 600)        # 60 s

    assert any(d.state == "running" for d in diesel_state.diesel_states)
    assert abs(diesel_state.t_in - T_COOLANT_INLET) < abs(sbo_state.t_in - T_COOLANT_INLET)
