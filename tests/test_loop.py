"""Simulation loop integration test — 60-second hot-standby stability."""

import asyncio

import pytest

from api.state import default_state
from api.simulation_loop import simulation_tick
from physics.constants import T_REF_FUEL, T_REF_COOLANT, PRESSURE_NOMINAL


async def _run_ticks(n: int, dt: float = 0.1) -> object:
    state = default_state()
    for _ in range(n):
        state = await simulation_tick(state, dt=dt)
    return state


def test_steady_state_60s() -> None:
    """600 ticks (60 s) from hot standby: power, temperatures, and pressure stay stable."""
    state = asyncio.run(_run_ticks(600))

    assert not state.scram, "Unexpected SCRAM"
    assert state.alarms == [], f"Unexpected alarms: {state.alarms}"

    # Power within 1 % of nominal
    assert abs(state.n - 1.0) < 0.01, f"Power drifted: n={state.n:.6f}"

    # Temperatures within 1 K of reference (fixed point of thermal model)
    assert abs(state.t_fuel - T_REF_FUEL) < 1.0, (
        f"Fuel temp drifted: {state.t_fuel:.3f} K (ref {T_REF_FUEL} K)"
    )
    assert abs(state.t_cool - T_REF_COOLANT) < 1.0, (
        f"Coolant temp drifted: {state.t_cool:.3f} K (ref {T_REF_COOLANT} K)"
    )

    # Pressure within 1 bar (1e5 Pa) of nominal
    assert abs(state.pressure - PRESSURE_NOMINAL) < 1.0e5, (
        f"Pressure drifted: {state.pressure:.0f} Pa (nominal {PRESSURE_NOMINAL} Pa)"
    )

    # All pumps at full speed
    assert state.flow_fraction > 0.999, f"Flow fraction dropped: {state.flow_fraction:.4f}"

    # Simulation time advanced correctly
    assert state.t == pytest.approx(60.0, abs=1e-6)
