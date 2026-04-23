"""Tests for plant/pumps.py and plant/diesels.py."""

import asyncio

import numpy as np
import pytest

from api.simulation_loop import simulation_tick
from api.state import default_state
from plant.pumps import step_pumps, total_flow_fraction
from plant.diesels import DieselState, step_diesels
from physics.constants import (
    DIESEL_PUMP_SPEED_MAX,
    DIESEL_START_DELAY_MAX,
    DIESEL_START_DELAY_MIN,
    PUMP_COASTDOWN_TAU,
)


# ---------------------------------------------------------------------------
# Pump tests
# ---------------------------------------------------------------------------

class TestPumpCostdown:
    def _run_coastdown(self, seconds: float, dt: float = 1.0) -> np.ndarray:
        """Simulate all pumps coasting (offsite_power=False) for `seconds`."""
        speeds = np.ones(4)
        powered = [True, True, True, True]
        steps = int(seconds / dt)
        for _ in range(steps):
            speeds = step_pumps(speeds, powered, offsite_power=False, dt=dt)
        return speeds

    def test_speeds_decay_to_near_zero_within_120s(self) -> None:
        """After 120 s coastdown (4× tau), speeds should be < 2% of initial."""
        speeds = self._run_coastdown(120.0)
        assert np.all(speeds < 0.02), f"Expected near-zero speeds, got {speeds}"

    def test_speeds_are_half_at_one_tau(self) -> None:
        """After one tau (30 s), speeds should be ~1/e ≈ 0.368."""
        speeds = self._run_coastdown(PUMP_COASTDOWN_TAU, dt=0.1)
        expected = np.exp(-1.0)
        np.testing.assert_allclose(speeds, expected, rtol=1e-3)

    def test_pumps_stay_at_full_speed_with_power(self) -> None:
        speeds = np.ones(4)
        new = step_pumps(speeds, [True] * 4, offsite_power=True, dt=5.0)
        np.testing.assert_array_equal(new, np.ones(4))

    def test_individual_pump_coastdown(self) -> None:
        """Only unpowered pumps coast; powered ones stay at 1."""
        speeds = np.ones(4)
        # Pump 0 loses power (manual off), others have offsite power + enabled
        powered = [False, True, True, True]
        new = step_pumps(speeds, powered, offsite_power=True, dt=30.0)
        assert new[0] == pytest.approx(np.exp(-1.0), rel=1e-3)
        assert np.all(new[1:] == 1.0)

    def test_partial_coastdown_monotone_decreasing(self) -> None:
        speeds = np.ones(4)
        powered = [True] * 4
        prev = speeds.copy()
        for _ in range(10):
            speeds = step_pumps(speeds, powered, offsite_power=False, dt=5.0)
            assert np.all(speeds < prev)
            prev = speeds.copy()

    def test_diesel_powered_pumps_ramp_and_cap(self) -> None:
        speeds = np.zeros(4)
        powered = [True] * 4
        for _ in range(200):
            speeds = step_pumps(
                speeds,
                powered,
                offsite_power=False,
                dt=0.1,
                diesel_powered=[True, True, True, True],
            )
        assert speeds.max() <= DIESEL_PUMP_SPEED_MAX + 1e-9
        assert speeds.min() >= DIESEL_PUMP_SPEED_MAX - 1e-3

    def test_diesel_bus_assignment_partial_power(self) -> None:
        speeds = np.zeros(4)
        powered = [True] * 4
        for _ in range(200):
            speeds = step_pumps(
                speeds,
                powered,
                offsite_power=False,
                dt=0.1,
                diesel_powered=[True, True, False, False],
            )
        assert speeds[0] >= 0.49
        assert speeds[1] >= 0.49
        assert speeds[2] <= 0.1
        assert speeds[3] <= 0.1


class TestFlowFraction:
    def test_all_pumps_full_speed(self) -> None:
        assert total_flow_fraction(np.ones(4)) == pytest.approx(1.0)

    def test_all_pumps_stopped(self) -> None:
        assert total_flow_fraction(np.zeros(4)) == pytest.approx(0.0)

    def test_half_pumps_running(self) -> None:
        speeds = np.array([1.0, 1.0, 0.0, 0.0])
        assert total_flow_fraction(speeds) == pytest.approx(0.5)

    def test_flow_fraction_zero_after_full_coastdown(self) -> None:
        """After very long coastdown, flow fraction reaches 0."""
        speeds = np.ones(4)
        powered = [True] * 4
        # Simulate 10 minutes (>> 4 × tau) with offsite power lost
        for _ in range(600):
            speeds = step_pumps(speeds, powered, offsite_power=False, dt=1.0)
        ff = total_flow_fraction(speeds)
        assert ff == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Diesel generator tests
# ---------------------------------------------------------------------------

class TestDieselStateMachine:
    def _make_diesels(self, n: int = 2) -> list[DieselState]:
        return [DieselState() for _ in range(n)]

    def test_initial_state_is_standby(self) -> None:
        states = self._make_diesels()
        assert all(s.state == "standby" for s in states)

    def test_start_signal_transitions_to_starting(self) -> None:
        states = self._make_diesels(2)
        new = step_diesels(states, [True, False], dt=1.0)
        assert new[0].state == "starting"
        assert new[1].state == "standby"

    def test_standby_to_starting_to_running(self) -> None:
        """Full sequence: standby → starting → running after start_delay seconds."""
        states = [DieselState(start_delay=12.0)]
        # Step 1: send start signal
        states = step_diesels(states, [True], dt=1.0)
        assert states[0].state == "starting"
        # Step until running
        elapsed = 0.0
        dt = 0.5
        max_iter = 200
        for _ in range(max_iter):
            states = step_diesels(states, [False], dt=dt)
            elapsed += dt
            if states[0].state == "running":
                break
        assert states[0].state == "running", "Diesel never reached running state"
        assert elapsed >= 12.0 - dt  # reached running at or after delay

    def test_running_transition_within_spec_delay(self) -> None:
        """Diesel must reach RUNNING within DIESEL_START_DELAY_MAX seconds of signal."""
        states = [DieselState(start_delay=DIESEL_START_DELAY_MAX)]
        dt = 0.25
        # Send start signal once
        states = step_diesels(states, [True], dt=dt)
        elapsed = dt
        for _ in range(int(DIESEL_START_DELAY_MAX / dt) + 10):
            states = step_diesels(states, [False], dt=dt)
            elapsed += dt
            if states[0].state == "running":
                break
        assert states[0].state == "running"
        assert elapsed <= DIESEL_START_DELAY_MAX + dt

    def test_no_transition_without_signal(self) -> None:
        states = self._make_diesels(2)
        for _ in range(100):
            states = step_diesels(states, [False, False], dt=1.0)
        assert all(s.state == "standby" for s in states)

    def test_failed_diesel_stays_failed(self) -> None:
        states = [DieselState(state="failed")]
        new = step_diesels(states, [True], dt=5.0)
        assert new[0].state == "failed"

    def test_running_diesel_stays_running(self) -> None:
        states = [DieselState(state="running")]
        new = step_diesels(states, [False], dt=5.0)
        assert new[0].state == "running"

    def test_start_timer_accumulates_in_starting_state(self) -> None:
        states = [DieselState(start_delay=100.0)]  # long delay so it won't flip
        states = step_diesels(states, [True], dt=1.0)
        assert states[0].state == "starting"
        states = step_diesels(states, [False], dt=5.0)
        # Transition step (dt=1.0) resets timer to 0; only the second step (dt=5.0) adds time
        assert states[0].start_timer == pytest.approx(5.0)

    def test_randomized_delay_in_range(self) -> None:
        rng = np.random.default_rng(42)
        delays: list[float] = []
        for _ in range(100):
            states = [DieselState()]
            states = step_diesels(states, [True], dt=0.1, rng=rng)
            delays.append(states[0].start_delay)
        assert all(DIESEL_START_DELAY_MIN <= d <= DIESEL_START_DELAY_MAX for d in delays)

    def test_seeded_rng_is_deterministic(self) -> None:
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        a = step_diesels([DieselState()], [True], dt=0.1, rng=rng_a)[0].start_delay
        b = step_diesels([DieselState()], [True], dt=0.1, rng=rng_b)[0].start_delay
        assert a == pytest.approx(b)

    def test_default_midpoint_when_rng_none(self) -> None:
        state = step_diesels([DieselState()], [True], dt=0.1, rng=None)[0]
        assert state.start_delay == pytest.approx((DIESEL_START_DELAY_MIN + DIESEL_START_DELAY_MAX) / 2.0)

    def test_invalid_state_raises(self) -> None:
        with pytest.raises(ValueError):
            DieselState(state="unknown")


class TestLoopRecovery:
    async def _run(self, state: object, n: int, dt: float = 0.1) -> object:
        for _ in range(n):
            state = await simulation_tick(state, dt=dt)
        return state

    def test_loop_diesel_restores_forced_flow(self) -> None:
        state = default_state()
        state.offsite_power = False
        state.diesel_start_signals = [True, True]
        state = asyncio.run(self._run(state, 300))  # 30 s
        assert any(d.state == "running" for d in state.diesel_states)
        assert state.flow_fraction > 0.2

    def test_loop_pump_speed_capped_at_diesel_max(self) -> None:
        state = default_state()
        state.offsite_power = False
        state.diesel_start_signals = [True, True]
        state = asyncio.run(self._run(state, 400))  # 40 s
        assert state.pump_speeds.max() <= DIESEL_PUMP_SPEED_MAX + 0.01

    def test_sbo_no_pump_recovery(self) -> None:
        state = default_state()
        state.offsite_power = False
        state.diesel_start_signals = [False, False]
        for ds in state.diesel_states:
            ds.state = "failed"
        state = asyncio.run(self._run(state, 1200))  # 120 s coastdown
        assert state.flow_fraction <= 0.05
