"""Loss of Coolant Accident (LOCA) — Phase 2 two-phase blowdown + ECCS + reflood."""

from __future__ import annotations

from physics.constants import (
    ECCS_HPSI_THRESHOLD,
    ECCS_LPSI_THRESHOLD,
    ECCS_WATER_TEMP,
)
from api.state import PlantState


def trigger_loca(state: PlantState, break_size: float = 1.0) -> PlantState:
    """Initiate LOCA: primary break causes instantaneous partial depressurization.

    Args:
        state: current PlantState
        break_size: fraction of main coolant pipe cross-section (0.0–1.0).
                    1.0 = large double-ended guillotine break (DEGB).
                    0.05 = small-break LOCA (SBLOCA).

    Instantaneous effects:
        - loca_active = True, loca_break_size = break_size
        - pressure -= break_size * 50e5 Pa
          (large break: 155 → 105 bar; SCRAM on low-pressure fires next tick)
        - ECCS armed (ensures automatic actuation is enabled)
    """
    state.loca_active = True
    state.loca_break_size = break_size
    state.eccs_armed = True
    state.pressure -= break_size * 50e5
    if "LOCA" not in state.alarms:
        state.alarms = list(state.alarms) + ["LOCA"]
    return state


def update_loca(state: PlantState, dt: float) -> PlantState:
    """Advance LOCA dynamics by dt seconds.  Called every tick when loca_active.

    Blowdown (pressure > 20 bar):
        dP/dt = -break_size * pressure * 0.25 / s
        d(loca_flow_fraction)/dt = -break_size * 0.08 / s

    ECCS (when eccs_armed):
        HPSI (pressure <= 100 bar): loca_flow_fraction += 0.1 * dt * 2.0 per tick;
            T_in clamped to 293 K; boron += 500 ppm/min
        LPSI (pressure <= 20 bar): loca_flow_fraction += 0.3 * dt * 2.0 additionally

    Reflood (LPSI active): lpsi_timer advances; every 5 s one more bottom node
        has void_fraction cleared to 0 and quality set to -0.5.

    The computed loca_flow_fraction caps state.flow_fraction so downstream
    thermal/DNBR calculations see the correct reduced coolant flow.
    """
    s = state
    bp = s.loca_break_size
    N = len(s.void_fraction)

    # ── Blowdown ──────────────────────────────────────────────────────────────
    if s.pressure > ECCS_LPSI_THRESHOLD:
        s.pressure = max(
            s.pressure - bp * s.pressure * 0.25 * dt,
            ECCS_LPSI_THRESHOLD,
        )

    s.loca_flow_fraction = max(0.0, s.loca_flow_fraction - bp * 0.08 * dt)

    # ── ECCS injection ────────────────────────────────────────────────────────
    hpsi_active = s.eccs_armed and s.pressure <= ECCS_HPSI_THRESHOLD
    lpsi_active = s.eccs_armed and s.pressure <= ECCS_LPSI_THRESHOLD

    if hpsi_active:
        s.loca_flow_fraction = min(1.0, s.loca_flow_fraction + 0.1 * dt * 2.0)

    if lpsi_active:
        s.loca_flow_fraction = min(1.0, s.loca_flow_fraction + 0.3 * dt * 2.0)

    if hpsi_active or lpsi_active:
        s.t_in = ECCS_WATER_TEMP
        s.boron_ppm += 500.0 * dt / 60.0

    # ── Apply LOCA flow cap ───────────────────────────────────────────────────
    s.flow_fraction = min(s.flow_fraction, s.loca_flow_fraction)

    # ── Reflood: clear void bottom→top once per 5 s of LPSI ──────────────────
    if lpsi_active:
        s.lpsi_timer += dt
        cleared_nodes = int(s.lpsi_timer / 5.0)
        for n in range(min(cleared_nodes, N)):
            s.void_fraction[n] = 0.0
            s.quality[n] = -0.5

    return s
