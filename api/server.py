"""FastAPI server: REST endpoints + WebSocket real-time state streaming."""

from __future__ import annotations

import asyncio
import json

import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api.state import PlantState, default_state
from api.simulation_loop import simulation_tick
from physics.thermal import t_sat_celsius

# ---------------------------------------------------------------------------
# Global mutable state — protected by _lock in every coroutine that touches it
# ---------------------------------------------------------------------------
_state: PlantState = default_state()
_lock: asyncio.Lock = asyncio.Lock()
_connections: set[WebSocket] = set()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_to_ws(s: PlantState) -> dict:
    """Serialize PlantState to the SPEC WebSocket JSON format.

    Unit conversions happen here (K→°C, Pa→bar) per CLAUDE.md.
    """
    rod_pos = [0.0, 0.0, 0.0, 0.0] if s.scram else s.rod_positions
    rod_target_pos = s.rod_target_positions
    rho_total = s.rho_rod + s.rho_doppler + s.rho_moderator + s.rho_xenon + s.rho_boron + s.rho_eccs + s.ejection_rho
    t_cool_c = s.t_cool - 273.15
    t_hot_leg_c = 2.0 * s.t_cool - s.t_in - 273.15
    subcooling = t_sat_celsius(s.pressure) - t_hot_leg_c
    eccs = s.eccs_state
    return {
        "t": round(s.t, 2),
        "power_pct": round(s.n * 100.0, 3),
        "t_fuel": round(s.t_fuel - 273.15, 2),
        "t_cool": round(t_cool_c, 2),
        "t_hot_leg": round(t_hot_leg_c, 2),
        "pressure": round(s.pressure / 1.0e5, 2),
        "flow_pct": round(s.flow_fraction * 100.0, 2),
        "xenon_worth": round(s.rho_xenon, 8),
        "rho_total": round(rho_total, 8),
        "rho_rod": round(s.rho_rod, 8),
        "rho_doppler": round(s.rho_doppler, 8),
        "rho_moderator": round(s.rho_moderator, 8),
        "rho_boron": round(s.rho_boron, 8),
        "rho_eccs": round(s.rho_eccs, 8),
        "rod_position": [round(r, 1) for r in rod_pos],
        "rod_target_position": [round(r, 1) for r in rod_target_pos],
        "pumps": list(s.pumps),
        "diesels": [d.state for d in s.diesel_states],
        "alarms": list(s.alarms),
        "scram": s.scram,
        "scram_cause": s.scram_cause,
        "eccs_armed": s.eccs_armed,
        "boron_ppm": round(s.boron_ppm, 1),
        "boron_target_ppm": round(s.boron_target_ppm, 1),
        "subcooling": round(subcooling, 1),
        "eccs_hpsi": eccs.hpsi_active if eccs else False,
        "eccs_lpsi": eccs.lpsi_active if eccs else False,
        "t_in": round(s.t_in - 273.15, 1),
        "pressurizer_level": round(s.pressurizer_level * 100.0, 1),
        "przr_heater_pct": round(s.przr_heater_fraction * 100.0, 1),
        "przr_spray_pct": round(s.przr_spray_fraction * 100.0, 1),
        "porv_open": s.porv_open,
        "porv_stuck_open": s.porv_stuck_open,
        "scram_bypasses": list(s.scram_bypasses),
    }


async def _broadcast(payload: dict) -> None:
    dead: set[WebSocket] = set()
    data = json.dumps(payload)
    for ws in _connections:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


async def _sim_loop() -> None:
    global _state
    while True:
        await asyncio.sleep(0.1)
        async with _lock:
            _state = await simulation_tick(_state, dt=0.1)
            msg = _state_to_ws(_state)
        await _broadcast(msg)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_sim_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Nuclear Reactor Simulator", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_panel() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "panel.html")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    # Send current state immediately so the client doesn't wait up to 100 ms.
    async with _lock:
        msg = _state_to_ws(_state)
    await websocket.send_text(json.dumps(msg))
    try:
        while True:
            # Drain any incoming frames (ping/pong, client commands).
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections.discard(websocket)


class ControlInput(BaseModel):
    rod_positions: Optional[list[float]] = None   # 4 values, 0–100 %
    pumps: Optional[list[bool]] = None             # 4 booleans
    boron_ppm: Optional[float] = None
    scram: Optional[bool] = None
    eccs_armed: Optional[bool] = None
    diesel_start_signals: Optional[list[bool]] = None  # 2 booleans
    przr_heater_fraction: Optional[float] = None  # 0–1 operator heater demand
    przr_spray_fraction: Optional[float] = None   # 0–1 operator spray demand
    porv_open: Optional[bool] = None              # manually open/close PORV
    porv_stuck_open: Optional[bool] = None        # latch PORV in stuck-open failure
    scram_bypasses: Optional[list[str]] = None    # list of bypassed SCRAM channels


@app.post("/control")
async def control(cmd: ControlInput) -> dict:
    global _state
    async with _lock:
        if cmd.rod_positions is not None:
            _state.rod_target_positions = list(cmd.rod_positions)
        if cmd.pumps is not None:
            _state.pumps = list(cmd.pumps)
        if cmd.boron_ppm is not None:
            _state.boron_target_ppm = float(cmd.boron_ppm)
        if cmd.scram is not None:
            if cmd.scram and not _state.scram:
                _state.scram = True
                _state.t_since_scram = 0.0
            elif not cmd.scram:
                _state.scram = False
        if cmd.eccs_armed is not None:
            _state.eccs_armed = cmd.eccs_armed
        if cmd.diesel_start_signals is not None:
            _state.diesel_start_signals = list(cmd.diesel_start_signals)
        if cmd.przr_heater_fraction is not None:
            _state.przr_heater_fraction = float(max(0.0, min(1.0, cmd.przr_heater_fraction)))
        if cmd.przr_spray_fraction is not None:
            _state.przr_spray_fraction = float(max(0.0, min(1.0, cmd.przr_spray_fraction)))
        if cmd.porv_open is not None:
            _state.porv_open = bool(cmd.porv_open)
        if cmd.porv_stuck_open is not None:
            _state.porv_stuck_open = bool(cmd.porv_stuck_open)
            if _state.porv_stuck_open:
                _state.porv_open = True  # stuck-open means it opens and stays open
        if cmd.scram_bypasses is not None:
            _state.scram_bypasses = list(cmd.scram_bypasses)
    return {"status": "ok"}


class ScenarioInput(BaseModel):
    name: str


@app.post("/scenario")
async def trigger_scenario(scenario: ScenarioInput) -> dict:
    global _state
    name = scenario.name.strip().lower()
    async with _lock:
        if name == "loop":
            # Loss of offsite power — pumps coast down, diesel start signals sent
            _state.offsite_power = False
            _state.diesel_start_signals = [True, True]
        elif name == "sbo":
            # Station blackout — LOOP + both diesels fail immediately
            _state.offsite_power = False
            _state.diesel_start_signals = [False, False]
            for ds in _state.diesel_states:
                ds.state = "failed"
        elif name == "loca":
            # Large-break LOCA: rapid RCS depressurization
            _state.pressure = 5.0e6  # ~50 bar — triggers SCRAM and ECCS
        elif name == "rod_ejection":
            # Eject bank A: mechanically failed rod moves from current position to 100%
            # withdrawn instantly. ejection_rho is a permanent positive reactivity insertion
            # (scales with how far the bank was inserted) that persists until SCRAM fires,
            # at which point the remaining banks insert and it is zeroed out.
            _ejected_bank = 0
            current_pos = _state.rod_positions[_ejected_bank]
            pos_change_fraction = (100.0 - current_pos) / 100.0  # 0 if already at 100%
            # Ejection is mechanical failure — bypasses CRDM rate limit; set both actual and target.
            rod_positions = list(_state.rod_positions)
            rod_positions[_ejected_bank] = 100.0
            _state.rod_positions = rod_positions
            rod_targets = list(_state.rod_target_positions)
            rod_targets[_ejected_bank] = 100.0
            _state.rod_target_positions = rod_targets
            _state.ejection_rho = 0.01 * pos_change_fraction  # up to +0.01 dk/k (SPEC value)
            if "ROD_EJECTION" not in _state.alarms:
                _state.alarms = list(_state.alarms) + ["ROD_EJECTION"]
        elif name == "stuck_open_porv":
            # Stuck-open PORV (TMI-2 initiating event):
            # PORV opens at high pressure but fails to reclose — slow primary bleed.
            # Pressure drops ~2 bar/min; SCRAM on low pressure in ~2.5 min unless closed.
            # To "close" the PORV in the sim the operator uses the block valve (porv_stuck_open=False).
            _state.porv_open = True
            _state.porv_stuck_open = True
        elif name == "xenon_pit":
            # Fast-forward xenon kinetics to the post-shutdown peak (~3 hr after full-power trip).
            # Iodine-135 (t½ 6.7 hr) decays into Xe-135 (t½ 9.2 hr) which builds to peak,
            # then slowly clears over ~30–40 hr.  At the 3-hr peak, xenon worth ≈ −0.033 dk/k.
            # Emergency boration raises to 1500 ppm (CVCS procedure).  Even at 100% rod
            # withdrawal: rho_rods(+0.143) + rho_Xe(−0.033) + rho_B(−0.150) = −0.040 dk/k
            # — subcritical.  Restart requires xenon clearance (~30 h) then boron dilution.
            from physics.xenon import xenon_equilibrium, rk4_step_xenon
            I0, Xe0 = xenon_equilibrium(1.0)          # equilibrium at 100% power
            I, Xe = I0, Xe0
            _DT_XE = 60.0                              # 1-minute integration steps
            for _ in range(int(3 * 3600 / _DT_XE)):   # 3 hours of post-shutdown decay
                I, Xe = rk4_step_xenon(I, Xe, 0.0, _DT_XE)
            # After 3 h, every precursor group has decayed to ≈ 0 (slowest t½ = 56 s,
            # so 3 h = ~190 half-lives).  Reset n and precursors accordingly.
            _state.iodine = I
            _state.xenon = Xe
            _state.boron_ppm = 1500.0
            _state.boron_target_ppm = 1500.0
            _state.scram = True
            _state.t_since_scram = 3 * 3600
            _state.n = 1e-6
            _state.precursors = np.zeros(6)
        else:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": f"Unknown scenario: {scenario.name!r}"},
            )
    return {"status": "ok", "scenario": scenario.name}


@app.post("/reset")
async def reset() -> dict:
    global _state
    async with _lock:
        _state = default_state()
    return {"status": "ok"}
