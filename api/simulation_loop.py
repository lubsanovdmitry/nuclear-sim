"""100 ms simulation tick — orchestrates all physics and plant subsystems."""

from __future__ import annotations

import copy

import numpy as np

from api.state import PlantState
from api.alarms import PHASE2_ALARMS
from physics.pke_solver import rk4_step as pke_rk4
from physics.thermal import step_thermal, step_thermal_nodal
from physics.xenon import rk4_step_xenon, xenon_reactivity
from physics.decay_heat import decay_heat_power
from physics.reactivity import (
    rod_reactivity,
    doppler_reactivity,
    moderator_reactivity,
    boron_reactivity,
    void_reactivity,
)
from physics.axial import cosine_power_shape
from physics.two_phase import (
    actual_heat_flux_array,
    boiling_regime as _boiling_regime,
    critical_heat_flux,
    heat_transfer_coefficient,
    saturation_temp,
    thermodynamic_quality,
    void_fraction as _void_fraction,
    void_fraction_subcooled,
)
from physics.constants import (
    NOMINAL_POWER_W, ECCS_WATER_TEMP, T_COOLANT_INLET,
    T_CLADDING_FAILURE, ECCS_HPSI_THRESHOLD,
    H_TRANSFER_A, M_FUEL_CP_FUEL, ALPHA_DOPPLER, T_REF_FUEL,
    TAU_STEAM_GENERATOR, BORON_RATE_PPM_PER_S, ROD_SPEED_PCT_PER_S,
    PORV_OPEN_SETPOINT, PORV_CLOSE_SETPOINT, PORV_LEVEL_DRAIN,
    PRESSURE_SCRAM_HIGH, PRESSURE_SCRAM_LOW,
    HEAT_FLUX_SCALE,
)
from physics.thermal import t_sat_celsius
from plant.pumps import step_pumps, total_flow_fraction
from plant.diesels import step_diesels
from plant.eccs import step_eccs, eccs_reactivity, ECCSState
from plant.pressurizer import step_pressurizer, step_pressurizer_level
from accidents.loca import update_loca

# Alarm setpoints (SI)
_ALARM_HIGH_POWER: float = 1.10
_SCRAM_HIGH_POWER: float = 1.20
_ALARM_FUEL_TEMP_K: float = 1073.15
_ALARM_COOL_TEMP_K: float = 590.15
_ALARM_LOW_FLOW: float = 0.90
_SCRAM_LOW_FLOW: float = 0.25
_ALARM_HIGH_PRESSURE: float = 1.65e7
_ALARM_LOW_PRESSURE: float = 1.30e7
_ALARM_LOW_SUBCOOLING: float = 5.0
_ALARM_HIGH_XENON_WORTH: float = -0.025
_ALARM_XENON_PIT_WORTH: float = -0.030
_ALARM_PRZR_LEVEL_LOW: float = 0.25
_ALARM_PRZR_LEVEL_HIGH: float = 0.75
_SCRAM_HIGH_FUEL_TEMP: float = 1173.15
_SCRAM_HIGH_COOL_TEMP: float = 623.15

_N_NODES: int = 10


async def simulation_tick(state: PlantState, dt: float = 0.1) -> PlantState:
    """Advance plant state by dt seconds — Phase 2 25-step order.

    Args:
        state: current PlantState
        dt: timestep in seconds (default 0.1 s = 100 ms)

    Returns:
        New PlantState advanced by dt.
    """
    s = copy.deepcopy(state)

    if s.eccs_state is None:
        s.eccs_state = ECCSState()

    # ── Step 1: CVCS boron rate-limiting ──────────────────────────────────────
    boron_diff = s.boron_target_ppm - s.boron_ppm
    boron_step = min(abs(boron_diff), BORON_RATE_PPM_PER_S * dt)
    s.boron_ppm += boron_step * (1.0 if boron_diff > 0 else -1.0)

    # ── Step 2: CRDM — rate-limited rod motion ────────────────────────────────
    if not s.scram:
        rod_step = ROD_SPEED_PCT_PER_S * dt
        s.rod_positions = [
            pos + max(-rod_step, min(rod_step, tgt - pos))
            for pos, tgt in zip(s.rod_positions, s.rod_target_positions)
        ]

    # ── Step 3: Axial power shape (cosine; Phase 3 will replace with nodal flux) ──
    s.axial_power_shape = cosine_power_shape(_N_NODES)

    # Current fission power (pre-PKE; used in thermal and heat-flux steps)
    fission_power = s.n * NOMINAL_POWER_W
    dh = decay_heat_power(s.t_since_scram) if s.scram else 0.0

    # ── Step 4: Per-node heat flux (scaled to physical rod-surface flux) ──────
    s.heat_flux = actual_heat_flux_array(s.axial_power_shape, fission_power, _N_NODES) * HEAT_FLUX_SCALE

    # ── Step 5: Critical heat flux ────────────────────────────────────────────
    s.chf = critical_heat_flux(s.flow_fraction, s.pressure, float(s.quality.mean()))

    # ── Step 6: Boiling regime per node ──────────────────────────────────────
    s.boiling_regime = [
        _boiling_regime(float(s.void_fraction[n]), float(s.heat_flux[n]), s.chf)
        for n in range(_N_NODES)
    ]

    # ── Step 7: Heat transfer coefficient per node ────────────────────────────
    s.htc = np.array([
        heat_transfer_coefficient(s.boiling_regime[n], s.flow_fraction, float(s.void_fraction[n]))
        for n in range(_N_NODES)
    ])

    # ── Step 8: Nodal thermal step → t_fuel_axial, t_cool_axial ─────────────
    s.t_fuel_axial, s.t_cool_axial = step_thermal_nodal(
        s.t_fuel_axial, s.t_cool_axial, s.void_fraction, s.axial_power_shape,
        fission_power, s.flow_fraction, dh, s.pressure, s.t_in, dt, htc=s.htc,
    )
    # ── Step 9: Thermodynamic quality per node ────────────────────────────────
    s.quality = np.array([
        thermodynamic_quality(float(s.t_cool_axial[n]), s.pressure)
        for n in range(_N_NODES)
    ])

    # ── Step 10: Void fraction per node (bulk drift-flux + subcooled Saha-Zuber) ─
    t_sat_k = saturation_temp(s.pressure)
    new_void = np.array([
        _void_fraction(float(s.quality[n]), s.pressure)
        + void_fraction_subcooled(float(s.t_cool_axial[n]), t_sat_k, float(s.heat_flux[n]), s.pressure)
        for n in range(_N_NODES)
    ])
    s.void_fraction = np.clip(new_void, 0.0, 0.95)

    # ── Step 11: DNBR ─────────────────────────────────────────────────────────
    s.peak_heat_flux_node = int(np.argmax(s.heat_flux))
    s.dnbr = s.chf / max(float(s.heat_flux[s.peak_heat_flux_node]), 1.0)

    # ── Step 12: Film-boiling nodes ───────────────────────────────────────────
    s.film_boiling_nodes = [n for n in range(_N_NODES) if s.boiling_regime[n] == 'film_boiling']

    # ── Step 13: DNBR low timer → fuel damage (irreversible) ─────────────────
    if s.dnbr < 1.0:
        s.dnbr_low_timer += dt
    else:
        s.dnbr_low_timer = 0.0
    if s.dnbr_low_timer > 5.0:
        s.fuel_damage = True

    # ── Coolant pre-step (before reactivity/PKE) ─────────────────────────────
    # Update s.t_cool only so PKE sub-steps see a current-tick coolant temperature.
    # s.t_fuel is intentionally NOT updated here — the PKE sub-step loop owns fuel
    # temperature (t_fuel_sub) and writes it back to s.t_fuel after integration.
    eccs_flow_pre = s.eccs_state.injection_flow_fraction
    total_flow_pre = s.flow_fraction + eccs_flow_pre
    if total_flow_pre > 1e-6 and eccs_flow_pre > 0.0:
        from physics.constants import M_DOT_NOM_CP_COOL, M_COOL_CP_COOL
        t_in_eff = (s.flow_fraction * s.t_in + eccs_flow_pre * ECCS_WATER_TEMP) / total_flow_pre
        q_transfer = H_TRANSFER_A * (s.t_fuel - s.t_cool)
        q_removed = total_flow_pre * M_DOT_NOM_CP_COOL * (s.t_cool - t_in_eff)
        s.t_cool += (q_transfer - q_removed) / M_COOL_CP_COOL * dt
    else:
        _, s.t_cool = step_thermal(
            s.t_fuel, s.t_cool, fission_power, s.flow_fraction, dh, dt, t_in=s.t_in
        )

    # ── Step 14: Reactivity components ───────────────────────────────────────
    rod_pos = [0.0, 0.0, 0.0, 0.0] if s.scram else s.rod_positions
    s.rho_rod = rod_reactivity(rod_pos)
    s.rho_doppler = doppler_reactivity(s.t_fuel)
    # Moderator feedback: use scalar t_cool to preserve Phase 1 equilibrium point.
    # (compute_reactivity() uses axial average; the loop keeps scalar for stability.)
    s.rho_moderator = moderator_reactivity(s.t_cool)
    s.rho_xenon = xenon_reactivity(s.xenon)
    s.rho_boron = boron_reactivity(s.boron_ppm)
    s.rho_eccs = eccs_reactivity(s.eccs_state, s.t_cool)
    rho_void = void_reactivity(s.void_fraction, s.axial_power_shape)
    rho = (
        s.rho_rod + s.rho_doppler + s.rho_moderator
        + s.rho_xenon + s.rho_boron + s.rho_eccs + rho_void
        + s.ejection_rho
    )

    # ── Step 15: PKE — RK4 with 0.1 ms sub-steps ─────────────────────────────
    pke_state = np.concatenate([[s.n], s.precursors])
    _DT_PKE = 1e-4
    n_sub = max(1, round(dt / _DT_PKE))
    dt_sub = dt / n_sub

    rho_no_doppler = rho - ALPHA_DOPPLER * (s.t_fuel - T_REF_FUEL)
    t_fuel_sub = s.t_fuel

    for _ in range(n_sub):
        rho = rho_no_doppler + ALPHA_DOPPLER * (t_fuel_sub - T_REF_FUEL)
        pke_state = pke_rk4(pke_state, rho, dt_sub)
        n_cur = max(0.0, float(pke_state[0]))
        pke_state[0] = n_cur

        q_fission_sub = n_cur * NOMINAL_POWER_W
        q_xfer_sub = H_TRANSFER_A * (t_fuel_sub - s.t_cool)
        t_fuel_sub += (q_fission_sub - q_xfer_sub) / M_FUEL_CP_FUEL * dt_sub

        if not s.scram and "HI_POWER" not in s.scram_bypasses and n_cur > _SCRAM_HIGH_POWER:
            s.scram = True
            s.scram_cause = "HI_POWER"
            s.t_since_scram = 0.0
            s.ejection_rho = 0.0
            rho_no_doppler = (
                rod_reactivity([0.0, 0.0, 0.0, 0.0])
                + moderator_reactivity(s.t_cool)
                + xenon_reactivity(s.xenon)
                + boron_reactivity(s.boron_ppm)
                + eccs_reactivity(s.eccs_state, s.t_cool)
                + rho_void
            )

    s.n = max(0.0, float(pke_state[0]))
    s.precursors = np.maximum(0.0, pke_state[1:7])
    s.t_fuel = t_fuel_sub  # carry sub-step fuel temp forward; PKE owns t_fuel

    # ── Step 16: Xenon / Iodine ───────────────────────────────────────────────
    s.iodine, s.xenon = rk4_step_xenon(s.iodine, s.xenon, s.n, dt)

    # ── Step 17: Decay heat timer ─────────────────────────────────────────────
    if s.scram:
        s.t_since_scram += dt

    # ── Step 18: T_in — dynamic cold-leg temperature ──────────────────────────
    if s.offsite_power:
        secondary_flow = 1.0
    elif any(d.state == "running" for d in s.diesel_states):
        secondary_flow = 0.2
    else:
        secondary_flow = 0.0
    t_in_target = T_COOLANT_INLET + (s.t_cool - T_COOLANT_INLET) * (1.0 - secondary_flow)
    s.t_in += (t_in_target - s.t_in) / TAU_STEAM_GENERATOR * dt

    # ── Step 19: Pumps — coastdown / natural circulation ─────────────────────
    s.pump_speeds = step_pumps(s.pump_speeds, s.pumps, s.offsite_power, dt)
    s.flow_fraction = total_flow_fraction(s.pump_speeds, s.t_fuel - s.t_cool)

    # ── LOCA blowdown, ECCS ramp, reflood ────────────────────────────────────
    if s.loca_active:
        s = update_loca(s, dt)

    # ── Step 20: Diesels ──────────────────────────────────────────────────────
    s.diesel_states = step_diesels(s.diesel_states, s.diesel_start_signals, dt)

    # ── Step 21: ECCS actuation ───────────────────────────────────────────────
    s.eccs_state = step_eccs(s.pressure, s.eccs_armed, s.t_cool, s.flow_fraction)

    # ── Step 22: PORV logic ───────────────────────────────────────────────────
    if not s.porv_stuck_open:
        if s.pressure > PORV_OPEN_SETPOINT:
            s.porv_open = True
        elif s.pressure <= PORV_CLOSE_SETPOINT:
            s.porv_open = False

    # ── Step 23: Pressurizer ──────────────────────────────────────────────────
    s.pressure = step_pressurizer(
        s.pressure, s.t_cool, s.flow_fraction, dt,
        s.przr_heater_fraction, s.przr_spray_fraction, s.porv_open,
    )
    s.pressurizer_level = step_pressurizer_level(
        s.pressurizer_level, s.t_cool, s.pressure,
        s.eccs_state.injection_flow_fraction, dt,
    )
    if s.porv_open:
        s.pressurizer_level = max(0.0, s.pressurizer_level - PORV_LEVEL_DRAIN * dt)

    # ── Step 24: Alarms — Phase 1 inline + Phase 2 from alarms.py ────────────
    alarms: list[str] = []
    if s.n > _ALARM_HIGH_POWER:
        alarms.append("HIGH_POWER")
    if s.t_fuel > _ALARM_FUEL_TEMP_K:
        alarms.append("HIGH_FUEL_TEMP")
    if s.t_cool > _ALARM_COOL_TEMP_K:
        alarms.append("HIGH_COOL_TEMP")
    if s.flow_fraction < _ALARM_LOW_FLOW:
        alarms.append("LOW_FLOW")
    if s.pressure > _ALARM_HIGH_PRESSURE:
        alarms.append("HIGH_PRESSURE")
    if s.pressure < _ALARM_LOW_PRESSURE:
        alarms.append("LOW_PRESSURE")
    if s.eccs_state.hpsi_active or s.eccs_state.lpsi_active:
        alarms.append("ECCS_ACTIVE")
    if s.scram:
        alarms.append("SCRAM")
    if s.t_fuel >= T_CLADDING_FAILURE:
        alarms.append("CLADDING_FAIL")
    if not s.offsite_power:
        alarms.append("LOOP")
        if all(d.state == "failed" for d in s.diesel_states):
            alarms.append("SBO")
    if any(d.state == "failed" for d in s.diesel_states):
        alarms.append("DIESEL_FAILED")
    if not all(s.pumps):
        alarms.append("PUMP_TRIP")
    if s.ejection_rho > 1e-4:
        alarms.append("ROD_EJECTION")
    if s.pressure < ECCS_HPSI_THRESHOLD:
        alarms.append("LOCA")
    t_hot_leg_k = 2.0 * s.t_cool - s.t_in
    subcooling_c = t_sat_celsius(s.pressure) - (t_hot_leg_k - 273.15)
    if subcooling_c < _ALARM_LOW_SUBCOOLING:
        alarms.append("LOW_SUBCOOLING")
    xe_worth = xenon_reactivity(s.xenon)
    if xe_worth < _ALARM_HIGH_XENON_WORTH:
        alarms.append("HIGH_XENON")
    if s.scram and xe_worth < _ALARM_XENON_PIT_WORTH:
        alarms.append("XENON_PIT")
    if s.pressurizer_level < _ALARM_PRZR_LEVEL_LOW:
        alarms.append("LOW_PRZR_LEVEL")
    if s.pressurizer_level > _ALARM_PRZR_LEVEL_HIGH:
        alarms.append("HIGH_PRZR_LEVEL")
    if s.porv_stuck_open:
        alarms.append("PORV_STUCK_OPEN")
    elif s.porv_open:
        alarms.append("PORV_OPEN")

    # Phase 2 alarms
    for name, predicate, _severity in PHASE2_ALARMS:
        if predicate(s):
            alarms.append(name)

    s.alarms = alarms

    # SCRAM checks (pressure, flow, temperature)
    bypasses = s.scram_bypasses
    if not s.scram and "HI_PRESSURE" not in bypasses and s.pressure > PRESSURE_SCRAM_HIGH:
        s.scram = True
        s.scram_cause = "HI_PRESSURE"
        s.t_since_scram = 0.0
    if not s.scram and "LO_PRESSURE" not in bypasses and s.pressure < PRESSURE_SCRAM_LOW:
        s.scram = True
        s.scram_cause = "LO_PRESSURE"
        s.t_since_scram = 0.0
    if not s.scram and "LO_FLOW" not in bypasses and s.flow_fraction < _SCRAM_LOW_FLOW:
        s.scram = True
        s.scram_cause = "LO_FLOW"
        s.t_since_scram = 0.0
    if not s.scram and "HI_FUEL_TEMP" not in bypasses and s.t_fuel > _SCRAM_HIGH_FUEL_TEMP:
        s.scram = True
        s.scram_cause = "HI_FUEL_TEMP"
        s.t_since_scram = 0.0
    if not s.scram and "HI_COOL_TEMP" not in bypasses and s.t_cool > _SCRAM_HIGH_COOL_TEMP:
        s.scram = True
        s.scram_cause = "HI_COOL_TEMP"
        s.t_since_scram = 0.0

    s.t += dt
    return s
