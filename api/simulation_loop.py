"""100 ms simulation tick — orchestrates all physics and plant subsystems."""

from __future__ import annotations

import copy

import numpy as np

from api.state import PlantState
from physics.pke_solver import rk4_step as pke_rk4
from physics.thermal import step_thermal
from physics.xenon import rk4_step_xenon, xenon_reactivity
from physics.decay_heat import decay_heat_power
from physics.reactivity import (
    rod_reactivity,
    doppler_reactivity,
    moderator_reactivity,
    boron_reactivity,
)
from physics.constants import (
    NOMINAL_POWER_W, ECCS_WATER_TEMP, T_COOLANT_INLET,
    T_CLADDING_FAILURE, ECCS_HPSI_THRESHOLD,
    H_TRANSFER_A, M_FUEL_CP_FUEL, ALPHA_DOPPLER, T_REF_FUEL,
    TAU_STEAM_GENERATOR, BORON_RATE_PPM_PER_S, ROD_SPEED_PCT_PER_S,
    PORV_OPEN_SETPOINT, PORV_CLOSE_SETPOINT, PORV_LEVEL_DRAIN,
    PRESSURE_SCRAM_HIGH, PRESSURE_SCRAM_LOW,
)
from physics.thermal import t_sat_celsius
from plant.pumps import step_pumps, total_flow_fraction
from plant.diesels import step_diesels
from plant.eccs import step_eccs, eccs_reactivity, ECCSState
from plant.pressurizer import step_pressurizer, step_pressurizer_level

# Alarm setpoints (SI)
_ALARM_HIGH_POWER: float = 1.10          # fraction of nominal
_SCRAM_HIGH_POWER: float = 1.20
_ALARM_FUEL_TEMP_K: float = 1073.15     # 800 °C
_ALARM_COOL_TEMP_K: float = 590.15      # 317 °C — reached ~40 s into LOOP without pump restart
_ALARM_LOW_FLOW: float = 0.90
_SCRAM_LOW_FLOW: float = 0.25          # < 25 % nominal flow → SCRAM
_ALARM_HIGH_PRESSURE: float = 1.65e7   # 165 bar
_ALARM_LOW_PRESSURE: float = 1.30e7    # 130 bar
_ALARM_LOW_SUBCOOLING: float = 5.0     # °C — nominal subcooling is ~11 °C; alarm below 5 °C
_ALARM_HIGH_XENON_WORTH: float = -0.025  # dk/k — above full-power equilibrium (~-0.021)
_ALARM_XENON_PIT_WORTH: float = -0.030   # dk/k — deep pit, restart blocked
_ALARM_PRZR_LEVEL_LOW: float = 0.25      # fraction — inventory low (LOCA indicator)
_ALARM_PRZR_LEVEL_HIGH: float = 0.75     # fraction — pressurizer nearly solid
_SCRAM_HIGH_FUEL_TEMP: float = 1173.15  # K (900 °C — approaching cladding damage at 1200 °C)
_SCRAM_HIGH_COOL_TEMP: float = 623.15   # K (350 °C — 50 K above nominal, before bulk boiling)


async def simulation_tick(state: PlantState, dt: float = 0.1) -> PlantState:
    """Advance plant state by dt seconds.

    Call order: reactivity → PKE → thermal → xenon → decay_heat →
                pumps → diesels → ECCS → pressurizer → alarms → return.

    Args:
        state: current PlantState
        dt: timestep in seconds (default 0.1 s = 100 ms)

    Returns:
        New PlantState advanced by dt.
    """
    s = copy.deepcopy(state)

    # Ensure eccs_state is an ECCSState object (tolerates None from bare constructor)
    if s.eccs_state is None:
        s.eccs_state = ECCSState()

    # 1a. CVCS boron rate-limiting — advance actual toward target at ≤10 ppm/min
    boron_diff = s.boron_target_ppm - s.boron_ppm
    boron_step = min(abs(boron_diff), BORON_RATE_PPM_PER_S * dt)
    s.boron_ppm += boron_step * (1.0 if boron_diff > 0 else -1.0)

    # 1b. CRDM — advance actual rod positions toward operator target at ≤ ROD_SPEED_PCT_PER_S.
    # Rods are not rate-limited during SCRAM (they drop by gravity/spring, essentially instant).
    if not s.scram:
        rod_step = ROD_SPEED_PCT_PER_S * dt
        s.rod_positions = [
            pos + max(-rod_step, min(rod_step, tgt - pos))
            for pos, tgt in zip(s.rod_positions, s.rod_target_positions)
        ]

    # 1c. Reactivity — insert rods fully on SCRAM; store individual components for display
    rod_pos = [0.0, 0.0, 0.0, 0.0] if s.scram else s.rod_positions
    s.rho_rod      = rod_reactivity(rod_pos)
    s.rho_doppler  = doppler_reactivity(s.t_fuel)
    s.rho_moderator = moderator_reactivity(s.t_cool)
    s.rho_xenon    = xenon_reactivity(s.xenon)
    s.rho_boron    = boron_reactivity(s.boron_ppm)
    s.rho_eccs     = eccs_reactivity(s.eccs_state, s.t_cool)
    rho = (
        s.rho_rod + s.rho_doppler + s.rho_moderator
        + s.rho_xenon + s.rho_boron + s.rho_eccs
        + s.ejection_rho
    )

    # 2. PKE — sub-step with 0.1 ms inner steps to keep RK4 within stability region.
    # The prompt lifetime is 2e-5 s; dt=0.1 s is far too large for a single RK4 step.
    #
    # SCRAM is checked inline each sub-step (not just at end of tick) so that prompt-
    # supercritical excursions are terminated within ~0.1 ms rather than 100 ms.
    # Doppler is also updated each sub-step so fuel-temperature feedback can self-limit
    # fast transients (rod ejection, accidental large rod withdrawals).
    pke_state = np.concatenate([[s.n], s.precursors])
    _DT_PKE = 1e-4  # 0.1 ms — stable up to |rho| ≈ 0.25; ROD_WORTH_MIN=-0.10 needs ≤ 0.52 ms,
                    # but 0.1 ms gives a 5× safety margin for transient spikes.
    n_sub = max(1, round(dt / _DT_PKE))
    dt_sub = dt / n_sub

    # Separate the Doppler contribution so it can be updated each sub-step
    rho_no_doppler = rho - ALPHA_DOPPLER * (s.t_fuel - T_REF_FUEL)
    t_fuel_sub = s.t_fuel  # inline fuel temperature tracker

    for _ in range(n_sub):
        # Recompute Doppler with current inline fuel temperature
        rho = rho_no_doppler + ALPHA_DOPPLER * (t_fuel_sub - T_REF_FUEL)

        pke_state = pke_rk4(pke_state, rho, dt_sub)
        n_cur = max(0.0, float(pke_state[0]))
        pke_state[0] = n_cur

        # Inline fuel temperature update (forward-Euler; accurate since τ_fuel >> dt_sub)
        q_fission_sub = n_cur * NOMINAL_POWER_W
        q_xfer_sub = H_TRANSFER_A * (t_fuel_sub - s.t_cool)
        t_fuel_sub += (q_fission_sub - q_xfer_sub) / M_FUEL_CP_FUEL * dt_sub

        # Inline SCRAM — fire immediately when power hits SCRAM setpoint rather than
        # waiting until the end of the 100 ms tick (prevents billions-of-% excursions).
        if not s.scram and "HI_POWER" not in s.scram_bypasses and n_cur > _SCRAM_HIGH_POWER:
            s.scram = True
            s.scram_cause = "HI_POWER"
            s.t_since_scram = 0.0
            # Ejected rod is permanently gone, but the remaining banks fully insert.
            # ROD_WORTH_MIN (-0.10) is deeply negative even without the one missing rod.
            s.ejection_rho = 0.0
            # Recompute rho with rods fully inserted
            rho_no_doppler = (
                rod_reactivity([0.0, 0.0, 0.0, 0.0])
                + moderator_reactivity(s.t_cool)
                + xenon_reactivity(s.xenon)
                + boron_reactivity(s.boron_ppm)
                + eccs_reactivity(s.eccs_state, s.t_cool)
            )

    s.n = max(0.0, float(pke_state[0]))
    s.precursors = np.maximum(0.0, pke_state[1:7])
    # t_fuel_sub is discarded — it was used only for inline Doppler feedback during
    # sub-stepping. The definitive fuel temperature is advanced by the thermal step below,
    # which prevents double-counting the dt interval.

    # 2.5. T_in — dynamic cold-leg temperature (secondary heat-sink degradation).
    # Main feedwater pumps trip with offsite power loss (LOOP/SBO).  Emergency
    # feedwater (~20 % of nominal) is available if at least one diesel is running.
    # Without any feedwater the steam generators can no longer cool the primary, so
    # T_in rises toward T_cool on the steam-generator thermal time constant.
    if s.offsite_power:
        secondary_flow = 1.0
    elif any(d.state == "running" for d in s.diesel_states):
        secondary_flow = 0.2   # emergency feedwater on diesel
    else:
        secondary_flow = 0.0   # SBO — complete secondary heat-sink loss
    t_in_target = T_COOLANT_INLET + (s.t_cool - T_COOLANT_INLET) * (1.0 - secondary_flow)
    s.t_in += (t_in_target - s.t_in) / TAU_STEAM_GENERATOR * dt

    # 3. Thermal — fission power + decay heat (decay heat only post-SCRAM).
    # ECCS injection adds cold flow; blend its temperature with the dynamic T_in to get
    # an effective inlet temperature, then pass the combined flow fraction to step_thermal.
    fission_power = s.n * NOMINAL_POWER_W
    dh = decay_heat_power(s.t_since_scram) if s.scram else 0.0
    eccs_flow = s.eccs_state.injection_flow_fraction
    total_flow = s.flow_fraction + eccs_flow
    if total_flow > 1e-6 and eccs_flow > 0.0:
        # Blend dynamic pump inlet (s.t_in) with ECCS cold injection
        t_in_eff = (s.flow_fraction * s.t_in + eccs_flow * ECCS_WATER_TEMP) / total_flow
        from physics.constants import M_DOT_NOM_CP_COOL, M_COOL_CP_COOL
        q_transfer = H_TRANSFER_A * (s.t_fuel - s.t_cool)
        q_removed = total_flow * M_DOT_NOM_CP_COOL * (s.t_cool - t_in_eff)
        d_tf = (fission_power + dh - q_transfer) / M_FUEL_CP_FUEL
        d_tc = (q_transfer - q_removed) / M_COOL_CP_COOL
        s.t_fuel = s.t_fuel + d_tf * dt
        s.t_cool = s.t_cool + d_tc * dt
    else:
        s.t_fuel, s.t_cool = step_thermal(
            s.t_fuel, s.t_cool, fission_power, s.flow_fraction, dh, dt, t_in=s.t_in
        )

    # 4. Xenon / Iodine — RK4 step
    s.iodine, s.xenon = rk4_step_xenon(s.iodine, s.xenon, s.n, dt)

    # 5. Decay heat — advance shutdown timer
    if s.scram:
        s.t_since_scram += dt

    # 6. Pumps — coastdown or full speed; recompute flow
    s.pump_speeds = step_pumps(s.pump_speeds, s.pumps, s.offsite_power, dt)
    s.flow_fraction = total_flow_fraction(s.pump_speeds, s.t_fuel - s.t_cool)

    # 7. Diesels
    s.diesel_states = step_diesels(s.diesel_states, s.diesel_start_signals, dt)

    # 8. ECCS
    s.eccs_state = step_eccs(s.pressure, s.eccs_armed, s.t_cool, s.flow_fraction)

    # 9a. PORV — auto-actuation logic (before pressurizer step so p_target is correct this tick)
    if not s.porv_stuck_open:
        if s.pressure > PORV_OPEN_SETPOINT:
            s.porv_open = True
        elif s.pressure <= PORV_CLOSE_SETPOINT:
            s.porv_open = False

    # 9. Pressurizer — PORV open flag shifts p_target down so the lag drives pressure
    #    downward rather than opposing the drain (same flow-deficit pattern as LOCA).
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

    # 10. Alarms
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
    s.alarms = alarms

    # Check pressure, flow, and temperature SCAMs (power SCRAM is handled inline in PKE loop).
    # Each channel is individually bypassable via s.scram_bypasses.
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
