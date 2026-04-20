# Nuclear Reactor Simulator — Project Specification

## Overview

A web-based nuclear reactor simulator with real-time physics, an interactive control panel UI,
and triggered accident scenarios. The goal is educational realism: physically meaningful behavior
without requiring a supercomputer.

**Reactor type:** Generic Pressurized Water Reactor (PWR) — chosen for simplicity and familiarity.
(RBMK or BWR can be added as alternate reactor configs later.)

---

## Physics Engine

### 1. Point Kinetics Equations (PKE)

The core neutronics model. Solve using 4th-order Runge-Kutta (RK4).

The prompt neutron lifetime (Λ = 2×10⁻⁵ s) requires sub-stepping within each simulation tick:
the PKE integrator uses 0.1 ms inner timesteps (1000 steps per 100 ms tick), with inline Doppler
feedback and SCRAM check at each sub-step for stability.

Six delayed neutron groups. State vector:

- `n(t)` — normalized neutron population (= relative power level)
- `C_i(t)` — precursor concentration for group i (i = 1..6)

Parameters (PWR values):

| Group | Beta_i   | Lambda_i (1/s) |
|-------|----------|----------------|
| 1     | 0.000215 | 0.0124         |
| 2     | 0.001424 | 0.0305         |
| 3     | 0.001274 | 0.111          |
| 4     | 0.002568 | 0.301          |
| 5     | 0.000748 | 1.14           |
| 6     | 0.000273 | 3.01           |

- Total beta = 0.0065
- Prompt neutron lifetime Λ = 2e-5 s

ODEs:

```
dn/dt = ((rho - beta) / Lambda) * n + sum(lambda_i * C_i)
dC_i/dt = (beta_i / Lambda) * n - lambda_i * C_i
```

### 2. Reactivity Model

Total reactivity = sum of all contributions:

```
rho_total = rho_control_rods + rho_doppler + rho_moderator + rho_xenon + rho_boron + rho_eccs
```

- **Control rods:** S-curve (sinusoidal integral) map from average bank position (0–100%) to
  reactivity worth. Range: -0.10 dk/k (all rods fully inserted) to +0.14276 dk/k (fully withdrawn).
  Formula: `worth_fraction = f - sin(2πf)/(2π)` where f = position/100.
  Worth is chosen so that 75% withdrawal + equilibrium Xe + 1000 ppm boron = zero net reactivity.
  Real PWR total rod worth (shutdown margin) is typically 6–15% dk/k; this range is realistic.

- **Doppler (fuel temp feedback):** α_D = -2.5×10⁻⁵ dk/k/K × (T_fuel − T_ref_fuel)

- **Moderator (coolant temp feedback):** α_M = -1.5×10⁻⁴ dk/k/K × (T_cool − T_ref_cool)
  (Mid-range PWR value; range for real PWRs is -1×10⁻⁴ to -4×10⁻⁴ dk/k/K)

- **Xenon-135:** Separate Xe/I kinetics model (see Section 4)

- **Boron:** Linear: rho_boron = -1×10⁻⁴ × boron_ppm

- **ECCS:** Cold borated water injection (2500 ppm) adds negative reactivity during actuation

### 3. Thermal Hydraulics

Lumped parameter model (one-node fuel, one-node coolant). Explicit Euler integration is
stable here because thermal time constants (~5 s fuel, ~2 s coolant) are large relative
to the 0.1 s timestep.

```
M_fuel_cp * dT_fuel/dt = Q_fission + Q_decay - Q_transfer
M_cool_cp * dT_cool/dt = Q_transfer - Q_removed

Q_transfer = h·A · (T_fuel - T_cool)
Q_removed  = flow_fraction · ṁ·cp_cool · (T_cool - T_in)
```

Decay heat is placed in the fuel node (fission-product decay occurs in fuel rods before
heat transfers to coolant).

**T_in (cold-leg inlet) — dynamic secondary heat-sink model:**

- Offsite power available (full feedwater): T_in = T_COOLANT_INLET (270 °C, nominal)
- Diesel running (emergency feedwater, 20% nominal): T_in held at nominal
- No feedwater (SBO / all diesels failed): T_in drifts toward T_cool with τ = 60 s
  (steam generator heat-sink lost; secondary side heats up)

Parameters (all temperatures in Kelvin internally; °C values shown for reference):

- T_ref_cool = 573.15 K (300°C nominal coolant temp)
- T_ref_fuel = 873.15 K (600°C nominal fuel centerline temp)
- T_in = 543.15 K (270°C coolant inlet temperature)
- Nominal power = 3000 MWt
- h·A = 1×10⁷ W/K
- M_fuel·cp_fuel = 5×10⁷ J/K → τ_fuel ≈ 5 s
- M_cool·cp_cool = 2.2×10⁸ J/K, ṁ·cp_cool = 1×10⁸ W/K → τ_cool ≈ 2 s

### 4. Xenon-135 / Iodine-135 Kinetics

```
dI/dt  = gamma_I · Σ_f · phi − lambda_I · I
dXe/dt = gamma_Xe · Σ_f · phi + lambda_I · I − lambda_Xe · Xe − sigma_Xe · phi · Xe

rho_xenon = -sigma_Xe · Xe · phi_0 / (nu · Σ_f · phi_0)   [normalized]
```

- lambda_I  = 2.87×10⁻⁵ /s  (half-life ~6.7 hr)
- lambda_Xe = 2.09×10⁻⁵ /s  (half-life ~9.2 hr)
- sigma_Xe  = 2.6×10⁻²² m²  (2.6 Megabarns — huge absorption cross-section)
- gamma_I   = 0.061  (cumulative iodine fission yield)
- gamma_Xe  = 0.002  (direct xenon fission yield)
- phi_0     = 3×10¹⁷ n/m²/s  (reference full-power thermal flux)
- Σ_f (core-averaged macroscopic) = 30 /m, ν = 2.4

This produces the xenon pit (reactor cannot restart for ~30–40 hr after shutdown).

### 5. Decay Heat

ANS-5.1 standard approximation (sum of exponentials):

```
Q_decay(t) = Q0 * sum_k(A_k * exp(-alpha_k * t))
```

Use 11-group fit (U-235 infinite irradiation). Active even after SCRAM. ~6.3% at t=0, ~1.1% at 1 hour.
Critical for LOOP/SBO scenarios.

---

## Plant Systems

### Coolant Pumps (4 total, each 25% flow)

- State: ON / OFF / COASTING
- Coastdown: exponential decay of pump speed, time constant τ = 30 s
- On LOOP: all pumps coast down (no AC power)
- On diesel start: pumps can restart (2 diesels, each powers 2 pumps)
- Natural circulation: when total pump flow < 3%, thermosiphon natural circulation
  kicks in, scaling with buoyancy head: `flow_nc = (delta_T / 30 K)` fraction of nominal.
  This prevents complete flow stagnation in LOOP/SBO.

### Diesel Generators (2 total)

- State: STANDBY / STARTING / RUNNING / FAILED
- Start delay: 10–15 s (randomized) after signal
- Failure: state set to FAILED immediately (for SBO scenario); probability-based failure
  not currently modeled but can be added for challenge mode
- Powers: emergency buses, pump motors, ECCS

### Emergency Core Cooling System (ECCS)

- High-pressure injection (HPSI): activates when pressure ≤ 100 bar
- Low-pressure injection (LPSI): activates when pressure ≤ 20 bar
- Activates automatically on low pressure signal (when armed; operator can inhibit)
- Effect: injects cold water (20°C, 2500 ppm boron) at 20% (HPSI) or 50% (LPSI) of nominal flow
- Reactivity effect: negative (cold borated water)

### Pressurizer

- Maintains RCS pressure ~155 bar nominal
- First-order lag model (τ = 10 s) with flow-deficit term for LOCA depressurization
- Setpoint shifts with coolant temperature: ±1×10⁵ Pa/K
- SCRAM signal at >170 bar or <120 bar
- Operator controls: heater (0–1 demand, raises pressure) and spray (0–1 demand, lowers pressure)
- PORV (Pilot-Operated Relief Valve): auto-opens at >157 bar, recloses at ≤155 bar; can be
  manually opened or latched stuck-open (TMI-2 scenario)
- Pressurizer level tracked as fraction 0–1; tracks thermal expansion and inventory changes
  (LOCA drain proportional to pressure deficit; ECCS makeup)

---

## Accident Scenarios

Scenarios are triggered via POST /scenario. All state mutations happen inline in the server;
the `accidents/` module files are available as standalone helpers used by tests.

### LOOP — Loss of Offsite Power

**Trigger:** External grid fails. All AC power lost instantly.
**Sequence:**

1. t=0: Grid disconnect. All 4 coolant pumps begin coastdown (offsite_power=False).
2. t=0: Diesel start signal sent automatically to both diesels.
3. t=10–15 s: Diesels (if successful) come online, restore partial AC.
4. t=0–30 s: Coolant flow decays; core temperature rises.
5. SCRAM triggered automatically on low flow / high temp alarms.
6. Decay heat must be removed by natural circulation or ECCS.

**Player task:** Monitor temps, manage diesel starts, maintain coolant inventory.
**Failure condition:** T_fuel > 1200°C / 1473 K (cladding failure threshold).

### Station Blackout (SBO)

**Same as LOOP but both diesels are immediately set to FAILED.** Only battery-backed systems
remain (instrumentation, valve control). No pump power available. Natural circulation only.
Decay heat builds until ECCS injection (if ECCS armed).

### LOCA — Loss of Coolant Accident

**Trigger:** Large break in primary loop simulated by instantly dropping RCS pressure to 50 bar.
**Sequence:**

1. t=0: Pressure set to 5×10⁶ Pa (50 bar), below SCRAM_LOW (120 bar) and HPSI (100 bar).
2. SCRAM triggered by low pressure at next tick.
3. ECCS HPSI activates immediately (pressure < 100 bar).

**Simplification note:** The void-formation positive reactivity spike seen during rapid
depressurization is not modeled (no void reactivity coefficient). The physics captures
the pressure transient and ECCS response, but not the prompt power spike from voiding.

**Player task:** Verify ECCS operation, monitor core cooling.

### Control Rod Ejection

**Trigger:** Mechanical failure ejects bank A fully (to 100% withdrawal) in one tick.
**Sequence:**

1. Bank A moved to 100% withdrawal; `ejection_rho` = 0.01 × (100 − original_pos)/100 dk/k.
2. Positive reactivity insertion produces prompt power spike.
3. Doppler feedback activates — self-limits if within design basis.
4. High-power SCRAM signal (at >120% power, inline during PKE sub-stepping).
5. On SCRAM: ejection_rho is zeroed and all rods insert.

**Player task:** Observe the self-limiting nature of Doppler feedback (or fail if reactivity insertion too large).

### Stuck-Open PORV

**Trigger:** PORV opens but fails to reclose — slow primary-side bleed (TMI-2 initiating event).
**Sequence:**

1. `porv_open = True`, `porv_stuck_open = True` — PORV latched open.
2. Pressure bleeds down ~2 bar/min via pressurizer flow-deficit term.
3. SCRAM fires on low pressure (<120 bar) in ~2.5 min unless operator intervenes.
4. ECCS HPSI actuates at ≤100 bar.

**Operator recovery:** Close the block valve via `porv_stuck_open = False` in `/control`.
**Player task:** Recognize the slow bleed, isolate PORV before SCRAM/ECCS actuation.

### Xenon Pit

**Trigger:** Simulates 3 hours post-shutdown by fast-forwarding Xe/I kinetics offline.
**Sequence:**

1. Xe/I concentrations integrated at n=0 for 3 hr (180 RK4 steps of 60 s each).
2. Emergency boration: boron raised to 1500 ppm (CVCS procedure).
3. State set to SCRAM, n ≈ 0, precursors zeroed.
4. At this point: rho_rods(100%) ≈ +0.143, rho_Xe ≈ -0.033, rho_B(1500 ppm) = -0.150
   → total ≈ -0.040 dk/k — subcritical even at full rod withdrawal.
5. Xe clears at ~30–40 hr — restart becomes possible after boron dilution.

**Educational goal:** Demonstrate why operators cannot rapidly restart after shutdown.

---

## User Interface

### Main Control Panel (HTML/CSS/JS)

- Large central power meter (0–120% of nominal), SVG arc gauge
- Coolant temperature gauges (T_fuel, T_cool), SVG tube gauges
- Pressure gauge (RCS, 0–200 bar), SVG arc gauge with 120–170 bar safe zone
- Control rod position display (% withdrawn, 4 individual bank sliders)
- Coolant flow rate indicator
- Xenon worth bar
- Alarm annunciator grid (18 tiles, color-coded: yellow=caution, red=emergency)

### Trend Plots (Chart.js, real-time)

- Power (% nominal) vs. time — rolling 300-point history (~30 s at 10 Hz)
- T_fuel and T_cool vs. time
- Total reactivity & xenon worth vs. time

### Control Inputs

- Rod bank sliders (banks A, B, C, D — 0–100% withdrawn)
- Boron injection input (0–2000 ppm)
- Manual SCRAM button
- Pump start/stop buttons (4 pumps)
- Diesel generator START buttons (2 diesels)
- ECCS arm/inhibit toggle

### Scenario Panel

- Dropdown: select accident to trigger (LOOP, SBO, LOCA, ROD_EJECTION, STUCK_OPEN_PORV, XENON_PIT)
- "TRIGGER" button
- RESET button (returns to hot standby at 100% power)
- FULL POWER button (restores nominal rod/pump state)
- COLD STOP button (SCRAM + insert rods)

### Client-side Protections (configurable)

- Power >115%: auto SCRAM
- Pressure >170 bar or <120 bar: auto SCRAM
- Flow <25%: auto SCRAM

---

## API (FastAPI + WebSocket)

```
GET  /               → Serve frontend HTML
WS   /ws             → Real-time state streaming (100 ms tick, 10 Hz)
POST /control        → Accept control inputs (rods, pumps, etc.)
POST /scenario       → Trigger named accident scenario (loop, sbo, loca, rod_ejection, stuck_open_porv, xenon_pit)
POST /reset          → Reset to hot standby
```

WebSocket message format (JSON, every 100 ms):

```json
{
  "t": 1234.5,
  "power_pct": 98.3,
  "t_fuel": 612.4,
  "t_cool": 305.1,
  "t_hot_leg": 310.2,
  "t_in": 270.0,
  "pressure": 154.8,
  "flow_pct": 100.0,
  "xenon_worth": -0.0021,
  "rho_total": 0.000012,
  "rho_rod": 0.012,
  "rho_doppler": -0.005,
  "rho_moderator": -0.003,
  "rho_boron": -0.100,
  "rho_eccs": 0.0,
  "rod_position": [75.0, 75.0, 75.0, 75.0],
  "rod_target_position": [75.0, 75.0, 75.0, 75.0],
  "pumps": [true, true, true, true],
  "diesels": ["running", "standby"],
  "alarms": ["LOOP", "LOW_FLOW"],
  "scram": false,
  "scram_cause": null,
  "scram_bypasses": [],
  "eccs_armed": true,
  "eccs_hpsi": false,
  "eccs_lpsi": false,
  "boron_ppm": 1000.0,
  "boron_target_ppm": 1000.0,
  "subcooling": 18.4,
  "pressurizer_level": 50.0,
  "przr_heater_pct": 0.0,
  "przr_spray_pct": 0.0,
  "porv_open": false,
  "porv_stuck_open": false
}
```

All temperatures in °C, pressure in bar, pressurizer_level in % (unit conversions in `_state_to_ws`).
`t_hot_leg` = 2×T_cool − T_in (estimated hot-leg temperature). `subcooling` = T_sat(pressure) − T_hot_leg.

POST /control accepts (all fields optional):

```json
{
  "rod_positions": [75, 75, 75, 75],
  "pumps": [true, true, true, true],
  "boron_ppm": 1000,
  "scram": false,
  "eccs_armed": true,
  "diesel_start_signals": [true, false],
  "przr_heater_fraction": 0.0,
  "przr_spray_fraction": 0.0,
  "porv_open": false,
  "porv_stuck_open": false,
  "scram_bypasses": []
}
```

`rod_positions` sets rod target positions; CRDM rate-limits actual movement to 5 %/s (SCRAM overrides instantly).
`scram_bypasses` is a list of SCRAM channel names to suppress (e.g. `["HIGH_PRESSURE", "LOW_FLOW"]`).

---

## File Layout

```
nuclear-sim/
├── CLAUDE.md
├── SPEC.md
├── requirements.txt
├── physics/
│   ├── __init__.py
│   ├── constants.py         # All SI constants — single source of truth
│   ├── pke_solver.py        # Point kinetics RK4 solver
│   ├── thermal.py           # Fuel/coolant thermal model (Euler, stable at 0.1 s)
│   ├── xenon.py             # Xe-135 / I-135 kinetics
│   ├── decay_heat.py        # ANS-5.1 decay heat
│   └── reactivity.py        # Reactivity component summation
├── plant/
│   ├── __init__.py
│   ├── pumps.py             # Coolant pump models + natural circulation
│   ├── diesels.py           # Diesel generator state machine
│   ├── pressurizer.py       # Pressure model
│   └── eccs.py              # Emergency core cooling (HPSI/LPSI)
├── accidents/
│   ├── __init__.py
│   ├── loop.py              # Loss of offsite power (used in tests)
│   ├── sbo.py               # Station blackout (used in tests)
│   ├── loca.py              # Loss of coolant (used in tests)
│   ├── rod_ejection.py      # Control rod ejection (used in tests)
│   └── xenon_pit.py         # Xenon poisoning demo (used in tests)
│   # Note: stuck_open_porv scenario is dispatched inline in api/server.py (no separate helper)
├── api/
│   ├── __init__.py
│   ├── server.py            # FastAPI app + WebSocket + scenario dispatch
│   ├── state.py             # PlantState dataclass + default_state()
│   └── simulation_loop.py   # 100 ms tick orchestrator
├── tests/
│   ├── test_pke.py
│   ├── test_thermal.py
│   ├── test_xenon.py
│   ├── test_plant.py
│   ├── test_loop.py
│   ├── test_eccs_pressurizer.py   # ECCS actuation, pressurizer level, PORV logic
│   ├── test_accidents.py
│   └── test_reactivity.py
└── frontend/
    ├── panel.html           # Main UI (CSS + JS inline)
    ├── panel.css
    └── panel.js             # WebSocket client + Chart.js
```

---

## Requirements (requirements.txt)

```
fastapi
uvicorn[standard]
numpy
websockets
```

---

## Implementation Order (Claude Code Sessions)

1. `physics/constants.py` — all SI constants, single source of truth
2. `physics/pke_solver.py` — PKE + RK4 + unit tests
3. `physics/thermal.py` — thermal model
4. `physics/xenon.py` + `physics/decay_heat.py`
5. `physics/reactivity.py` — combines all feedback
6. `plant/` — pumps (+ natural circulation), diesels, ECCS, pressurizer
7. `api/state.py` + `api/simulation_loop.py` — tick order: CVCS boron → CRDM → reactivity → PKE (RK4, 0.1 ms sub-steps with inline Doppler + SCRAM) → T_in feedback → thermal → xenon → decay heat → pumps → diesels → ECCS → PORV → pressurizer → pressurizer level → alarms → end-of-tick SCRAM channels (high/low pressure, low flow, high T_fuel >900 °C, high T_cool >350 °C; each channel bypassable)
8. `api/server.py` — FastAPI + WebSocket + scenario dispatch
9. `accidents/` — each scenario file (for tests)
10. `frontend/` — HTML panel + Chart.js trends
11. Integration + end-to-end test

---

## Acceptance Criteria

- [ ] Reactor reaches steady state at 100% power from hot standby
- [ ] SCRAM drops power to <5% within 10 s (physics-limited by delayed neutron decay)
- [ ] LOOP scenario: coolant temp rises to alarm threshold within 60 s without pump restart
- [ ] Xenon pit: reactor cannot restart at t+3 hr after full-power shutdown
- [ ] Doppler feedback self-limits rod ejection within design basis
- [ ] UI updates at ≥10 Hz with no visible lag
- [ ] All alarms fire at correct setpoints
