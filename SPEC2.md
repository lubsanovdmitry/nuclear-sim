# SPEC Addendum — Phase 2: Axial Nodalization & Two-Phase Thermal Hydraulics

All Phase 1 interfaces remain valid.
Phase 2 extends them; nothing is replaced.

***

## Design Decisions

- **N = 10 axial nodes** — fixed for Phase 2. Indexed 0 (bottom) to 9 (top).
- **Reactor type context:** Two-phase is standard in BWR; accident condition in PWR.
  The model handles both — void coefficient magnitude differs by config.
- **Backward compatibility:** All scalar thermal interfaces get a wrapper so
  Phase 1 tests continue to pass unchanged.

***

## 1. Axial Nodalization (`physics/axial.py`)

### Power Shape Functions

```python
def cosine_power_shape(N: int = 10) -> np.ndarray:
    """Chopped cosine axial power distribution. Normalized so sum = N (avg = 1.0)."""

def flat_power_shape(N: int = 10) -> np.ndarray:
    """Uniform axial power. Normalized so sum = N."""

def chopped_cosine(N: int, extrapolation_length: float = 0.1) -> np.ndarray:
    """More realistic: cosine with extrapolation beyond core boundaries.
    extrapolation_length as fraction of core height (typical 0.1)."""
```

### Axial Temperature Profile

Coolant temperature rises from bottom to top (flow direction in PWR; also BWR).

```python
def axial_coolant_temp(
    t_in: float,               # K, coolant inlet temperature
    power_shape: np.ndarray,   # shape (N,), normalized (avg=1.0)
    total_power: float,        # W, total core power
    flow_fraction: float,      # 0.0–1.0
    N: int = 10
) -> np.ndarray:               # K, shape (N,)
    """Compute coolant temp at each node using enthalpy rise method.
    T_n = T_{n-1} + (power_shape[n] * total_power / N) / (m_dot * cp)
    m_dot = flow_fraction * NOMINAL_FLOW_RATE
    cp = 5500 J/kgK (water at PWR conditions)
    NOMINAL_FLOW_RATE = 17000 kg/s (3000 MWt PWR)"""

def axial_fuel_temp(
    t_cool: np.ndarray,        # K, shape (N,)
    power_shape: np.ndarray,   # shape (N,)
    total_power: float,        # W
    N: int = 10
) -> np.ndarray:               # K, shape (N,)
    """T_fuel[n] = T_cool[n] + power_shape[n] * total_power / (N * h * A_fuel)
    h = 50000 W/m²K (fuel-to-coolant HTC, nominal)
    A_fuel = 0.06 m² (effective heat transfer area per node)"""
```

***

## 2. Two-Phase Thermal Hydraulics (`physics/two_phase.py`)

### Saturation Properties

```python
def saturation_temp(pressure: float) -> float:
    """T_sat in K as function of pressure in Pa.
    Use Antoine approximation valid 1–200 bar:
    T_sat = 373.15 * (pressure / 101325) ** 0.25  [rough but sufficient]
    More accurate: use tabulated steam tables via numpy interp."""

def saturation_properties(pressure: float) -> dict:
    """Returns dict with:
    h_f   — saturated liquid enthalpy [J/kg]
    h_fg  — latent heat of vaporization [J/kg]
    rho_f — saturated liquid density [kg/m³]
    rho_g — saturated vapor density [kg/m³]
    Use piecewise linear interpolation from table at 1,10,50,75,100,155 bar."""

STEAM_TABLE = {
    # pressure (bar): (T_sat K, h_f J/kg, h_fg J/kg, rho_f kg/m³, rho_g kg/m³)
    1:   (373.15, 417e3, 2257e3, 958.0, 0.60),
    10:  (453.03, 763e3, 2015e3, 887.0, 5.16),
    50:  (537.09, 1155e3, 1641e3, 777.0, 25.4),
    75:  (560.22, 1292e3, 1494e3, 732.0, 39.5),
    100: (584.15, 1408e3, 1317e3, 688.0, 55.5),
    155: (618.15, 1630e3, 1000e3, 594.0, 101.0),
}
```

### Void Fraction (Drift-Flux Model)

```python
def thermodynamic_quality(
    t_cool: float,    # K, local coolant temperature
    pressure: float   # Pa
) -> float:
    """x = (h - h_f) / h_fg
    For subcooled: x < 0 (but actual void can still exist — subcooled boiling)
    For saturated: 0 <= x <= 1
    For superheated: x > 1 (treat as x=1 in this model)"""

def void_fraction(quality: float, pressure: float) -> float:
    """Drift-flux model (Zuber-Findlay):
    alpha = x / (C0 * (x + (1-x)*rho_g/rho_f) + rho_g*V_gj/(G*x))
    Simplified for this model:
    alpha = x / (x + (1-x) * rho_g/rho_f)   [homogeneous equilibrium]
    Clamp to [0.0, 1.0]. Returns 0 if x <= 0."""

def void_fraction_subcooled(
    t_cool: float,    # K
    t_sat: float,     # K
    heat_flux: float, # W/m²
    pressure: float   # Pa
) -> float:
    """Subcooled void fraction (Saha-Zuber model).
    Even when T_cool < T_sat, local boiling can occur near fuel surface.
    Returns small positive alpha even for subcooled bulk coolant at high heat flux."""
```

### Heat Transfer Regimes

```python
def boiling_regime(alpha: float, heat_flux: float, CHF: float) -> str:
    """Returns one of:
    'single_phase'      — alpha = 0, T_cool < T_sat
    'subcooled_boiling' — small alpha, T_cool < T_sat but high heat flux
    'nucleate_boiling'  — 0 < alpha < 0.7, heat_flux < CHF
    'film_boiling'      — alpha > 0.7 OR heat_flux > CHF (DANGEROUS)
    """

def heat_transfer_coefficient(
    alpha: float,
    flow_fraction: float,
    pressure: float,
    regime: str
) -> float:
    """Returns h in W/m²K by regime:
    single_phase:      h = 30000 * flow_fraction          (Dittus-Boelter)
    subcooled_boiling: h = 50000 * flow_fraction
    nucleate_boiling:  h = 80000 * (1 - alpha)
    film_boiling:      h = 3000   ← dramatic drop — fuel heats rapidly
    """
```

### Critical Heat Flux and DNBR

```python
def critical_heat_flux(
    flow_fraction: float,
    pressure: float,      # Pa
    quality: float
) -> float:
    """CHF in W/m².
    Use simplified Bowring-style correlation:
    CHF = CHF_0 * F_flow * F_pressure * F_quality

    CHF_0 = 1.5e6  W/m² (nominal PWR CHF)

    F_flow     = max(0.1, flow_fraction)         (low flow → low CHF)
    F_pressure = 1.0 - abs(pressure - 155e5) / 155e5  (peaks at 155 bar)
    F_quality  = max(0.2, 1.0 - 2.0 * max(0, quality))  (high quality → low CHF)

    Clamp CHF to minimum 0.2e6 W/m²."""

def dnbr(
    actual_heat_flux: float,   # W/m², q''_actual at hottest node
    CHF: float                 # W/m²
) -> float:
    """Departure from Nucleate Boiling Ratio.
    DNBR = CHF / q''_actual
    Safety limit: DNBR >= 1.3 (below this: fuel damage possible)
    Fuel damage:  DNBR < 1.0"""

def actual_heat_flux(power_shape: np.ndarray, total_power: float) -> np.ndarray:
    """q''[n] = power_shape[n] * total_power / (N * A_fuel)
    A_fuel = 0.06 m² per node."""
```

***

## 3. Modifications to Existing Modules

### `physics/thermal.py` — Extended to Arrays

```python
# NEW primary interface (array-based):
def step_thermal_nodal(
    t_fuel: np.ndarray,        # K, shape (N,)
    t_cool: np.ndarray,        # K, shape (N,)
    void_fraction: np.ndarray, # shape (N,)
    power_shape: np.ndarray,   # shape (N,), normalized
    total_power: float,        # W
    flow_fraction: float,
    decay_heat: float,         # W
    pressure: float,           # Pa
    dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (new_t_fuel, new_t_cool) arrays shape (N,).
    Each node solved independently with local h from two_phase.py."""

# BACKWARD COMPAT wrapper (keeps Phase 1 tests passing):
def step_thermal(t_fuel, t_cool, power, flow_fraction, decay_heat, dt):
    """Scalar wrapper around step_thermal_nodal using N=1, flat shape."""
```

### `physics/reactivity.py` — Void Feedback

```python
# ADD to compute_reactivity():
def void_reactivity(
    void_fraction: np.ndarray,  # shape (N,)
    power_shape: np.ndarray,    # shape (N,), for power-weighted average
    reactor_type: str           # 'PWR' or 'BWR'
) -> float:
    """Power-weighted average void fraction:
    alpha_avg = sum(power_shape * void_fraction) / sum(power_shape)

    Void coefficient:
    PWR: alpha_void = -0.02   (small, negative — subcooled boiling rare)
    BWR: alpha_void = -0.15   (large, negative — primary feedback mechanism)

    rho_void = alpha_void * alpha_avg"""

# MODIFY moderator_reactivity() to accept either scalar T_cool or array:
# If array given, use power-weighted average for feedback calculation.
```

***

## 4. Updated `PlantState` (additions to api/state.py)

```python
# Add to PlantState dataclass:

# Axial arrays (shape N=10)
t_fuel_axial: np.ndarray        # K, shape (10,) — default cosine profile
t_cool_axial: np.ndarray        # K, shape (10,) — default linear rise
void_fraction: np.ndarray       # shape (10,) — default zeros
quality: np.ndarray             # shape (10,) — default zeros
heat_flux: np.ndarray           # W/m², shape (10,)
boiling_regime: list[str]       # shape (10,) — regime per node
axial_power_shape: np.ndarray   # shape (10,) — current normalized shape

# Scalar derived values
dnbr: float                     # default 2.5 (safe)
chf: float                      # W/m², current CHF
peak_heat_flux_node: int        # index 0–9 of hottest node
fuel_damage: bool               # True if any node DNBR < 1.0
film_boiling_nodes: list[int]   # indices of nodes in film boiling

# Scalar — exists in Phase 1 but now has more meaning
pressure: float                 # Pa — drives saturation temp
```

***

## 5. New Alarms (add to api/alarms.py)

```python
("LOW_DNBR",       lambda s: s.dnbr < 1.3,                          "RED"),
("APPROACH_CHF",   lambda s: 1.3 <= s.dnbr < 2.0,                   "YELLOW"),
("FILM_BOILING",   lambda s: len(s.film_boiling_nodes) > 0,          "RED"),
("FUEL_DAMAGE",    lambda s: s.fuel_damage,                          "RED"),
("VOID_HIGH",      lambda s: s.void_fraction.mean() > 0.5,          "RED"),
("VOID_MODERATE",  lambda s: 0.3 < s.void_fraction.mean() <= 0.5,   "YELLOW"),
("AXIAL_TILT",     lambda s: s.axial_power_shape.max() /
                              s.axial_power_shape.mean() > 1.8,      "YELLOW"),
```

***

## 6. Frontend Additions (frontend/panel.html)

### Axial Core Visualization

A vertical bar chart showing 10 nodes bottom-to-top. Three overlapping data series:

- **Power shape** (blue bars, background) — normalized 0–2.0
- **Void fraction** (colored fill, foreground) — 0–1.0
  - 0.0–0.1: transparent (subcooled)
  - 0.1–0.4: light blue (subcooled boiling)
  - 0.4–0.7: white/steam (saturated boiling)
  - 0.7–1.0: orange/red (approaching film boiling)
- **Film boiling nodes** — node outline turns red

Update at same rate as other gauges (100ms).

### DNBR Gauge

Circular gauge 0–4.0 range:

- Green zone: > 2.0
- Yellow zone: 1.3–2.0
- Red zone: < 1.3
- Needle animates continuously
- Digital readout below: "DNBR: 2.34"

### Axial Temperature Profile (optional but useful)

Horizontal bar chart showing T_fuel and T_cool per node as dual series.
Toggle button to switch between void visualization and temperature visualization.

***

## 7. LOCA Scenario — Phase 2 Version (replace accidents/loca.py)

Sequence with two-phase physics:

```
t=0s     Large break opens. Leak rate = f(pressure, break_size).
t=0–10s  Rapid depressurization: pressure 155 → ~10 bar over 10s.
         T_sat drops from 618K → 453K.
         All nodes where T_cool > new T_sat → flash to steam instantly.
         void_fraction spikes across all nodes simultaneously.
         Void reactivity spike (negative) → power drops before SCRAM.
t=2s     Low pressure SCRAM signal fires (P < 120 bar).
t=3s     SCRAM: rods insert, power → decay heat only.
t=5s     HPSI activates (if ECCS armed). Flow of cold water into core.
t=5–30s  Core uncovery phase: flow_fraction drops as coolant inventory lost.
         CHF drops with flow. DNBR collapses.
         If ECCS flow sufficient: DNBR recovers, no fuel damage.
         If ECCS insufficient or delayed: film_boiling_nodes fills up.
         fuel_damage = True if any node DNBR < 1.0 for > 5s.
t=30s+   Reflood: LPSI activates when pressure < 20 bar.
         Cold water refills vessel bottom→top.
         void_fraction clears node by node (node 0 first).
         T_fuel drops as each node is reflood.
```

Parameters:

- `break_size`: float 0.0–1.0 (fraction of main coolant pipe area)
- `eccs_delay`: float seconds (0 = instant, 30 = late activation)
- Small break (< 0.1): slow depressurization, HPSI handles it easily
- Large break (> 0.5): rapid, DNBR collapses within 10s without ECCS

***

## 8. Phase 2 Acceptance Criteria

- [ ] At nominal PWR conditions: all nodes subcooled, void_fraction ≈ 0, DNBR > 2.5
- [ ] At 115% power: peak node approaches subcooled boiling, DNBR approaches 2.0
- [ ] At 130% power: nucleate boiling in top 3 nodes, DNBR < 1.3 alarm fires
- [ ] LOCA (large break): void spike visible in axial visualization within 5s
- [ ] LOCA with ECCS: fuel_damage = False if ECCS activates within 20s
- [ ] LOCA without ECCS: fuel_damage = True within 45s
- [ ] Axial power shape cosine visible in visualization at all times
- [ ] DNBR gauge responds continuously to power/flow changes
- [ ] Phase 1 pytest suite passes unchanged

***

## 9. Implementation Order

```
P2-Session 1:  physics/axial.py + refactor physics/thermal.py
P2-Session 2:  physics/two_phase.py (full module)
P2-Session 3:  physics/reactivity.py void feedback + api/state.py additions
P2-Session 4:  api/alarms.py new alarms + api/simulation_loop.py nodal integration
P2-Session 5:  accidents/loca.py Phase 2 rewrite
P2-Session 6:  frontend axial visualization + DNBR gauge
P2-Session 7:  Integration + Phase 2 acceptance criteria verification
```
