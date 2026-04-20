"""PWR nuclear constants — all values in SI units."""

import numpy as np

# Delayed neutron group parameters (PWR, 6 groups)
# beta_i: delayed neutron fractions
BETA_GROUPS: np.ndarray = np.array([
    0.000215,  # group 1
    0.001424,  # group 2
    0.001274,  # group 3
    0.002568,  # group 4
    0.000748,  # group 5
    0.000273,  # group 6
])

# lambda_i: decay constants (1/s)
LAMBDA_GROUPS: np.ndarray = np.array([
    0.0124,  # group 1
    0.0305,  # group 2
    0.111,   # group 3
    0.301,   # group 4
    1.14,    # group 5
    3.01,    # group 6
])

BETA_TOTAL: float = float(np.sum(BETA_GROUPS))  # 0.0065
PROMPT_NEUTRON_LIFETIME: float = 2.0e-5  # s (Lambda)

# Thermal-hydraulic parameters
T_REF_COOLANT: float = 573.15    # K (300°C nominal coolant temp)
T_REF_FUEL: float = 873.15       # K (600°C nominal fuel centerline temp)
NOMINAL_POWER_W: float = 3.0e9   # W (3000 MWt)

# Reactivity feedback coefficients
ALPHA_DOPPLER: float = -2.5e-5   # dk/k per K (fuel temp)
# Mid-range PWR value (-1e-4 to -4e-4); -4e-4 caused cold-temp feedback to
# overwhelm xenon pit (cold-shutdown ΔT≈170 K → +0.068, masking peak Xe of -0.027).
ALPHA_MODERATOR: float = -1.5e-4 # dk/k per K (coolant temp)

# Control rod reactivity worth: 0% withdrawn = -0.10, 100% withdrawn = +0.143
# Real PWR total rod worth (shutdown margin) is typically 6–15% dk/k.
# -0.10 gives <5% power within 10 s post-SCRAM (physically realistic).
# +0.14276 is chosen so that 75% withdrawal + equilibrium Xe (≈ -0.0207 dk/k)
# + nominal 1000 ppm boron (= -0.10 dk/k) sums to zero:
#   rho_rods(75%) = 0.0207 + 0.10 = 0.1207; solve ROD_WORTH_MAX via S-curve.
ROD_WORTH_MIN: float = -0.10
ROD_WORTH_MAX: float = 0.14276

# Initial dissolved boron at BOC (beginning of cycle); decreases to 0 ppm at EOC.
INITIAL_BORON_PPM: float = 1000.0

# Boron reactivity coefficient
BORON_COEFFICIENT: float = -1.0e-4  # dk/k per ppm

# Xenon / Iodine constants
LAMBDA_IODINE: float = 2.87e-5   # 1/s
LAMBDA_XENON: float = 2.09e-5    # 1/s
SIGMA_XENON: float = 2.6e-22     # m² (2.6 Megabarns; SPEC listed cm² value by mistake)
GAMMA_IODINE: float = 0.061      # fission yield
GAMMA_XENON: float = 0.002       # direct fission yield

# Xenon model flux parameters (derived from core-averaged PWR values)
SIGMA_F_MACRO: float = 30.0       # 1/m  macroscopic fission cross-section (core-averaged)
NU_FISSION: float = 2.4           # neutrons per fission (U-235)
PHI_0: float = 3.0e17             # n/m²/s  reference full-power thermal flux
PHI_SIGMA_F_REF: float = SIGMA_F_MACRO * PHI_0   # fissions/m³/s at n=1
SIGMA_XE_PHI_0: float = SIGMA_XENON * PHI_0      # Xe burnout rate at n=1 (1/s)
NU_SIGMA_F: float = NU_FISSION * SIGMA_F_MACRO   # nu·Σ_f used in reactivity formula (1/m)

# ANS-5.1 11-group decay heat fit for U-235 (infinite irradiation)
# Q_decay(t)/Q0 = sum_k A_k * exp(-alpha_k * t)
# Source: ANS-5.1-1979 standard; sum(A_k) ≈ 0.063 (~6.3% at t=0)
ANS51_A: np.ndarray = np.array([
    6.50e-3, 1.56e-2, 7.60e-3, 7.39e-3, 6.84e-3,
    3.68e-3, 1.60e-3, 3.40e-3, 5.05e-3, 1.59e-3, 3.72e-3,
])
ANS51_ALPHA: np.ndarray = np.array([
    2.23e+00, 5.63e-01, 1.34e-01, 3.50e-02, 1.06e-02,
    3.33e-03, 8.62e-04, 2.43e-04, 5.02e-05, 1.76e-05, 3.53e-06,
])

# Cladding failure temperature
T_CLADDING_FAILURE: float = 1473.15  # K (1200°C)

# Coolant pump parameters
NUM_PUMPS: int = 4
PUMP_COASTDOWN_TAU: float = 30.0  # s — exponential decay time constant

# Diesel generator parameters
DIESEL_START_DELAY_MIN: float = 10.0  # s — minimum time to come online
DIESEL_START_DELAY_MAX: float = 15.0  # s — maximum time to come online

# ECCS parameters
ECCS_HPSI_THRESHOLD: float = 1.0e7    # Pa (100 bar) — HPSI actuates when pressure ≤ 100 bar
ECCS_LPSI_THRESHOLD: float = 2.0e6    # Pa (20 bar)  — LPSI actuates when pressure ≤ 20 bar
ECCS_HPSI_FLOW_FRACTION: float = 0.2  # fraction of nominal coolant flow injected by HPSI
ECCS_LPSI_FLOW_FRACTION: float = 0.5  # fraction of nominal coolant flow injected by LPSI
ECCS_WATER_TEMP: float = 293.15       # K (20°C — cold injection water from RWST)
ECCS_WATER_BORON_PPM: float = 2500.0  # ppm boron in RWST — drives negative reactivity

# Pressurizer parameters
PRESSURE_NOMINAL: float = 1.55e7      # Pa (155 bar — nominal RCS operating pressure)
PRESSURE_SCRAM_HIGH: float = 1.70e7   # Pa (170 bar — high-pressure SCRAM setpoint)
PRESSURE_SCRAM_LOW: float = 1.20e7    # Pa (120 bar — low-pressure SCRAM setpoint)
PRESSURE_TEMP_COEFF: float = 1.0e5    # Pa/K — pressure sensitivity to coolant temperature
PRESSURE_TAU: float = 10.0            # s — pressurizer first-order time constant

# Thermal-hydraulic model parameters
# At steady state: H_TRANSFER_A * (T_REF_FUEL - T_REF_COOLANT) = NOMINAL_POWER_W
#   → 1e7 W/K × 300 K = 3 GW ✓
# τ_fuel = M_FUEL_CP_FUEL / H_TRANSFER_A = 5e7 / 1e7 = 5 s ✓
# τ_cool = M_COOL_CP_COOL / (H_TRANSFER_A + M_DOT_NOM_CP_COOL) = 2.2e8 / 1.1e8 = 2 s ✓
H_TRANSFER_A: float = 1.0e7    # W/K (fuel-to-coolant h×A)
M_FUEL_CP_FUEL: float = 5.0e7  # J/K (fuel thermal mass, τ_fuel ≈ 5 s)
M_COOL_CP_COOL: float = 2.2e8  # J/K (coolant thermal mass, τ_cool ≈ 2 s)
M_DOT_NOM_CP_COOL: float = 1.0e8  # W/K (nominal ṁ·cp_cool; ΔT_cool = 30 K)
T_COOLANT_INLET: float = 543.15   # K (270 °C = T_REF_COOLANT − 30 K)

# Steam generator secondary-side time constant (cold-leg T_in feedback)
# Main feedwater pumps trip with offsite power; T_in then rises toward T_cool
# on this time constant as the steam generators lose secondary-side heat removal.
TAU_STEAM_GENERATOR: float = 60.0  # s

# CVCS boration/dilution rate (Chemical and Volume Control System)
# Real PWRs borate/dilute at 5–30 ppm/min; 10 ppm/min is a conservative mid-range value.
BORON_RATE_PPM_PER_S: float = 10.0 / 60.0  # ppm/s

# Control rod drive mechanism speed (real PWR CRDMs: ~45–72 steps/min ≈ 5–8 %/s).
# 5 %/s prevents abrupt prompt-supercritical insertions while keeping manoeuvring practical.
ROD_SPEED_PCT_PER_S: float = 5.0  # %/s

# Pressurizer liquid level model
# Level tracks RCS inventory: 50% at nominal, drops on LOCA, rises with ECCS makeup.
PRZR_LEVEL_NOMINAL: float = 0.50    # fraction at T_REF_COOLANT (normal operating band)
PRZR_LEVEL_TEMP_COEFF: float = 0.006  # fraction/K — thermal expansion of RCS water
PRZR_LEVEL_TAU: float = 30.0         # s — time constant for thermal-expansion level change
PRZR_DRAIN_COEFF: float = 0.08       # per second at full pressure deficit (large-break LOCA)
PRZR_MAKEUP_COEFF: float = 0.030     # per second per unit ECCS injection flow

# Pressurizer heater/spray operator controls
# Heaters evaporate water into steam → raise pressure; spray condenses steam → lower pressure.
# Rates are ~5× real-plant values for sim interactivity (real: ~1 bar/min heater, ~2 bar/min spray).
PRZR_HEATER_DPDT_MAX: float = 2.0e4  # Pa/s maximum pressure-raise rate at 100% heater demand
PRZR_SPRAY_DPDT_MAX: float = 4.0e4   # Pa/s maximum pressure-drop  rate at 100% spray demand

# Pilot-Operated Relief Valve (PORV) — mounted on pressurizer top
# Opens automatically above 157 bar; recloses at ≤ 155 bar (nominal setpoint).
# A stuck-open PORV (TMI-2 initiating event) causes a slow primary-system bleed.
PORV_OPEN_SETPOINT: float = 1.57e7   # Pa (157 bar) — auto-open threshold
PORV_CLOSE_SETPOINT: float = 1.55e7  # Pa (155 bar) — auto-close threshold
# PORV is modelled as an equivalent flow deficit inside step_pressurizer, the same way
# LOCA is modelled. This shifts p_target down so the first-order lag actually drives
# pressure downward rather than fighting the drain. At 0.75 the target becomes ~117 bar
# (below the 120-bar SCRAM setpoint); with tau=10 s the SCRAM is reached in ~27 s.
PORV_FLOW_DEFICIT: float = 0.75      # equivalent flow-loss fraction when PORV is open
PORV_LEVEL_DRAIN: float = 0.005      # fraction/s pressurizer level drain when open
