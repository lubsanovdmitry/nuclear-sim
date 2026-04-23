"""Plant state dataclass — single source of truth for all simulation modules."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from physics.axial import cosine_power_shape


@dataclass
class PlantState:
    # Simulation time (s)
    t: float = 0.0

    # Neutronics
    n: float = 1.0                     # normalized neutron population (power)
    precursors: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )

    # Temperatures (K, SI)
    t_fuel: float = 873.15             # K (600 °C nominal)
    t_cool: float = 573.15             # K (300 °C nominal)
    t_in: float = 543.15               # K (270 °C nominal cold-leg; rises on heat-sink loss)

    # Control rods: % withdrawn per bank (0 = fully inserted, 100 = fully withdrawn)
    # rod_target_positions = operator-commanded setpoint (CRDM drives toward this)
    # rod_positions        = actual physical position (rate-limited by ROD_SPEED_PCT_PER_S)
    rod_positions: list[float] = field(
        default_factory=lambda: [100.0, 100.0, 100.0, 100.0]
    )
    rod_target_positions: list[float] = field(
        default_factory=lambda: [100.0, 100.0, 100.0, 100.0]
    )

    # Xenon / Iodine kinetics (atoms/m³)
    iodine: float = 0.0
    xenon: float = 0.0

    # Dissolved boron — boron_ppm is the actual concentration; boron_target_ppm is
    # the CVCS setpoint the system is slowly approaching at BORON_RATE_PPM_PER_S.
    boron_ppm: float = 0.0
    boron_target_ppm: float = 0.0

    # Reactivity component breakdown (dk/k) — updated each tick for display
    rho_rod: float = 0.0
    rho_doppler: float = 0.0
    rho_moderator: float = 0.0
    rho_boron: float = 0.0
    rho_xenon: float = 0.0
    rho_eccs: float = 0.0

    # Plant systems — operator commands
    pumps: list[bool] = field(
        default_factory=lambda: [True, True, True, True]
    )
    pump_speeds: np.ndarray = field(
        default_factory=lambda: np.ones(4)   # normalised speed per pump [0, 1]
    )
    flow_fraction: float = 1.0         # fraction of nominal coolant flow (0–1)
    pressure: float = 1.55e7           # Pa (155 bar nominal)
    pressurizer_level: float = 0.50    # fraction 0–1 (50% at nominal operating conditions)
    przr_heater_fraction: float = 0.0  # operator heater demand 0–1 (raises pressure)
    przr_spray_fraction: float = 0.0   # operator spray demand 0–1 (lowers pressure)
    porv_open: bool = False            # PORV position (auto or manually forced open)
    porv_stuck_open: bool = False      # True when PORV fails to reclose (TMI-style fault)

    # Offsite power and diesel generators
    offsite_power: bool = True
    diesel_states: list = field(       # list[DieselState], 2 units
        default_factory=list
    )
    diesel_start_signals: list[bool] = field(
        default_factory=lambda: [False, False]
    )

    # Emergency Core Cooling System
    # eccs_armed=True  → automatic actuation on pressure signals (default, realistic)
    # eccs_armed=False → operator-inhibited (used in training/settings panel)
    eccs_armed: bool = True
    eccs_state: object = field(        # ECCSState; typed as object to avoid import cycle
        default_factory=lambda: None
    )

    # Alarms
    alarms: list[str] = field(
        default_factory=list
    )

    # Permanent positive reactivity from ejected rod; zeroed when SCRAM fires
    ejection_rho: float = 0.0

    # Phase 2 — axial nodalization (all shape (10,)); defaults keep Phase 1 unaffected
    t_fuel_axial: np.ndarray = field(default_factory=lambda: np.full(10, 873.15))
    t_cool_axial: np.ndarray = field(default_factory=lambda: np.linspace(543.15, 598.15, 10))
    void_fraction: np.ndarray = field(default_factory=lambda: np.zeros(10))
    void_fraction_dyn: np.ndarray = field(default_factory=lambda: np.zeros(10))
    quality: np.ndarray = field(default_factory=lambda: np.zeros(10))
    heat_flux: np.ndarray = field(default_factory=lambda: np.zeros(10))
    htc: np.ndarray = field(default_factory=lambda: np.full(10, 30000.0))
    boiling_regime: list = field(default_factory=lambda: ['single_phase'] * 10)
    axial_power_shape: np.ndarray = field(default_factory=lambda: cosine_power_shape(10))

    # Phase 2 — derived safety parameters
    dnbr: float = 2.5
    chf: float = 1.5e6
    peak_heat_flux_node: int = 5
    fuel_damage: bool = False
    film_boiling_nodes: list = field(default_factory=list)

    # Phase 2 — LOCA tracking
    loca_active: bool = False
    loca_break_size: float = 0.0
    dnbr_low_timer: float = 0.0
    loca_flow_fraction: float = 1.0   # max flow allowed by LOCA coolant loss (1.0 = no restriction)
    lpsi_timer: float = 0.0           # seconds of sustained LPSI injection (drives reflood)

    # Control flags
    scram: bool = False
    scram_cause: str = ""              # initiating signal, e.g. "HI_POWER", "LO_PRESSURE", "LO_FLOW"
    t_since_scram: float = 0.0        # s elapsed since SCRAM (for decay heat)
    # Channels present here are bypassed (server will not auto-fire that SCRAM).
    # Valid names: "HI_POWER", "HI_PRESSURE", "LO_PRESSURE", "LO_FLOW",
    #              "HI_FUEL_TEMP", "HI_COOL_TEMP"
    scram_bypasses: list[str] = field(default_factory=list)


def default_state() -> PlantState:
    """Return a hot-standby PlantState: 100% power, all pumps on, offsite power available,
    no alarms, core at full-power equilibrium (xenon at steady-state level)."""
    from physics.pke_solver import steady_state_initial_conditions
    from physics.xenon import xenon_equilibrium
    from physics.constants import (
        T_REF_FUEL, T_REF_COOLANT, T_COOLANT_INLET,
        PRESSURE_NOMINAL, INITIAL_BORON_PPM, PRZR_LEVEL_NOMINAL,
    )
    from plant.diesels import DieselState
    from plant.eccs import ECCSState

    pke0 = steady_state_initial_conditions(1.0)
    I_eq, Xe_eq = xenon_equilibrium(1.0)

    return PlantState(
        t=0.0,
        n=1.0,
        precursors=pke0[1:].copy(),
        t_fuel=T_REF_FUEL,
        t_cool=T_REF_COOLANT,
        t_in=T_COOLANT_INLET,
        rod_positions=[75.0, 75.0, 75.0, 75.0],
        rod_target_positions=[75.0, 75.0, 75.0, 75.0],
        iodine=I_eq,
        xenon=Xe_eq,
        boron_ppm=INITIAL_BORON_PPM,
        boron_target_ppm=INITIAL_BORON_PPM,
        pumps=[True, True, True, True],
        pump_speeds=np.ones(4),
        flow_fraction=1.0,
        pressure=PRESSURE_NOMINAL,
        pressurizer_level=PRZR_LEVEL_NOMINAL,
        offsite_power=True,
        diesel_states=[DieselState(), DieselState()],
        diesel_start_signals=[False, False],
        eccs_armed=True,
        eccs_state=ECCSState(),
        alarms=[],
        scram=False,
        scram_cause="",
        t_since_scram=0.0,
    )
