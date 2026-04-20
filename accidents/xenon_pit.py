"""Xenon Pit scenario trigger."""

from __future__ import annotations

from physics.xenon import rk4_step_xenon, xenon_equilibrium
from api.state import PlantState

# Simulate 3 hours of shutdown in 60-second steps
_SHUTDOWN_DURATION_S: float = 3.0 * 3600.0   # 10 800 s
_DT_S: float = 60.0
_N_STEPS: int = int(_SHUTDOWN_DURATION_S / _DT_S)


def trigger_xenon_pit(state: PlantState) -> PlantState:
    """Simulate shutdown from 100% power, then fast-forward 3 hours.

    Sequence per SPEC:
      t=0     Reactor was running at 100% power.  SCRAM inserted; rods fully in.
      t=0..3h Xe builds up as I-135 decays; no neutron flux to burn Xe.
      t=3h    (this state) Xe near its post-shutdown peak.
              Restart is impossible — Xe worth exceeds available rod worth.

    The Xe/I concentrations are computed analytically via RK4 at n=0 for 3 hours,
    starting from full-power equilibrium.
    """
    # 1. Equilibrium Xe/I at 100% power
    I0, Xe0 = xenon_equilibrium(1.0)

    # 2. Advance 3 hours at zero power (no flux to burn Xe)
    I, Xe = I0, Xe0
    for _ in range(_N_STEPS):
        I, Xe = rk4_step_xenon(I, Xe, n=0.0, dt=_DT_S)

    # 3. Apply to state
    state.iodine = I
    state.xenon = Xe

    # 4. Reactor is shut down: SCRAM, rods fully inserted, decay heat accumulating.
    #    Emergency boration raises concentration to 1500 ppm (real CVCS procedure)
    #    to maintain cold-shutdown margin as moderator temp feedback turns positive.
    state.scram = True
    state.n = 0.0
    state.rod_positions = [0.0, 0.0, 0.0, 0.0]
    state.t_since_scram = _SHUTDOWN_DURATION_S
    state.boron_ppm = 1500.0

    if "XENON_PIT" not in state.alarms:
        state.alarms = list(state.alarms) + ["XENON_PIT"]
    return state
