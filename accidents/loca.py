"""Loss of Coolant Accident (LOCA) trigger."""

from __future__ import annotations

from physics.constants import ECCS_HPSI_THRESHOLD
from api.state import PlantState


# Drop to 110 bar — below SCRAM_LOW (120 bar) so SCRAM fires, above HPSI threshold
# (100 bar) so HPSI injection is immediately active.
_LOCA_INITIAL_PRESSURE: float = 1.10e7  # Pa (110 bar)


def trigger_loca(state: PlantState) -> PlantState:
    """Initiate LOCA: large primary break causes rapid depressurization.

    Sequence per SPEC:
      t=0  RCS pressure drops — set to 110 bar (below SCRAM_LOW, above HPSI threshold).
      t=0  ECCS armed; HPSI activates automatically at >= 100 bar.
      Simulation loop triggers SCRAM on low-pressure signal and drives depressurization.
    """
    state.pressure = _LOCA_INITIAL_PRESSURE
    state.eccs_armed = True
    if "LOCA" not in state.alarms:
        state.alarms = list(state.alarms) + ["LOCA"]
    return state
