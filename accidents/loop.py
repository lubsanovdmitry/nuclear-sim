"""Loss of Offsite Power (LOOP) accident trigger."""

from __future__ import annotations

from api.state import PlantState


def trigger_loop(state: PlantState) -> PlantState:
    """Initiate LOOP: grid disconnects, all pumps begin coastdown, diesels auto-start.

    Sequence per SPEC:
      t=0  Grid fails — offsite_power set False; all pumps lose AC, begin coastdown.
      t=0  Diesel start signals sent automatically.
      The simulation loop handles pump coastdown and diesel state machine from here.
    """
    state.offsite_power = False
    state.diesel_start_signals = [True, True]
    if "LOOP" not in state.alarms:
        state.alarms = list(state.alarms) + ["LOOP"]
    return state
