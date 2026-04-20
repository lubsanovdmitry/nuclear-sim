"""Station Blackout (SBO) accident trigger."""

from __future__ import annotations

from plant.diesels import DieselState
from api.state import PlantState


def trigger_sbo(state: PlantState) -> PlantState:
    """Initiate SBO: LOOP + both diesel generators fail.

    Like LOOP but no AC recovery is possible.  Only battery-backed systems remain.
    Natural circulation and passive accumulators are the only cooling paths.
    """
    state.offsite_power = False
    # Diesels fail immediately — override the state machine
    state.diesel_states = [
        DieselState(state="failed"),
        DieselState(state="failed"),
    ]
    state.diesel_start_signals = [False, False]
    if "SBO" not in state.alarms:
        state.alarms = list(state.alarms) + ["SBO"]
    return state
