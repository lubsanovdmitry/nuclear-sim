"""Control Rod Ejection accident trigger."""

from __future__ import annotations

from api.state import PlantState

# Bank index that is mechanically ejected (bank A = index 0)
_EJECTED_BANK: int = 0


def trigger_rod_ejection(state: PlantState) -> PlantState:
    """Initiate control rod ejection: bank A is fully withdrawn in <0.1 s.

    Sequence per SPEC:
      t=0  Mechanical failure ejects rod bank A to 100% withdrawn instantly.
           This inserts ~+0.01 dk/k positive reactivity (from typical partial-insertion).
      Doppler feedback self-limits the power excursion if within design basis.
      High-power SCRAM signal fires from the simulation loop.
    """
    rod_positions = list(state.rod_positions)
    rod_positions[_EJECTED_BANK] = 100.0
    state.rod_positions = rod_positions
    if "ROD_EJECTION" not in state.alarms:
        state.alarms = list(state.alarms) + ["ROD_EJECTION"]
    return state
