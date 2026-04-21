"""Phase 2 alarm definitions — list of (name, predicate, severity) tuples.

Evaluated each tick in simulation_loop.py after Phase 2 derived quantities are computed.
Phase 1 alarms are defined inline in simulation_loop.py; these extend them.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.state import PlantState

PHASE2_ALARMS: list[tuple[str, object, str]] = [
    ("LOW_DNBR",      lambda s: s.dnbr < 2.0,                                          "RED"),
    ("APPROACH_CHF",  lambda s: 2.0 <= s.dnbr < 2.5,                                  "YELLOW"),
    ("FILM_BOILING",  lambda s: len(s.film_boiling_nodes) > 0,                         "RED"),
    ("FUEL_DAMAGE",   lambda s: s.fuel_damage,                                          "RED"),
    ("VOID_HIGH",     lambda s: float(s.void_fraction.mean()) > 0.5,                   "RED"),
    ("VOID_MODERATE", lambda s: 0.3 < float(s.void_fraction.mean()) <= 0.5,            "YELLOW"),
    ("AXIAL_TILT",    lambda s: float(s.axial_power_shape.max() /
                                      s.axial_power_shape.mean()) > 1.8,               "YELLOW"),
]
