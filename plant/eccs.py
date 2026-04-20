"""Emergency Core Cooling System (ECCS).

HPSI: injects at high pressure (>= 100 bar), e.g. early LOCA phase.
LPSI: injects at low pressure (<= 20 bar), e.g. after full depressurization.
Both inject cold water — provides negative reactivity via coolant cooling and
direct core cooling.
"""
from __future__ import annotations

from dataclasses import dataclass

from physics.constants import (
    BORON_COEFFICIENT,
    ECCS_HPSI_FLOW_FRACTION,
    ECCS_HPSI_THRESHOLD,
    ECCS_LPSI_FLOW_FRACTION,
    ECCS_LPSI_THRESHOLD,
    ECCS_WATER_BORON_PPM,
    ECCS_WATER_TEMP,
)


@dataclass
class ECCSState:
    hpsi_active: bool = False
    lpsi_active: bool = False
    injection_flow_fraction: float = 0.0  # additional flow as fraction of nominal
    injection_temp: float = ECCS_WATER_TEMP  # K


def step_eccs(
    pressure: float,
    armed: bool,
    t_cool: float,
    flow_fraction: float,
) -> ECCSState:
    """Determine ECCS injection state based on RCS pressure.

    ECCS is automatic: HPSI activates when pressure >= 100 bar; LPSI at <= 20 bar.
    The ``armed`` flag acts as an operator **disable** switch — set to False to inhibit.
    Default in PlantState is True (ECCS active / not disabled).

    Args:
        pressure: RCS pressure (Pa)
        armed: False to disable ECCS (operator inhibit); True = automatic (default)
        t_cool: current coolant temperature (K)
        flow_fraction: current coolant flow fraction (0–1)

    Returns:
        ECCSState with active flags and injection parameters.
    """
    if not armed:
        return ECCSState()

    # HPSI: injects when RCS pressure has dropped to ≤ 100 bar (LOCA signal)
    # LPSI: injects when pressure has further fallen to ≤ 20 bar
    hpsi = pressure <= ECCS_HPSI_THRESHOLD
    lpsi = pressure <= ECCS_LPSI_THRESHOLD

    inj_flow = 0.0
    if hpsi:
        inj_flow += ECCS_HPSI_FLOW_FRACTION
    if lpsi:
        inj_flow += ECCS_LPSI_FLOW_FRACTION

    return ECCSState(
        hpsi_active=hpsi,
        lpsi_active=lpsi,
        injection_flow_fraction=inj_flow,
        injection_temp=ECCS_WATER_TEMP,
    )


def eccs_reactivity(eccs_state: ECCSState, t_cool: float) -> float:
    """Compute reactivity from ECCS borated cold-water injection.

    ECCS water contains ~2500 ppm boron (from the RWST).  Injecting it into the
    RCS raises the effective core boron concentration, inserting negative
    reactivity.  The boron effect dominates the opposing moderator temperature
    coefficient and keeps the reactor subcritical.

    Args:
        eccs_state: current ECCS injection state
        t_cool: current bulk coolant temperature (K) — unused; kept for API consistency

    Returns:
        rho_eccs (dk/k) — zero or negative.
    """
    if eccs_state.injection_flow_fraction == 0.0:
        return 0.0

    # Effective boron ppm added to the core from mixing injected water
    total_flow = 1.0 + eccs_state.injection_flow_fraction
    effective_boron_ppm = ECCS_WATER_BORON_PPM * eccs_state.injection_flow_fraction / total_flow
    return BORON_COEFFICIENT * effective_boron_ppm
