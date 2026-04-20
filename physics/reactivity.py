"""Reactivity component model — pure functions, numpy only."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from physics.constants import (
    ALPHA_DOPPLER,
    ALPHA_MODERATOR,
    BORON_COEFFICIENT,
    ROD_WORTH_MIN,
    ROD_WORTH_MAX,
    T_REF_COOLANT,
    T_REF_FUEL,
)
from physics.xenon import xenon_reactivity

if TYPE_CHECKING:
    from api.state import PlantState


def rod_reactivity(positions: list[float] | np.ndarray) -> float:
    """Control rod reactivity from average bank position.

    Args:
        positions: rod bank positions (% withdrawn, 0–100 each)

    Returns:
        rho_rods (dk/k); ROD_WORTH_MIN at 0%, ROD_WORTH_MAX at 100%
    """
    avg = float(np.mean(positions))
    f = avg / 100.0
    # Sinusoidal integral gives S-curve: near-zero worth at top/bottom, peaks at core center
    worth_fraction = f - np.sin(2 * np.pi * f) / (2 * np.pi)
    return ROD_WORTH_MIN + worth_fraction * (ROD_WORTH_MAX - ROD_WORTH_MIN)


def doppler_reactivity(t_fuel: float) -> float:
    """Fuel-temperature (Doppler) reactivity feedback.

    Args:
        t_fuel: fuel temperature (K)

    Returns:
        rho_doppler (dk/k)
    """
    return ALPHA_DOPPLER * (t_fuel - T_REF_FUEL)


def moderator_reactivity(t_cool: float) -> float:
    """Coolant/moderator temperature reactivity feedback.

    Args:
        t_cool: coolant temperature (K)

    Returns:
        rho_moderator (dk/k)
    """
    return ALPHA_MODERATOR * (t_cool - T_REF_COOLANT)


def boron_reactivity(boron_ppm: float) -> float:
    """Dissolved boron reactivity.

    Args:
        boron_ppm: boron concentration (ppm)

    Returns:
        rho_boron (dk/k, ≤ 0)
    """
    return BORON_COEFFICIENT * boron_ppm


def compute_reactivity(state: PlantState) -> float:
    """Total reactivity as sum of all feedback components.

    Args:
        state: current plant state

    Returns:
        rho_total = rho_rods + rho_doppler + rho_moderator + rho_xenon + rho_boron
    """
    rho_rods = rod_reactivity(state.rod_positions)
    rho_doppler = doppler_reactivity(state.t_fuel)
    rho_moderator = moderator_reactivity(state.t_cool)
    rho_xenon = xenon_reactivity(state.xenon)
    rho_boron = boron_reactivity(state.boron_ppm)
    return rho_rods + rho_doppler + rho_moderator + rho_xenon + rho_boron
