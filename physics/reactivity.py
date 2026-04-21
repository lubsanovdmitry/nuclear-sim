"""Reactivity component model — pure functions, numpy only."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from physics.constants import (
    ALPHA_DOPPLER,
    ALPHA_MODERATOR,
    ALPHA_VOID_BWR,
    ALPHA_VOID_PWR,
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


def void_reactivity(
    void_fraction_arr: np.ndarray,
    power_shape: np.ndarray,
    reactor_type: str = 'PWR',
) -> float:
    """Power-weighted void reactivity feedback.

    Args:
        void_fraction_arr: void fraction per axial node, shape (N,)
        power_shape: axial power shape (mean=1.0), shape (N,)
        reactor_type: 'PWR' or 'BWR'

    Returns:
        rho_void (dk/k); ≈ 0 at PWR nominal, ≈ -0.06 at BWR 40% void
    """
    alpha_avg = float(np.average(void_fraction_arr, weights=power_shape))
    alpha_void_coeff = ALPHA_VOID_BWR if reactor_type == 'BWR' else ALPHA_VOID_PWR
    return alpha_void_coeff * alpha_avg


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
        rho_total = rho_rods + rho_doppler + rho_moderator + rho_xenon + rho_boron + rho_void
    """
    rho_rods = rod_reactivity(state.rod_positions)
    rho_doppler = doppler_reactivity(state.t_fuel)
    t_cool_eff = float(np.average(state.t_cool_axial, weights=state.axial_power_shape))
    rho_moderator = moderator_reactivity(t_cool_eff)
    rho_xenon = xenon_reactivity(state.xenon)
    rho_boron = boron_reactivity(state.boron_ppm)
    rho_void = void_reactivity(state.void_fraction, state.axial_power_shape)
    return rho_rods + rho_doppler + rho_moderator + rho_xenon + rho_boron + rho_void
