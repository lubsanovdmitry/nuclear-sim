"""Lumped one-node fuel / one-node coolant thermal model — pure functions, numpy only."""

from physics.constants import (
    H_TRANSFER_A,
    M_FUEL_CP_FUEL,
    M_COOL_CP_COOL,
    M_DOT_NOM_CP_COOL,
    T_COOLANT_INLET,
)


def t_sat_celsius(pressure_pa: float) -> float:
    """Saturation temperature (°C) from RCS pressure — linear fit to steam tables.

    Accurate to ±3°C in the 100–200 bar PWR operating range.
    At 155 bar nominal: ~341°C; actual subcooling at 300°C coolant ≈ 41°C.
    """
    p_bar = pressure_pa / 1.0e5
    return 311.0 + 0.55 * (p_bar - 100.0)


def step_thermal(
    t_fuel: float,
    t_cool: float,
    power: float,
    flow_fraction: float,
    decay_heat: float,
    dt: float,
    t_in: float = T_COOLANT_INLET,
) -> tuple[float, float]:
    """Advance fuel and coolant temperatures by one explicit-Euler timestep.

    Thermal time constants (~5 s fuel, ~2 s coolant) are large relative to
    the expected 0.1 s timestep, so explicit Euler is stable and accurate here.

    Decay heat is placed in the fuel node (physically: fission-product decay
    occurs in the fuel rods before heat transfers to the coolant).

    Args:
        t_fuel: fuel temperature (K)
        t_cool: coolant temperature (K)
        power: fission power (W)
        flow_fraction: coolant flow as fraction of nominal (0–1)
        decay_heat: decay heat power (W)
        t_in: cold-leg inlet temperature (K); defaults to nominal T_COOLANT_INLET.
            Pass the dynamic state value to model secondary heat-sink degradation.
        dt: timestep (s)

    Returns:
        (new_t_fuel, new_t_cool) in Kelvin
    """
    q_transfer = H_TRANSFER_A * (t_fuel - t_cool)
    q_removed = flow_fraction * M_DOT_NOM_CP_COOL * (t_cool - t_in)

    d_t_fuel = (power + decay_heat - q_transfer) / M_FUEL_CP_FUEL
    d_t_cool = (q_transfer - q_removed) / M_COOL_CP_COOL

    return t_fuel + d_t_fuel * dt, t_cool + d_t_cool * dt
