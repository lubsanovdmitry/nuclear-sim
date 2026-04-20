"""Xe-135 / I-135 kinetics — pure functions, numpy only."""

from physics.constants import (
    GAMMA_IODINE, GAMMA_XENON,
    LAMBDA_IODINE, LAMBDA_XENON,
    SIGMA_XENON,
    PHI_SIGMA_F_REF, SIGMA_XE_PHI_0, NU_SIGMA_F,
)


def xenon_equilibrium(n: float) -> tuple[float, float]:
    """Equilibrium I-135 and Xe-135 concentrations at normalized power n.

    Args:
        n: normalized power level (1.0 = full power, 0.0 = shutdown)

    Returns:
        (I_eq, Xe_eq) in atoms/m³
    """
    phi_sigma_f = PHI_SIGMA_F_REF * n
    sigma_xe_phi = SIGMA_XE_PHI_0 * n

    I_eq = GAMMA_IODINE * phi_sigma_f / LAMBDA_IODINE
    Xe_eq = (GAMMA_XENON * phi_sigma_f + LAMBDA_IODINE * I_eq) / (LAMBDA_XENON + sigma_xe_phi)

    return I_eq, Xe_eq


def xenon_derivatives(I: float, Xe: float, n: float) -> tuple[float, float]:
    """Time derivatives of I-135 and Xe-135 concentrations.

    Args:
        I: iodine-135 concentration (atoms/m³)
        Xe: xenon-135 concentration (atoms/m³)
        n: normalized power level (0–1 at full power)

    Returns:
        (dI/dt, dXe/dt) in atoms/m³/s
    """
    phi_sigma_f = PHI_SIGMA_F_REF * n
    sigma_xe_phi = SIGMA_XE_PHI_0 * n

    dI_dt = GAMMA_IODINE * phi_sigma_f - LAMBDA_IODINE * I
    dXe_dt = (GAMMA_XENON * phi_sigma_f + LAMBDA_IODINE * I
              - LAMBDA_XENON * Xe - sigma_xe_phi * Xe)

    return dI_dt, dXe_dt


def xenon_reactivity(Xe: float) -> float:
    """Xenon-135 reactivity contribution.

    Args:
        Xe: xenon-135 concentration (atoms/m³)

    Returns:
        rho_xenon (dimensionless, ≤ 0)
    """
    return -SIGMA_XENON * Xe / NU_SIGMA_F


def rk4_step_xenon(I: float, Xe: float, n: float, dt: float) -> tuple[float, float]:
    """Advance I-135/Xe-135 by one RK4 step at constant power n.

    Args:
        I: iodine concentration (atoms/m³)
        Xe: xenon concentration (atoms/m³)
        n: normalized power level
        dt: timestep (s)

    Returns:
        (I_new, Xe_new)
    """
    dI1, dXe1 = xenon_derivatives(I, Xe, n)
    dI2, dXe2 = xenon_derivatives(I + 0.5 * dt * dI1, Xe + 0.5 * dt * dXe1, n)
    dI3, dXe3 = xenon_derivatives(I + 0.5 * dt * dI2, Xe + 0.5 * dt * dXe2, n)
    dI4, dXe4 = xenon_derivatives(I + dt * dI3, Xe + dt * dXe3, n)

    I_new = I + (dt / 6.0) * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4)
    Xe_new = Xe + (dt / 6.0) * (dXe1 + 2.0 * dXe2 + 2.0 * dXe3 + dXe4)

    return I_new, Xe_new
