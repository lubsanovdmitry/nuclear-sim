"""Two-phase thermal hydraulics for PWR/BWR.

Steam tables embedded as numpy arrays (no external library).
All interpolation via np.interp.
"""

import numpy as np

from physics.constants import (
    A_FUEL_NODE,
    C0_DRIFT,
    CHF_0,
    CP_COOL,
    G_NOMINAL,
    REGIME_FILM_EXIT_ALPHA,
    REGIME_FILM_EXIT_CHF_FRAC,
    V_GJ,
)

# ---------------------------------------------------------------------------
# Steam tables: pressure in bar, all other quantities in SI
# Source: SPEC2.md Phase 2 Section 2 (exact values)
# ---------------------------------------------------------------------------
STEAM_TABLE_P_BAR  = np.array([1.0,    10.0,   50.0,    75.0,    100.0,   155.0])
STEAM_TABLE_TSAT_K = np.array([373.15, 453.03, 537.09,  554.74,  584.15,  618.15])
STEAM_TABLE_HF     = np.array([417e3,  763e3,  1155e3,  1254e3,  1408e3,  1630e3])   # J/kg
STEAM_TABLE_HFG    = np.array([2257e3, 2015e3, 1641e3,  1548e3,  1317e3,  1000e3])   # J/kg
STEAM_TABLE_RHOF   = np.array([958.0,  887.0,  777.0,   748.0,   688.0,   594.0])    # kg/m³
STEAM_TABLE_RHOG   = np.array([0.60,   5.16,   25.4,    39.5,    55.5,    101.0])    # kg/m³

_P_MIN_PA: float = 1.0e5    # Pa — lower clamp for table lookup
_P_MAX_PA: float = 160.0e5  # Pa — upper clamp for table lookup


def _p_bar(pressure_pa: float) -> float:
    """Clamp and convert Pa → bar for table lookup."""
    return np.clip(pressure_pa, _P_MIN_PA, _P_MAX_PA) / 1.0e5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def saturation_temp(pressure_pa: float) -> float:
    """T_sat in K at given pressure (Pa). Interpolated from embedded steam table."""
    return float(np.interp(_p_bar(pressure_pa), STEAM_TABLE_P_BAR, STEAM_TABLE_TSAT_K))


def saturation_properties(pressure_pa: float) -> dict:
    """Saturation properties at pressure_pa.

    Returns dict with keys: h_f, h_fg, rho_f, rho_g (all SI).
    """
    p = _p_bar(pressure_pa)
    return {
        "h_f":   float(np.interp(p, STEAM_TABLE_P_BAR, STEAM_TABLE_HF)),
        "h_fg":  float(np.interp(p, STEAM_TABLE_P_BAR, STEAM_TABLE_HFG)),
        "rho_f": float(np.interp(p, STEAM_TABLE_P_BAR, STEAM_TABLE_RHOF)),
        "rho_g": float(np.interp(p, STEAM_TABLE_P_BAR, STEAM_TABLE_RHOG)),
    }


def thermodynamic_quality(t_cool_K: float, pressure_pa: float) -> float:
    """Thermodynamic quality x.

    Subcooled (T < T_sat): x = (T - T_sat) / 50  — negative
    Saturated/superheated:  x derived from approximate enthalpy
    Clamped to [-0.5, 1.0].
    """
    t_sat = saturation_temp(pressure_pa)
    if t_cool_K < t_sat:
        x = (t_cool_K - t_sat) / 50.0
    else:
        props = saturation_properties(pressure_pa)
        h_approx = CP_COOL * (t_cool_K - 273.15)
        x = (h_approx - props["h_f"]) / props["h_fg"]
    return float(np.clip(x, -0.5, 1.0))


def void_fraction(quality: float, pressure_pa: float) -> float:
    """Zuber-Findlay drift-flux void fraction.

    Returns 0.0 for quality <= 0.  Result clamped to [0.0, 0.95].
    """
    if quality <= 0.0:
        return 0.0

    props = saturation_properties(pressure_pa)
    rho_f = props["rho_f"]
    rho_g = props["rho_g"]

    x = quality
    # Protect denominator at very low quality
    x_safe = max(x, 0.01)

    denom = (
        C0_DRIFT * (x + (1.0 - x) * rho_g / rho_f)
        + rho_g * V_GJ / (G_NOMINAL * x_safe)
    )
    alpha = x / denom
    return float(np.clip(alpha, 0.0, 0.95))


def void_fraction_subcooled(
    t_cool_K: float,
    t_sat_K: float,
    heat_flux: float,
    pressure_pa: float,
) -> float:
    """Saha-Zuber subcooled void model.

    Returns 0.0 when bulk subcooling > 20 K or heat flux < 2e5 W/m².
    Result clamped to [0.0, 0.15].
    """
    subcooling = t_sat_K - t_cool_K
    if subcooling > 20.0 or heat_flux < 2.0e5:
        return 0.0
    alpha_sub = 0.1 * (heat_flux / 1.5e6) * max(0.0, 1.0 - subcooling / 20.0)
    return float(np.clip(alpha_sub, 0.0, 0.15))


def boiling_regime(
    alpha: float,
    heat_flux: float,
    chf: float,
    prev_regime: str | None = None,
) -> str:
    """Classify heat transfer regime.

    'single_phase'      — alpha == 0 and heat_flux < 0.3 * chf
    'subcooled_boiling' — alpha == 0 and heat_flux >= 0.3 * chf
    'nucleate_boiling'  — 0 < alpha <= 0.7 and heat_flux < chf
    'film_boiling'      — alpha > 0.7 OR heat_flux >= chf
    """
    if heat_flux >= chf or alpha > 0.7:
        return "film_boiling"
    # Hysteresis: once in film boiling, require lower alpha / CHF margin to retreat.
    if prev_regime == "film_boiling":
        if alpha > REGIME_FILM_EXIT_ALPHA or heat_flux >= REGIME_FILM_EXIT_CHF_FRAC * chf:
            return "film_boiling"
    if alpha > 0.0:
        return "nucleate_boiling"
    if heat_flux >= 0.3 * chf:
        return "subcooled_boiling"
    return "single_phase"


def heat_transfer_coefficient(regime: str, flow_fraction: float, alpha: float) -> float:
    """Heat transfer coefficient (W/m²K) for given regime."""
    if regime == "single_phase":
        return 30000.0 * max(0.1, flow_fraction)
    if regime == "subcooled_boiling":
        return 50000.0 * max(0.1, flow_fraction)
    if regime == "nucleate_boiling":
        return 80000.0 * (1.0 - alpha)
    # film_boiling — conservative worst-case
    return 2500.0


def critical_heat_flux(
    flow_fraction: float,
    pressure_pa: float,
    quality: float,
) -> float:
    """Simplified Bowring-style CHF correlation (W/m²).

    Minimum 3e5 W/m² (physical floor).
    """
    f_flow = max(0.15, flow_fraction)
    f_pres = max(0.3, 1.0 - abs(pressure_pa - 155.0e5) / 155.0e5)
    f_qual = max(0.2, 1.0 - 2.0 * max(0.0, quality))
    return float(max(3.0e5, CHF_0 * f_flow * f_pres * f_qual))


def dnbr(actual_heat_flux: float, chf: float) -> float:
    """Departure from Nucleate Boiling Ratio = CHF / q_actual."""
    return chf / max(actual_heat_flux, 1.0)


def actual_heat_flux_array(
    power_shape: np.ndarray,
    total_power: float,
    N: int = 10,
) -> np.ndarray:
    """Per-node heat flux array (W/m²).

    q[n] = power_shape[n] * total_power / (N * A_FUEL_NODE)
    """
    return power_shape * total_power / (N * A_FUEL_NODE)
