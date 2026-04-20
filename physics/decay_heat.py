"""ANS-5.1 11-group decay heat model — pure functions, numpy only."""

import numpy as np

from physics.constants import ANS51_A, ANS51_ALPHA, NOMINAL_POWER_W


def decay_heat_fraction(t: float) -> float:
    """Decay heat as a fraction of pre-shutdown operating power.

    Uses the ANS-5.1-1979 11-group exponential fit for U-235 fission products
    under the infinite-irradiation assumption (conservative for steady-state ops).

    Q_decay(t) / Q0 = Σ_k A_k · exp(−α_k · t)

    Physical note: sum(A_k) ≈ 6.3% at t=0, decaying to ~1.1% at 1 hour.

    Args:
        t: time after shutdown (s), must be ≥ 0

    Returns:
        dimensionless fraction ∈ (0, 1)
    """
    return float(np.dot(ANS51_A, np.exp(-ANS51_ALPHA * t)))


def decay_heat_power(t: float, nominal_power: float = NOMINAL_POWER_W) -> float:
    """Decay heat power in watts.

    Args:
        t: time after shutdown (s)
        nominal_power: reactor operating power before shutdown (W)

    Returns:
        Q_decay in watts
    """
    return decay_heat_fraction(t) * nominal_power
