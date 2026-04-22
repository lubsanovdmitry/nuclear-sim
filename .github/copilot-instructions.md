# Copilot Instructions

## Start Here

- Read `SPEC.md` and `SPEC2.md` before making code changes.
- If this file conflicts with the specs, follow the specs.

## Code Style

- Use type hints on all functions.
- Keep constants in `physics/constants.py` only.
- Import `PlantState` from `api/state.py`; do not redefine it.

## Simulation Rules

- PKE: RK4 with 0.1 ms inner sub-steps (1000 per 100 ms tick).
- Thermal: explicit Euler with 0.1 s timestep.
- Use `sigma_Xe = 2.6e-22 m^2`.
- Keep `T_in` dynamic per SBO/LOOP behavior in `SPEC.md`.

## Boundaries

- `physics/`: pure functions, numpy only, no I/O or side effects.
- `accidents/`: pure trigger functions shared by API and tests.
- Convert SI to websocket display units only in `api/server.py` (`_state_to_ws()`).

## Validation

- Keep existing interfaces backward compatible.
- Run `pytest tests/ -v` before finalizing.
