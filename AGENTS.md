# AGENTS.md

## Read First

- Read `SPEC.md` and `SPEC2.md` before writing or changing code.
- Treat those two files as the source of truth for behavior and physics.

## Scope and Architecture

- Keep `physics/` pure: numpy-only math, no I/O, no side effects.
- Keep `accidents/` as pure trigger helpers used by `api/server.py` and tests.
- `PlantState` is defined in `api/state.py`; import it, never redeclare it.

## Core Rules

- Add type hints to all functions.
- Put constants only in `physics/constants.py`.
- Internal units are SI (`K`, `Pa`, `W`, `m`, `s`).
- Convert to UI units only in `api/server.py` (`_state_to_ws()`).

## Physics Invariants

- PKE uses RK4 with 0.1 ms inner sub-steps (1000 per 100 ms tick).
- Thermal model uses explicit Euler at 0.1 s.
- Xenon cross section is `sigma_Xe = 2.6e-22 m^2`.
- `T_in` is dynamic (SBO/LOOP logic from SPEC Section 3).

## Compatibility and Testing

- Preserve backward compatibility unless task explicitly says otherwise.
- Run `pytest tests/ -v` before finishing and keep existing tests passing.
- In Phase 1, `scram_bypasses` is a stub that always returns `[]`.
