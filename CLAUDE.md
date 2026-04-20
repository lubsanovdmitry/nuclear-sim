# CLAUDE.md

## Read First

Read SPEC.md before writing any code.

## Style

- Type hints on all functions
- Constants only in physics/constants.py, never hardcoded elsewhere
- `PlantState` defined in api/state.py — import it, don't redeclare

## Key Physics Rules

- PKE: RK4 with 0.1 ms inner sub-steps (1000 per 100 ms tick)
- Thermal: explicit Euler at 0.1 s (stable — thermal τ >> dt)  
- sigma_Xe = 2.6e-22 m² (2.6 Megabarns — verify units before use)
- T_in is dynamic — see SBO/LOOP logic in SPEC Section 3

## Units (strict)

- Internal: SI (K, Pa, W, m, s)
- WebSocket output: °C, bar, %, °C
- Convert only in api/server.py _state_to_ws()

## Module Rules

- physics/ = pure functions, numpy only, no side effects, no I/O
- accidents/*.py = pure trigger functions, used by server.py AND tests
- PlantState lives in api/state.py — import, never redeclare
- scram_bypasses: implement as stub (always []) in Phase 1

## Session Protocol

/clear between every session. One file per session.
Run pytest tests/ -v before finishing — all prior tests must pass.
Use sonnet for plant/ api/ frontend/; use max for physics/ and integration.

## Run

pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000
