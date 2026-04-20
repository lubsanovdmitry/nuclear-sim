# Nuclear Reactor Simulator

A vibe-coded web-based PWR (Pressurized Water Reactor) simulator with real-time physics, interactive control panel, and accident scenarios. Designed for educational use.

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
pip install -r requirements.txt
```

## Running the Simulator

```bash
uvicorn api.server:app --reload --port 8000
```

Then open your browser at [http://localhost:8000](http://localhost:8000).

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```text
nuclear-sim/
├── physics/        # Pure physics engine (PKE, thermal, xenon, decay heat)
├── plant/          # Plant systems (pumps, diesels, ECCS, pressurizer)
├── accidents/      # Accident scenario definitions
├── api/            # FastAPI server, WebSocket, simulation loop
└── frontend/       # HTML/CSS/JS control panel
```

See [SPEC.md](SPEC.md) for full physics and architecture documentation.
See [REACTOR_INSTRUCTIONS.md](REACTOR_INSTRUCTIONS.md) for operator guidance.
