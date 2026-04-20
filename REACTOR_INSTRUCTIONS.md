# Reactor Operator Instructions

This guide explains how to operate the PWR simulator from startup to shutdown, and how to respond to accident scenarios.

---

## Starting the Simulator

```bash
uvicorn api.server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser. The reactor starts in **cold standby** (all rods inserted, pumps off).

---

## Normal Startup Procedure

### 1. Start Coolant Pumps

Click **Start** on all 4 coolant pump buttons. Verify flow rate reaches 100% on the flow indicator.

### 2. Withdraw Control Rods Gradually

Use the rod bank sliders (Banks A → B → C → D). Withdraw in small steps — a few percent at a time. Watch the **power meter** and **reactivity display**.

- Reactivity (rho_total) should approach zero as you near criticality.
- Power will begin rising when rho > 0.

### 3. Bring Reactor to Criticality

Adjust rods until power stabilizes at a low level (~1–5%). The reactor is now **critical**.

### 4. Raise Power to 100%

Continue withdrawing rods slowly to raise power. Monitor:

- **T_fuel** — must stay below 1200°C (cladding limit)
- **T_cool** — nominal ~305°C
- **RCS Pressure** — nominal ~155 bar

Allow temperatures to stabilize between each rod adjustment (~10–30 seconds).

### 5. Steady-State Operation

At 100% power, all parameters should be stable. The xenon worth bar will show a small negative value — this is normal equilibrium xenon.

---

## Normal Shutdown Procedure

### 1. Reduce Power

Insert rod banks gradually (D → C → B → A) to lower power to ~5%.

### 2. SCRAM

Press the **Manual SCRAM** button. All rods insert fully. Power drops rapidly to decay heat levels (~3–5% immediately, decaying over hours).

### 3. Maintain Cooling

Keep pumps running to remove decay heat. Do **not** shut down pumps immediately after SCRAM — decay heat can damage the core.

> **Xenon Pit Warning:** After a full-power shutdown, xenon-135 builds up and peaks ~3–6 hours later. The reactor cannot restart during this period. Wait ~30–40 hours for xenon to clear before attempting restart.

---

## Control Panel Reference

| Control | Function |
|---|---|
| Rod bank sliders (A–D) | Withdraw (right) or insert (left) rod banks |
| Boron injection | Adds negative reactivity; use for long-term shutdown |
| Manual SCRAM | Inserts all rods instantly |
| Pump Start/Stop (×4) | Controls each of the 4 coolant pumps |
| Diesel Start/Stop (×2) | Starts emergency diesel generators |
| ECCS Arm/Disable | Arms the emergency core cooling system |

### Alarms

- **Yellow** — Caution: parameter approaching limit
- **Red** — Emergency: immediate action required

Key alarm setpoints:

- High power: > 110% nominal
- High fuel temp: > 1000°C (caution), > 1200°C (emergency/cladding failure)
- High RCS pressure: > 170 bar
- Low RCS pressure: < 120 bar
- Low coolant flow: < 50%

---

## Accident Scenarios

Select a scenario from the **Scenario Panel** dropdown and press **Trigger**.

### LOOP — Loss of Offsite Power

All AC power lost. Coolant pumps begin coastdown.

- **Action:** Monitor diesel generator start (10–15s). Verify pumps restore on diesel power. Watch T_fuel — must stay below 1200°C. If diesels fail, this becomes an SBO.

### Station Blackout (SBO)

Both diesels fail. No pump power available.

- **Action:** Rely on natural circulation only. Passive ECCS accumulators will inject if pressure drops. Monitor temps closely — no active intervention available.

### LOCA — Loss of Coolant Accident

Large break in primary loop. Rapid depressurization.

- **Action:** Verify ECCS activates automatically (HPSI above 100 bar, LPSI below 20 bar). Do not disable ECCS. Monitor core cooling.

### Control Rod Ejection

One rod group ejects instantly — large positive reactivity insertion.

- **Action:** Observe Doppler feedback self-limiting the power spike. SCRAM fires automatically. If reactivity insertion exceeds design basis, fuel damage occurs.

### Xenon Pit

Reactor shut down at 100% power; restart attempted at t+3 hr.

- **Action:** Observe that rods cannot achieve criticality. Wait for xenon to clear (~30–40 hr). This scenario is educational — no corrective action possible.

---

## Challenge Mode

Enable **Challenge Mode** in the Scenario Panel to set diesel failure probability to 10%. LOOP scenarios may become SBOs — test your emergency response.

---

## Reset

Press **Reset** in the Scenario Panel to return the reactor to cold standby at any time.
