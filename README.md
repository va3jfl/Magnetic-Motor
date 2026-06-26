# Howard Johnson Magnetic-Gate Motor — Live Simulation

A 2-D physics simulation of Howard Johnson's permanent-magnet **"magnetic gate"**
motor (U.S. Patent 4,151,431): a magnet **cart** travels a circular track ringed
by a configurable array of stator magnets (the *gate*), with optional small
**rubber/ferrite flux-gate magnets** placed at the null zones — Johnson's actual
regauging mechanism. Everything is computed from a Coulombian (magnetic-charge)
force model, with a full live-dashboard GUI for experimentation.

This is a **pure magnetic / mechanical** machine — no electrical input, no
generator-load framing. Every number on screen is computed from the force model.

---

## What it does

- **Live cart canvas** — the rotor magnet circling the track, stator gate,
  rubber flux-gate aux magnets (drawn dark), optional field-line overlay, a
  **scale bar**, and a **dimension overlay** (track Ø, gap, magnet size, count)
  so the real-world size is always visible.
- **Work & energy calculators** — per-revolution / cumulative / instantaneous
  magnetic work, switching work, dissipated loss, net work (ΔKE), equivalent
  **horsepower**, **watt-hours**, RPM, torque, kinetic energy, field-energy
  fuel gauge.
- **Engineering & headroom readouts** — demagnetizing reverse field vs the
  stator's intrinsic coercivity (Hci), demag-limit gap, peak force per stator,
  track circumference, tooth pitch, pole charge, rotor inertia, surface speed,
  centripetal acceleration, gate-pass rate, angular momentum.
- **Rolling graphs** — power (magnetic / dissipated / net) and work-per-revolution.
- **Full control workshop** — magnet grades, gate pattern, stator count, air gap,
  skew, engage arc, retract, friction, rotor mass, track radius, demag
  timescale (log), **device scale** (0.25×–4×, true geometric+dynamic scaling),
  **sim-speed** (0.01×–8× slow-mo↔fast), flux-gate on/off + aux strength +
  aux position, **generator load**, push ⟲/⟳, pause, reset, field lines.
- **Presets** — *default*, *Johnson 4,151,431* (documented build), *Johnson
  flux-gate (tuned glide)*.
- **JSON I/O** — export the full stats snapshot, and save/load presets.

It **auto-starts** on launch — one push and the gate takes over.

---

## Physics model

Coulombian / magnetic-charge model (standard PM-machine approach):

- Each magnet = two point poles `±q = ±(Br/μ₀)·A`.
- Pole-pole force `F = (μ₀/4π)·q₁q₂/r²` along `r̂`.
- Force and torque by superposition over **all** poles (main stators **and**
  the flux-gate aux magnets), so field-warping from the aux magnets is automatic
  and visible in the field-line overlay.

Every reported quantity (torque, work/rev, RPM, demag field, force, headroom) is
computed from that model — nothing is added by hand to force an outcome.

### What the model shows (stated plainly)

| configuration | net work / rev | behaviour |
|---|---|---|
| **passive** (all static magnets, any strength/position) | **0** | conservative field — cogging cancels over a lap |
| **flux-gate aux + tuned geometry** | ~0 | reshapes the field: cuts the cogging barrier up to **73%**, so the rotor **glides** instead of grabbing (coasts on the start impulse) |
| **regauging** (timed engage/retract actuator) | **+3.5 J** (Johnson) | the actuator is the energy source — sustained continuous rotation |

The companion `fluxgate_sweep.py` searches passive flux-gate profiles for the
lowest cogging barrier; the best found (8 mm gap + aux 0.50 T at −20°) is shipped
as the **"tuned glide"** preset.

---

## Getting started

System Python on the machine may have no `pip`/`ensurepip`, so the project uses an
**isolated venv** that never touches system Python:

```bash
# one-time bootstrap (pip seeded into the venv only)
python3 -m venv .venv
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install numpy scipy matplotlib PySide6

# run the live GUI (needs a display)
.venv/bin/python magnetic_motor_ui.py
```

---

## Companion tools

| script | purpose |
|---|---|
| `magnetic_motor_ui.py` | the live GUI workshop (main app) |
| `calibrator.py` | engineering sizing: torque profile, steady-state RPM/torque, demag gap, structural force, gap & skew sweeps → `figs/calibrator.png` |
| `fluxgate_sweep.py` | searches **passive** flux-gate profiles for the minimum cogging barrier |
| `linear_gate.py` | Johnson's first (linear-track) prototype: cart released from rest at the gate entrance |
| `mag_smoketest.py`, `johnson_smoketest.py` | offscreen smoke tests + screenshots |

---

## File layout

```
magnetic_motor_ui.py        live GUI: canvas, calculators, engineering readouts, controls
calibrator.py               sizing / headroom calculator
fluxgate_sweep.py           passive flux-gate profile search
linear_gate.py              linear-track (first prototype) experiment
mag_smoketest.py            offscreen smoke test (default config)
johnson_smoketest.py        offscreen smoke test (Johnson geometry)
```

---

## Johnson geometry (encoded preset)

Documented build figures (from the patent and the compiled `motor.pdf`
evaluation, citing the 2011 Neo Teng Yi FEA thesis):

- **stator bars** — 4.0 in (100 mm) long, 1.0 in (25.4 mm) wide, 0.25 in (6 mm)
  thick, ends upturned into a shallow U, on a mu-metal apron cylinder.
- **armature** — 3 curved "banana" magnets, 3.125 in (79.4 mm), stepped and
  staggered off a 120° spacing, skewed in the direction of motion.
- **material** — Cobalt-Samarium (SmCo), Br ≈ 0.8–1.1 T.
- **air gap** — 4.4–4.6 mm (FEA); rotary drum 216.5 mm OD.

The 2-D model represents each pole face by an area-equivalent square of side
`√(width·thickness)`; the patent's 100 mm bar length is a structural dimension,
not injected into the pole spacing. Everything that enters the force law — Br,
pole-face area, gap, count, skew, flux-gate aux — is Johnson's.

---

## Notes

- The passive (all-static) field is conservative: net work per revolution is
  zero for any magnet arrangement, so a passive gate **coasts** on the start
  impulse (it can glide a long way at low friction) but does not self-accelerate.
- Sustained continuous rotation comes from the **regauging actuator** (the timed
  engage/retract), which is reported as separate "switching work."
- Magnet grades, Hci, and B_sat use representative reference ranges and are
  grade-dependent.
