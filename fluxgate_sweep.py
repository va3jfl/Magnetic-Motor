"""
fluxgate_sweep.py -- search PASSIVE flux-gate profiles (rubber aux magnets at the
null zones, NO actuator) for the configuration that best cancels the cogging
(dead-stop), so the rotor glides instead of grabbing.  This is the tuning
Howard Johnson was doing by hand.

Metric: the cogging barrier = peak-to-peak of the rotor's magnetic potential
U(theta) = -integral T dtheta over one revolution.  Lower barrier = flatter
field = fewer/smaller dead-stops = longer glide.  Net work/rev is reported too
(it is ~0 for any static config -- the field is conservative -- but the SHAPE
of the field is what we are tuning).

Run:  .venv/bin/python fluxgate_sweep.py
"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from calibrator import torque_profile
from magnetic_motor_ui import JOHNSON_PRESET, GRADES

BASE = dict(JOHNSON_PRESET); BASE["switching"] = False        # pure passive


def barrier_and_peak(cfg):
    th, T, work = torque_profile(cfg, regauge=False)
    U = -np.concatenate([[0.0], np.cumsum(0.5 * (T[:-1] + T[1:]) * np.diff(th))])  # -int T dth
    return (U.max() - U.min()), float(np.max(np.abs(T))), work


def main():
    b0, p0, w0 = barrier_and_peak(dict(BASE, flux_gate=False))
    print(f"baseline (no aux):  barrier = {b0*1e3:8.2f} mJ   peak|T| = {p0*1e3:8.1f} mN·m   "
          f"net/rev = {w0*1e6:+.1f} µJ")
    print("sweeping aux strength x position ...\n")
    best = None
    rows = []
    for br in np.linspace(0.0, 1.2, 13):
        for off in np.linspace(-30.0, 30.0, 13):
            b, p, w = barrier_and_peak(dict(BASE, flux_gate=True, aux_br=float(br),
                                            aux_offset_deg=float(off)))
            rows.append((b, br, off, p))
            if best is None or b < best[0]:
                best = (b, float(br), float(off), p)
    rows.sort()
    print("lowest-cogging passive flux-gate profiles:")
    print(f"{'barrier mJ':>11} {'aux_br T':>8} {'offset°':>8} {'peak|T| mN·m':>13}")
    for b, br, off, p in rows[:6]:
        star = "  <- best" if b == best[0] else ""
        print(f"{b*1e3:11.2f} {br:8.2f} {off:8.1f} {p*1e3:13.1f}{star}")
    print(f"\nbest profile: aux_br={best[1]:.2f} T  aux_offset={best[2]:+.1f}°  "
          f"-> barrier {best[0]*1e3:.2f} mJ (was {b0*1e3:.2f} mJ, "
          f"{(1-best[0]/b0)*100:.0f}% flatter)")

    # a few more knobs around the best: gap and skew
    bb, bbr, boff, _ = best
    print("\nrefining gap + skew around best aux ...")
    best2 = (bb, bbr, boff, BASE["gap"], BASE["skew_deg"])
    for gap in (0.003, 0.0045, 0.006, 0.008):
        for sk in (0.0, 3.0, 6.0, 9.0):
            cfg = dict(BASE, flux_gate=True, aux_br=bbr, aux_offset_deg=boff,
                       gap=gap, skew_deg=sk)
            b, p, _ = barrier_and_peak(cfg)
            if b < best2[0]:
                best2 = (b, bbr, boff, gap, sk)
    print(f"best overall: aux_br={best2[1]:.2f} T offset={best2[2]:+.1f}° gap={best2[3]*1e3:.2f} mm "
          f"skew={best2[4]:.0f}° -> barrier {best2[0]*1e3:.2f} mJ ({(1-best2[0]/b0)*100:.0f}% flatter than baseline)")
    print("\nPROFILE (paste into JOHNSON_FLUXGATE_PRESET):")
    print(f"  aux_br={best2[1]}, aux_offset_deg={best2[2]}, gap={best2[3]}, skew_deg={best2[4]}")


if __name__ == "__main__":
    main()
