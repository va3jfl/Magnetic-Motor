"""
calibrator.py -- engineering sizing calculator for the Howard Johnson magnetic
gate.  This is the "solve for the unknowns" tool: the UNKNOWN is the output
(RPM, torque, work/lap, headroom); the KNOWN inputs are the documented geometry
-- magnet grade / Br, dimensions, air gap, skew / offset, stator count, track
radius.  Outputs are *computed* from the same Coulombian force model the GUI
uses, the way any electric motor is sized.  No trusted measurement is required
to predict an output.

It reports:
  * the static torque profile T(theta) and the net work per lap, for the passive
    field (cogging, must integrate to ~0) and for the regauging gate (>0);
  * the predicted steady-state RPM and mean torque (from the dynamics);
  * the HEADROOM before the system hits a physical limit -- the demagnetization
    gap (where the rotor's reverse field exceeds the stator's intrinsic
    coercivity) and the structural attraction force the mount must hold;
  * sweeps over gap and skew showing the "play" the dimensions buy.

Run:  .venv/bin/python calibrator.py            (prints a report + figs/calibrator.png)
"""
from __future__ import annotations
import math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from magnetic_motor_ui import (GRADES, Gate, force_on_rotor, rotor_dims,
                               JOHNSON_PRESET, REF, MU0, MU0_4PI)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, QTimer

INV_4PI = 1.0 / (4.0 * math.pi)   # H-field prefactor (the force law uses MU0_4PI = mu0/4pi)

# --- material limit reference ranges (intrinsic coercivity Hci [A/m]) ---------
HCI = {                                       # intrinsic coercivity, representative
    "SmCo": 7.5e5, "NdFeB N42": 8.0e5, "NdFeB N52": 8.8e5, "Ferrite (ceramic)": 2.7e5,
    "AlNiCo": 1.2e5,
}
BSAT_MUMETAL = 0.65     # T -- mu-metal apron saturation (representative)


# ===========================================================================
# static torque profile over one full revolution
# ===========================================================================
def torque_profile(cfg, regauge=True, n_samples=3600):
    """T(theta) [N*m] over 0..2pi; returns (theta, T, work_per_rev [J])."""
    g = Gate(cfg)
    R = cfg["R_track"]
    th = np.linspace(0.0, 2.0 * math.pi, n_samples)
    T = np.empty_like(th)
    for k, a in enumerate(th):
        pos = np.array([R * math.cos(a), R * math.sin(a)])
        g._rotor_pos, g._rotor_ang = pos, a
        if regauge:
            g.update_switching(a)
        else:
            g.r[:] = g.Rs                  # passive: every magnet engaged (full cogging)
        _, tau = force_on_rotor(pos, a, cfg, g)
        T[k] = tau
    work = float(np.trapezoid(T, th))      # J per revolution
    return th, T, work


# ===========================================================================
# reverse demagnetizing field a stator sees from the rotor at closest approach
# ===========================================================================
def reverse_field(cfg, gap):
    """Magnitude [A/m] of the rotor's field along the stator magnetization axis,
    evaluated at the engaged stator's near face when the rotor is abreast of it."""
    c = dict(cfg); c["gap"] = gap
    g = Gate(c)
    br = GRADES[c["grade_rotor"]]
    rL, rW = rotor_dims(c); rarea = rW * rW
    rq = (br / MU0) * rarea
    R = c["R_track"]
    # rotor abreast of stator 0 (alpha[0]=0): rotor at (R,0), axis along x
    rotor_poles = np.array([[R + rL / 2.0, 0.0], [R - rL / 2.0, 0.0]])
    rotor_q = np.array([rq, -rq])
    # stator 0 near face (inward pole) at x = Rs - L/2
    face_x = g.Rs - g.L / 2.0
    P = np.array([face_x, 0.0])
    diff = P - rotor_poles
    d2 = np.sum(diff * diff, axis=1)
    d3 = d2 ** 1.5
    Hx = float(np.sum(INV_4PI * rotor_q * diff[:, 0] / d3))   # A/m (H-field uses 1/4pi)
    return abs(Hx)                                            # reverse (opposing) magnitude


def attraction_force(cfg, gap):
    """Peak |force| [N] between rotor and nearest engaged stator at this gap."""
    c = dict(cfg); c["gap"] = gap
    g = Gate(c); g.r[:] = g.Rs
    R = c["R_track"]
    worst = 0.0
    for a in np.linspace(0.0, 2.0 * math.pi / g.n, 60):      # over one tooth approach
        pos = np.array([R * math.cos(a), R * math.sin(a)])
        F, _ = force_on_rotor(pos, a, c, g)
        worst = max(worst, float(np.hypot(F[0], F[1])))
    return worst


# ===========================================================================
# steady-state RPM / mean torque from the actual dynamics (no measurement used)
# ===========================================================================
def steady_state(cfg, push=8.0, dwell=2.0):
    app = QApplication.instance() or QApplication(sys.argv)
    from magnetic_motor_ui import SimWorker
    w = SimWorker(dict(cfg))
    th = QThread(); w.moveToThread(th); th.started.connect(w.run); th.start(); w.go()
    w.push(push)
    out = {}
    def done():
        f = w.latest
        out["rpm"] = f.rpm
        out["rev"] = f.revolutions
        out["tau_mean"] = f.prop_last / (2.0 * math.pi)        # N*m = work per radian
        out["pmech"] = f.prop_last / max(f.rev_time, 1e-9)     # W   = work per lap / lap time
        out["work_rev"] = f.prop_last
        out["net_rev"] = f.net_last
        w.stop(); th.quit(); th.wait(2000); app.quit()
    QTimer.singleShot(int(dwell * 1000), done)
    app.exec()
    return out


# ===========================================================================
# report
# ===========================================================================
def line(s=""): print(s)

def run(cfg, label):
    c = cfg
    br_s = GRADES[c["grade_stator"]]; br_r = GRADES[c["grade_rotor"]]
    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)
    print("KNOWN INPUTS")
    print(f"  stator/rotor grade : {c['grade_stator']} / {c['grade_rotor']}  "
          f"(Br = {br_s:.2f} / {br_r:.2f} T)")
    print(f"  air gap            : {c['gap']*1e3:.2f} mm")
    print(f"  skew / offset      : {c['skew_deg']:+.1f} deg   pattern: {c['pattern']}")
    print(f"  stators / track R  : {c['n_stator']}   R_track = {c['R_track']*1e3:.1f} mm")
    sL, sW = c["mag_len"], c["mag_wid"]
    print(f"  stator pole face   : area-equiv side {sW*1e3:.2f} mm (from 25.4x6 mm), "
          f"len {sL*1e3:.2f} mm")

    line(); print("STATIC TORQUE PROFILE (computed)")
    _, Tp, Wp = torque_profile(c, regauge=False)
    _, Tr, Wr = torque_profile(c, regauge=True)
    print(f"  passive  : peak |T| = {np.max(np.abs(Tp))*1e3:7.2f} mN*m,  "
          f"net work/lap = {Wp*1e3:+8.3f} mJ   (cogging cancels -> ~0)")
    print(f"  regauging: peak |T| = {np.max(np.abs(Tr))*1e3:7.2f} mN*m,  "
          f"net work/lap = {Wr*1e3:+8.3f} mJ   (gate timing -> positive)")

    line(); print("PREDICTED OPERATING POINT (from dynamics, no measured source)")
    ss = steady_state(c)
    pmech = ss["pmech"]
    print(f"  steady RPM ~ {ss['rpm']:7.1f}   mean torque ~ {ss['tau_mean']*1e3:7.2f} mN*m")
    print(f"  work/lap ~ {ss['work_rev']*1e3:7.2f} mJ   net/lap ~ {ss['net_rev']*1e3:+7.2f} mJ")
    print(f"  -> mechanical power ~ {pmech:6.3f} W  ({pmech/745.7:.5f} hp)")

    line(); print("HEADROOM -- where the system gives way ('bows')")
    H = reverse_field(c, c["gap"])
    Hci = HCI.get(c["grade_stator"], 7.5e5)
    print(f"  demag    : reverse field at stator face = {H/1e3:7.1f} kA/m  vs  "
          f"Hci({c['grade_stator']}) = {Hci/1e3:.0f} kA/m   -> margin {H/Hci*100:5.1f}%")
    gdemag = demag_gap(c, Hci)
    print(f"             demag limit hit at gap <= {gdemag*1e3:5.2f} mm "
          f"(close the gap further -> stators demagnetize)")
    F = attraction_force(c, c["gap"])
    print(f"  struct   : peak magnet force/stator at {c['gap']*1e3:.1f} mm = {F:6.2f} N  "
          f"(the mount / apron must restrain this)")
    for gg in (0.003, 0.004, 0.010):
        print(f"             at {gg*1e3:4.1f} mm -> {attraction_force(c, gg):7.2f} N")
    return dict(Wp=Wp, Wr=Wr, ss=ss, H=H, Hci=Hci, gdemag=gdemag, F=F)


def demag_gap(cfg, Hci):
    lo, hi = 0.001, 0.040
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if reverse_field(cfg, mid) > Hci:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sweeps(cfg):
    line(); print("SWEEPS -- the play the dimensions / offset buy")
    print("  gap:    rpm      work/lap    demag margin")
    pts_g = []
    for gap in (0.003, 0.0038, 0.0045, 0.006, 0.008, 0.010):
        c = dict(cfg); c["gap"] = gap
        ss = steady_state(c, dwell=1.6)
        H = reverse_field(c, gap); Hci = HCI.get(c["grade_stator"], 7.5e5)
        pts_g.append((gap*1e3, ss["rpm"], ss["work_rev"]*1e3, H/Hci*100))
        flag = "  <-- demag" if H > Hci else ""
        print(f"   {gap*1e3:5.2f}mm  {ss['rpm']:7.1f}   {ss['work_rev']*1e3:+8.2f} mJ   "
              f"{H/Hci*100:5.1f}%{flag}")
    print("  skew:   rpm      work/lap")
    pts_s = []
    for sk in (0.0, 3.0, 6.0, 9.0, 12.0):
        c = dict(cfg); c["skew_deg"] = sk
        ss = steady_state(c, dwell=1.6)
        pts_s.append((sk, ss["rpm"], ss["work_rev"]*1e3))
        print(f"   {sk:+5.1f}deg {ss['rpm']:7.1f}   {ss['work_rev']*1e3:+8.2f} mJ")
    return pts_g, pts_s


def figure(cfg, pts_g, pts_s):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    th, Tp, _ = torque_profile(cfg, regauge=False)
    th2, Tr, _ = torque_profile(cfg, regauge=True)
    ax[0, 0].plot(np.degrees(th), Tp * 1e3, label="passive (cogging)")
    ax[0, 0].plot(np.degrees(th2), Tr * 1e3, label="regauging")
    ax[0, 0].set_title("Torque over one revolution"); ax[0, 0].set_ylabel("mN*m")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    gaps = np.linspace(0.003, 0.012, 40)
    Hci = HCI.get(cfg["grade_stator"], 7.5e5)
    Hv = [reverse_field(cfg, g) / 1e3 for g in gaps]
    ax[0, 1].plot(gaps * 1e3, Hv, label="reverse field at stator face")
    ax[0, 1].axhline(Hci / 1e3, color="r", ls="--", label=f"Hci = {Hci/1e3:.0f} kA/m")
    ax[0, 1].axvline(cfg["gap"] * 1e3, color="g", ls=":", label=f"Johnson gap {cfg['gap']*1e3:.1f} mm")
    ax[0, 1].set_title("Demagnetization headroom"); ax[0, 1].set_xlabel("gap mm"); ax[0, 1].set_ylabel("kA/m")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    g0 = [p[0] for p in pts_g]; ax[1, 0].plot(g0, [p[1] for p in pts_g], "o-", label="RPM")
    ax[1, 0].set_title("RPM vs gap"); ax[1, 0].set_xlabel("gap mm"); ax[1, 0].grid(alpha=0.3)
    sk = [p[0] for p in pts_s]; ax[1, 1].plot(sk, [p[2] for p in pts_s], "s-", color="C2")
    ax[1, 1].set_title("work/lap vs skew"); ax[1, 1].set_xlabel("skew deg"); ax[1, 1].set_ylabel("mJ")
    ax[1, 1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figs/calibrator.png", dpi=110); plt.close(fig)
    print("\nsaved figs/calibrator.png")


if __name__ == "__main__":
    res = run(JOHNSON_PRESET, "Howard Johnson magnetic gate -- patent 4,151,431")
    pts_g, pts_s = sweeps(JOHNSON_PRESET)
    figure(JOHNSON_PRESET, pts_g, pts_s)
