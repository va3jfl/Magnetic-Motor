"""
Howard Johnson-style permanent-magnet "magnetic gate" motor -- continuous
circular version.  Standalone PySide6 app (same style as the pulse-motor GUI).

A magnet "cart" (rotor) travels a circular track ringed by a configurable array
of stator magnets (the gate).  Two physical mechanisms are modelled honestly,
both computed from the same force model -- nothing is added to force an outcome:

  1. PASSIVE GATE -- static magnets only.  The magnet-magnet force is
     conservative, so the rotor is captured in a potential well (cogging) and
     stops after a push.

  2. REGAUGING / FLUX-SWITCHING (Johnson's "gate" timing) -- each stator magnet
     is engaged (held close) only while the rotor approaches it, and retracted
     (pulled radially away) the instant the rotor passes.  The next magnet ahead
     then takes over.  This timed switching is what lets the rotor loop
     continuously.  The work the gate does on the rotor (propulsion) and the work
     the actuator spends to yank each magnet out of the field (switching) are
     both reported as raw per-revolution numbers.

  3. DEMAGNETIZATION -- magnets lose strength over time (Br decays on a
     configurable timescale).  A field-energy "fuel gauge" shows the finite
     stored magnetic energy being drawn down.

FORCE MODEL (Coulombian / magnetic-charge): each magnet = two point poles
+/-q = +/-(Br/mu0)*A; pole-pole force F=(mu0/4pi) q1 q2 / r^2 along r_hat;
force/torque by superposition.  Standard PM-machine approach (Purdue thesis;
ACES journal; Magpylib).  Every reported number is computed from it.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal, QPointF, QRectF, QElapsedTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                               QGridLayout, QLabel, QSlider, QPushButton, QGroupBox,
                               QSizePolicy, QFrame, QCheckBox, QComboBox, QScrollArea,
                               QFileDialog)

MU0 = 4.0e-7 * math.pi
MU0_4PI = 1.0e-7

GRADES = {
    "NdFeB N35": 1.17, "NdFeB N40": 1.22, "NdFeB N42": 1.28, "NdFeB N45": 1.32,
    "NdFeB N48": 1.37, "NdFeB N50": 1.40, "NdFeB N52": 1.48,
    "SmCo": 1.05, "AlNiCo": 0.65, "Ferrite (ceramic)": 0.40,
}

REF = dict(
    grade_rotor="NdFeB N42", grade_stator="NdFeB N42",
    n_stator=12, pattern="uniform", base_orient="radial",
    gap=0.020, skew_deg=0.0, mag_len=0.012, mag_wid=0.010,
    mag_len_rotor=0.012, mag_wid_rotor=0.010,   # armature dims (default = stator dims)
    R_track=0.080, friction=1.5e-4, drag=3.0e-6, coulomb=1.0e-5,
    rotor_mass=0.030, time_scale=1.0,
    switching=True, engage_arc_deg=24.0, retract_mm=30.0,
    demag_tau=4.0e4,           # seconds for Br to decay by 1/e (very slow by default)
    flux_gate=False, aux_br=0.40, aux_offset_deg=0.0,   # Johnson null-zone rubber/ferrite gate magnets
    load=0.0,                  # generator load coefficient (N·m·s/rad); p_gen = load*omega^2
)


def rotor_dims(cfg):
    """Armature (rotor) magnet length & pole-face side; fall back to stator dims."""
    return cfg.get("mag_len_rotor", cfg["mag_len"]), cfg.get("mag_wid_rotor", cfg["mag_wid"])


# --- Geometry presets --------------------------------------------------------
# Howard Johnson permanent-magnet motor, U.S. Patent 4,151,431.  Documented
# build figures (from the supplied motor.pdf evaluation, citing the patent and
# the 2011 Neo Teng Yi FEA thesis):
#   * stator bars   : 4.0 in (100 mm) long, 1.0 in (25.4 mm) wide, 0.25 in
#                     (6 mm) thick; ends upturned into a shallow U; on a
#                     mu-metal apron cylinder that concentrates flux into the gap.
#   * armature      : 3 curved "banana" magnets, 3.125 in (79.4 mm), stepped and
#                     staggered (deliberately offset from a 120 deg spacing),
#                     skewed in the direction of motion.
#   * material      : Cobalt-Samarium (SmCo), Br ~0.8-1.1 T.
#   * working gap   : 4.4-4.6 mm (FEA).
#   * rotary drum   : Hohl V-magnet derivative, 2-V = 216.5 mm, 3-V = 325 mm OD.
#   * recorded RPM  : NONE -- "quantitative velocity records are absent".  Only
#                     self-starting + unidirectional force were demonstrated.
#
# The 2-D Coulombian gate model represents each pole face by an area-equivalent
# square of side sqrt(width*thickness); the magnet length is the pole spacing.
# The patent's 100 mm bar length is a structural dimension, not a facing-pole
# spacing, so it is documented above rather than injected into mag_len (which
# would push the poles off the gap).  Everything that DOES enter the force law
# -- Br, pole-face area, gap, count, skew, switching -- is Johnson's.
JOHNSON_PRESET = dict(
    grade_rotor="SmCo", grade_stator="SmCo",          # documented material
    n_stator=12, pattern="johnson_skew", base_orient="radial",
    gap=0.0045, skew_deg=6.0,                          # documented 4.5 mm gap + deliberate skew
    mag_len=0.016, mag_wid=math.sqrt(0.0254 * 0.006),          # area-equiv face of 25.4x6 mm stator
    mag_len_rotor=0.016, mag_wid_rotor=math.sqrt(0.025 * 0.006),  # 79.4 mm banana armature face
    R_track=0.108,                                     # 2-V drum 216.5 mm OD -> 108 mm radius
    friction=5.0e-6, drag=1.0e-9, coulomb=1.0e-8, rotor_mass=0.030, time_scale=1.0,
    switching=False, engage_arc_deg=24.0, retract_mm=30.0, demag_tau=4.0e4,
    flux_gate=True, aux_br=0.40, aux_offset_deg=0.0, load=0.0,
)

# Passive flux-gate profile found by fluxgate_sweep.py: the aux magnets + a wider
# gap flatten the cogging barrier 73% (416 -> 112 mJ) so the rotor glides instead
# of grabbing.  Net work/rev is still 0 (static field), so it coasts on the start
# impulse -- this is the field-shaping Johnson was tuning, not a self-runner.
JOHNSON_GLIDE_PRESET = dict(
    grade_rotor="SmCo", grade_stator="SmCo",
    n_stator=12, pattern="johnson_skew", base_orient="radial",
    gap=0.008, skew_deg=0.0,
    mag_len=0.016, mag_wid=math.sqrt(0.0254 * 0.006),
    mag_len_rotor=0.016, mag_wid_rotor=math.sqrt(0.025 * 0.006),
    R_track=0.108,
    friction=5.0e-6, drag=1.0e-9, coulomb=1.0e-8, rotor_mass=0.030, time_scale=1.0,
    switching=False, engage_arc_deg=24.0, retract_mm=30.0, demag_tau=4.0e4,
    flux_gate=True, aux_br=0.50, aux_offset_deg=-20.0, load=0.0,
)

PRESETS = {"default": dict(REF), "Johnson 4,151,431": dict(JOHNSON_PRESET),
           "Johnson flux-gate (tuned glide)": dict(JOHNSON_GLIDE_PRESET)}


# --------------------------------------------------------------------------- physics
class Gate:
    """Stator magnet array with per-magnet radial position (engaged / retracted)."""

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.br0 = GRADES[self.cfg["grade_stator"]]
        self.br = self.br0
        self._build_geom()

    def _build_geom(self):
        c = self.cfg
        n = int(c["n_stator"])
        self.n = n
        self.alpha = np.array([2.0 * math.pi * i / n for i in range(n)])
        skew = math.radians(c["skew_deg"])
        ang = []
        for i in range(n):
            a = self.alpha[i] if c["base_orient"] == "radial" else self.alpha[i] + math.pi / 2.0
            a += skew * i
            if c["pattern"] == "alternating":
                a += math.pi * (i % 2)
            elif c["pattern"] == "paired":
                a += math.pi * ((i // 2) % 2)
            elif c["pattern"] == "johnson_skew":
                # Johnson's deliberate stagger: offset alternate stators by half a
                # tooth to distribute the discrete force impulses across the cycle
                # (he skews the armature; represented here as a progressive gate
                # skew that breaks the symmetric dead-stop / "null moment").
                a -= (math.pi / n) * (0.5 if (i % 2) else -0.5)
            ang.append(a)
        self.ang = np.array(ang)
        self.Rs = c["R_track"] + c["gap"]
        self.L = c["mag_len"]
        self.area = c["mag_wid"] ** 2
        self.r = np.full(n, self.Rs)             # current radius of each magnet
        self.volume_total = n * self.L * self.area
        # which side of the rotor to engage so the gate runs forward (CW). The skew
        # pattern biases the dipole-dipole thrust one way; johnson_skew thrusts when
        # the magnets just behind are engaged, so it uses the opposite side.
        self.gate_dir = -1 if c["pattern"] == "johnson_skew" else 1
        # auxiliary flux-gate magnets: small weaker (rubber/ferrite) magnets at the
        # null zones (inter-stator midpoints + offset) that warp the static field.
        if c.get("flux_gate"):
            off = math.radians(c.get("aux_offset_deg", 0.0))
            self.aux_n = n
            self.aux_alpha = self.alpha + math.pi / n + off
            self.aux_R = self.Rs
            self.aux_br = c.get("aux_br", 0.40)
            self.aux_L = self.L * 0.5
            self.aux_W = c["mag_wid"] * 0.6
            self.aux_area = self.aux_W ** 2
            self.aux_ang = self.aux_alpha.copy()
        else:
            self.aux_n = 0

    def reconfigure(self, cfg: dict):
        self.cfg = dict(cfg)
        self.br0 = GRADES[self.cfg["grade_stator"]]
        # keep current relative demag if any
        self._build_geom()

    def field_energy(self):
        # stored magnetic energy ~ (Br^2 / 2mu0) * magnet volume
        return (self.br * self.br) / (2.0 * MU0) * self.volume_total

    def poles(self, rotor_pos=None):
        """Return charges (2n,) and positions (2n,2) at current radii and Br."""
        q = (self.br / MU0) * self.area
        hx = (self.L / 2.0) * np.cos(self.ang)
        hy = (self.L / 2.0) * np.sin(self.ang)
        cx = self.r * np.cos(self.alpha)
        cy = self.r * np.sin(self.alpha)
        pos = np.empty((2 * self.n, 2))
        pos[0::2, 0] = cx + hx; pos[0::2, 1] = cy + hy      # + poles
        pos[1::2, 0] = cx - hx; pos[1::2, 1] = cy - hy      # - poles
        charges = np.empty(2 * self.n)
        charges[0::2] = q
        charges[1::2] = -q
        if getattr(self, "aux_n", 0) > 0:           # flux-gate aux magnets (static)
            qa = (self.aux_br / MU0) * self.aux_area
            ahx = (self.aux_L / 2.0) * np.cos(self.aux_ang)
            ahy = (self.aux_L / 2.0) * np.sin(self.aux_ang)
            acx = self.aux_R * np.cos(self.aux_alpha)
            acy = self.aux_R * np.sin(self.aux_alpha)
            apos = np.empty((2 * self.aux_n, 2))
            apos[0::2, 0] = acx + ahx; apos[0::2, 1] = acy + ahy
            apos[1::2, 0] = acx - ahx; apos[1::2, 1] = acy - ahy
            acharges = np.empty(2 * self.aux_n)
            acharges[0::2] = qa; acharges[1::2] = -qa
            charges = np.concatenate([charges, acharges])
            pos = np.concatenate([pos, apos], axis=0)
        return charges, pos

    def update_switching(self, theta: float):
        """Engage/retract stator magnets based on rotor angle. Returns switching work [J].
        The gate has a preferred direction (Johnson's "unidirectional force") set by the
        stagger/skew; it sustains that direction once hand-started, like a ratchet."""
        c = self.cfg
        if not c["switching"]:
            self.r[:] = self.Rs
            return 0.0
        arc = math.radians(c["engage_arc_deg"])
        margin = 0.08 * arc              # retract just before abreast -- Johnson's "null
        dR = c["retract_mm"] * 1e-3      # moment": the gate releases so the rotor never
        work = 0.0                       # meets the repulsive exit half of the profile
        for i in range(self.n):
            delta = (self.alpha[i] - theta) % (2.0 * math.pi)       # >0 = ahead in +dir
            ahead = delta if self.gate_dir >= 0 else (2.0 * math.pi - delta) % (2.0 * math.pi)
            engaged = margin < ahead < arc
            r_target = self.Rs if engaged else self.Rs + dR
            if abs(r_target - self.r[i]) > 1e-6:
                work += self._switch_force(i) * abs(r_target - self.r[i])
                self.r[i] = r_target
        return work

    def _switch_force(self, i: int) -> float:
        """Radial force magnitude on stator magnet i from the rotor (for switching work)."""
        # reuse the current rotor position stored by the worker
        rp = getattr(self, "_rotor_pos", None)
        if rp is None:
            return 0.0
        # force on rotor from magnet i, then negate (Newton 3rd) and take radial comp
        qi = np.array([ (self.br / MU0) * self.area, -(self.br / MU0) * self.area ])
        ax = self.alpha[i]
        cx, cy = self.r[i] * math.cos(ax), self.r[i] * math.sin(ax)
        hx = (self.L / 2.0) * math.cos(self.ang[i]); hy = (self.L / 2.0) * math.sin(self.ang[i])
        spos = np.array([[cx + hx, cy + hy], [cx - hx, cy - hy]])
        F = np.zeros(2)
        # rotor poles
        rotor_ang = getattr(self, "_rotor_ang", 0.0)
        rcfg = self.cfg
        rL, rW = rotor_dims(rcfg)
        rarea = rW * rW; rbr = GRADES[rcfg["grade_rotor"]]
        rq_val = (rbr / MU0) * rarea
        rhx = (rL / 2.0) * math.cos(rotor_ang); rhy = (rL / 2.0) * math.sin(rotor_ang)
        rpos = np.array([[rp[0] + rhx, rp[1] + rhy], [rp[0] - rhx, rp[1] - rhy]])
        rq = np.array([rq_val, -rq_val])
        soft2 = (0.25 * rW) ** 2
        for k in range(2):
            diff = rpos[k] - spos            # (2,2)
            d2 = np.sum(diff * diff, axis=1) + soft2
            coef = MU0_4PI * rq[k] * qi / (d2 ** 1.5)
            F[0] += np.sum(coef * diff[:, 0])
            F[1] += np.sum(coef * diff[:, 1])
        # force on stator = -F_on_rotor_from_i
        Fr = -(F[0] * math.cos(ax) + F[1] * math.sin(ax))
        return abs(Fr)


def force_on_rotor(rotor_pos, rotor_ang, cfg, gate: Gate):
    """Net force [N], wheel torque [N*m] on the rotor from all (current) stator poles."""
    br = GRADES[cfg["grade_rotor"]]
    L, W = rotor_dims(cfg)
    area = W * W
    rq_val = (br / MU0) * area
    hx = (L / 2.0) * math.cos(rotor_ang); hy = (L / 2.0) * math.sin(rotor_ang)
    rpos = np.array([[rotor_pos[0] + hx, rotor_pos[1] + hy],
                     [rotor_pos[0] - hx, rotor_pos[1] - hy]])
    rq = np.array([rq_val, -rq_val])
    sq, spos = gate.poles()
    F = np.zeros(2)
    soft2 = (0.25 * W) ** 2
    for k in range(2):
        diff = rpos[k] - spos
        d2 = np.sum(diff * diff, axis=1) + soft2
        coef = MU0_4PI * rq[k] * sq / (d2 ** 1.5)
        F[0] += np.sum(coef * diff[:, 0])
        F[1] += np.sum(coef * diff[:, 1])
    tau = rotor_pos[0] * F[1] - rotor_pos[1] * F[0]
    return F, tau


# --- live engineering readouts (headroom / geometry / motion) ----------------
INV_4PI = 1.0 / (4.0 * math.pi)          # H-field prefactor (force uses MU0_4PI = mu0/4pi)
HCI = {                                  # intrinsic coercivity [A/m], representative
    "SmCo": 7.5e5, "NdFeB N35": 8.0e5, "NdFeB N40": 8.0e5, "NdFeB N42": 8.0e5,
    "NdFeB N45": 8.2e5, "NdFeB N48": 8.4e5, "NdFeB N50": 8.6e5, "NdFeB N52": 8.8e5,
    "AlNiCo": 1.2e5, "Ferrite (ceramic)": 2.7e5,
}


def reverse_field(cfg, gap):
    """Magnitude [A/m] of the rotor's field along the stator axis at the engaged
    stator's near face, rotor abreast -- the reverse demagnetizing field."""
    c = dict(cfg); c["gap"] = gap
    g = Gate(c)
    br = GRADES[c["grade_rotor"]]
    rL, rW = rotor_dims(c); rarea = rW * rW
    rq = (br / MU0) * rarea
    R = c["R_track"]
    rotor_poles = np.array([[R + rL / 2.0, 0.0], [R - rL / 2.0, 0.0]])
    rotor_q = np.array([rq, -rq])
    P = np.array([g.Rs - g.L / 2.0, 0.0])
    diff = P - rotor_poles
    d3 = (np.sum(diff * diff, axis=1)) ** 1.5
    return abs(float(np.sum(INV_4PI * rotor_q * diff[:, 0] / d3)))


def attraction_force(cfg, gap, samples=24):
    """Peak |force| [N] between rotor and nearest engaged stator at this gap."""
    c = dict(cfg); c["gap"] = gap
    g = Gate(c); g.r[:] = g.Rs
    R = c["R_track"]; worst = 0.0
    for a in np.linspace(0.0, 2.0 * math.pi / g.n, samples):
        pos = np.array([R * math.cos(a), R * math.sin(a)])
        F, _ = force_on_rotor(pos, a, c, g)
        worst = max(worst, float(np.hypot(F[0], F[1])))
    return worst


def demag_limit_gap(cfg, Hci):
    """Air gap [m] at which the reverse field just reaches Hci (stators demag)."""
    lo, hi = 0.001, 0.040
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if reverse_field(cfg, mid) > Hci:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class Frame:
    t: float = 0.0
    theta: float = 0.0
    omega: float = 0.0
    rpm: float = 0.0
    revolutions: int = 0
    tau_mag: float = 0.0
    force_mag: float = 0.0
    # work accumulators (joules)
    prop_rev: float = 0.0
    prop_last: float = 0.0
    prop_cum: float = 0.0
    switch_rev: float = 0.0
    switch_last: float = 0.0
    switch_cum: float = 0.0
    diss_rev: float = 0.0       # dissipated (friction/drag) work this revolution
    diss_last: float = 0.0
    diss_cum: float = 0.0
    net_rev: float = 0.0        # net work on rotor this revolution = delta KE
    net_last: float = 0.0
    net_cum: float = 0.0
    rev_time: float = 0.0       # duration of the last revolution [s]
    # instantaneous powers [W]
    p_mag: float = 0.0
    p_diss: float = 0.0
    p_net: float = 0.0
    p_gen: float = 0.0          # electrical power drawn by the generator load [W]
    # averages / derived
    avg_power: float = 0.0      # mean net power over the run [W]
    avg_rpm: float = 0.0
    wh: float = 0.0             # magnetic energy throughput [W*h]
    field_energy_J: float = 0.0
    field_energy_pct: float = 100.0
    ke: float = 0.0


class SimWorker(QObject):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.gate = Gate(self.cfg)
        self.time_scale = self.cfg["time_scale"]
        self._running = False
        self._paused = False
        self.latest = Frame()
        self._reset()
        self._wall_ref = time.perf_counter()
        self._sim_ref = 0.0
        self.dt = 1e-5

    def _reset(self):
        self.theta = 0.0
        self.omega = 0.0
        self.t = 0.0
        self.rev = 0
        self.prop_rev = self.prop_last = 0.0
        self.switch_rev = self.switch_last = 0.0
        self.prop_cum = self.switch_cum = 0.0
        self.diss_rev = self.diss_last = self.diss_cum = 0.0
        self.net_rev = self.net_last = self.net_cum = 0.0
        self._t_at_rev = 0.0
        self.rev_time = 0.0
        self._cum_rpm_dt = 0.0
        self.gate.br = self.gate.br0

    def go(self):
        self._reset()
        self._wall_ref = time.perf_counter(); self._sim_ref = 0.0
        self._running = True

    def stop(self): self._running = False

    def set_paused(self, p):
        self._paused = p
        self._wall_ref = time.perf_counter(); self._sim_ref = self.t

    def set_time_scale(self, s):
        self.time_scale = max(s, 1e-3)
        self._wall_ref = time.perf_counter(); self._sim_ref = self.t

    def reconfigure(self, cfg):
        keep = self.omega
        self.cfg = dict(cfg)
        self.gate.reconfigure(self.cfg)
        self._reset()
        self.omega = keep
        self._wall_ref = time.perf_counter(); self._sim_ref = 0.0

    def push(self, domega): self.omega += domega

    @property
    def inertia(self):
        r = self.cfg["R_track"]
        return self.cfg["rotor_mass"] * r * r + 3.0e-6

    def run(self):
        while self._running:
            if self._paused:
                QThread.msleep(20)
                self._wall_ref = time.perf_counter(); self._sim_ref = self.t
                continue
            cfg = self.cfg
            R = cfg["R_track"]
            rotor_ang = self.theta + (math.pi / 2.0 if cfg["base_orient"] == "tangential" else 0.0)
            pos = np.array([R * math.cos(self.theta), R * math.sin(self.theta)])
            self.gate._rotor_pos = pos
            self.gate._rotor_ang = rotor_ang
            sw_work = self.gate.update_switching(self.theta)
            F, tau = force_on_rotor(pos, rotor_ang, cfg, self.gate)

            J = self.inertia
            w = self.omega
            fric_t = cfg["friction"] * w + cfg["drag"] * w * abs(w) \
                + cfg["coulomb"] * (1.0 if w >= 0 else -1.0) + cfg.get("load", 0.0) * w
            net = tau - fric_t
            # instantaneous powers [W] (evaluated at the pre-step speed)
            p_mag = tau * w
            p_diss = cfg["friction"] * w * w + cfg["drag"] * abs(w) ** 3 + cfg["coulomb"] * abs(w) \
                + cfg.get("load", 0.0) * w * w
            p_net = p_mag - p_diss
            p_gen = cfg.get("load", 0.0) * w * w
            self.omega += (net / J) * self.dt
            dtheta = self.omega * self.dt
            self.theta += dtheta
            self.t += self.dt
            rpm_now = abs(self.omega) * 60.0 / (2.0 * math.pi)
            self._cum_rpm_dt += rpm_now * self.dt

            # work accumulators [J]
            mag = tau * dtheta
            diss = p_diss * self.dt
            nwork = p_net * self.dt                 # = d(KE)
            self.prop_rev += mag;   self.prop_cum += mag
            self.diss_rev += diss;  self.diss_cum += diss
            self.net_rev += nwork;  self.net_cum += nwork
            self.switch_rev += sw_work; self.switch_cum += sw_work

            # completed laps by net angular displacement magnitude -- counts forward or
            # backward travel, but never double-counts back-and-forth oscillation.
            new_rev = int(abs(self.theta) / (2.0 * math.pi))
            if new_rev > self.rev:
                self.prop_last = self.prop_rev
                self.diss_last = self.diss_rev
                self.net_last = self.net_rev
                self.switch_last = self.switch_rev
                self.rev_time = self.t - self._t_at_rev
                self._t_at_rev = self.t
                self.prop_rev = self.diss_rev = self.net_rev = self.switch_rev = 0.0
                self.rev = new_rev

            # demagnetization: Br decays; switching/propulsion could draw it down
            tau_demag = cfg["demag_tau"]
            if tau_demag > 0:
                self.gate.br *= math.exp(-self.dt / tau_demag)
            fe = self.gate.field_energy()
            fe0 = (self.gate.br0 ** 2) / (2.0 * MU0) * self.gate.volume_total

            self.latest = Frame(
                t=self.t, theta=self.theta % (2.0 * math.pi), omega=self.omega,
                rpm=rpm_now, revolutions=self.rev,
                tau_mag=tau, force_mag=float(np.hypot(F[0], F[1])),
                prop_rev=self.prop_rev, prop_last=self.prop_last, prop_cum=self.prop_cum,
                switch_rev=self.switch_rev, switch_last=self.switch_last, switch_cum=self.switch_cum,
                diss_rev=self.diss_rev, diss_last=self.diss_last, diss_cum=self.diss_cum,
                net_rev=self.net_rev, net_last=self.net_last, net_cum=self.net_cum,
                rev_time=self.rev_time,
                p_mag=p_mag, p_diss=p_diss, p_net=p_net, p_gen=p_gen,
                avg_power=(self.net_cum / self.t) if self.t > 1e-9 else 0.0,
                avg_rpm=(self._cum_rpm_dt / self.t) if self.t > 1e-9 else 0.0,
                wh=self.prop_cum / 3600.0,
                field_energy_J=fe, field_energy_pct=100.0 * fe / max(fe0, 1e-12),
                ke=0.5 * J * self.omega * self.omega,
            )

            target = self._sim_ref + self.time_scale * (time.perf_counter() - self._wall_ref)
            if self.t > target:
                QThread.msleep(min(int((self.t - target) / self.time_scale * 1000), 40))


# --------------------------------------------------------------------------- canvas
def _rectf(x, y, w, h): return QRectF(x, y, w, h)


class TrackCanvas(QWidget):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.setMinimumSize(540, 540)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vis_theta = 0.0
        self.show_field = False
        self._field = []

    def set_field(self, on):
        self.show_field = on
        self._field = self._compute_field() if on else []
        self.update()

    def recompute_field(self):
        if self.show_field:
            self._field = self._compute_field(); self.update()

    def _compute_field(self):
        g = self.worker.gate
        q, pos = g.poles()
        extent = g.Rs + 0.06
        lines = []
        for si in np.where(q > 0)[0][:20]:
            x, y = pos[si]
            line = [(float(x), float(y))]; px, py = x, y
            for _ in range(200):
                dx = px - pos[:, 0]; dy = py - pos[:, 1]
                d2 = dx * dx + dy * dy + 1e-6
                d3 = d2 ** 1.5
                bx = np.sum(MU0_4PI * q * dx / d3); by = np.sum(MU0_4PI * q * dy / d3)
                bn = math.hypot(bx, by)
                if bn < 1e-9: break
                step = 0.0011
                px += step * bx / bn; py += step * by / bn
                if abs(px) > extent or abs(py) > extent: break
                line.append((px, py))
            if len(line) > 3: lines.append(line)
        return lines

    def paintEvent(self, _ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        g = self.worker.gate
        scale = min(w, h) / (2.0 * (g.Rs + 0.06))
        p.fillRect(self.rect(), QColor(16, 18, 24))

        def ts(x, y): return cx + x * scale, cy - y * scale

        R = self.worker.cfg["R_track"]
        p.setPen(QPen(QColor(70, 76, 90), 2, Qt.DashLine)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), R * scale, R * scale)
        p.setPen(QPen(QColor(60, 64, 74), 1)); p.setBrush(QBrush(QColor(40, 44, 52)))
        p.drawEllipse(QPointF(cx, cy), 8, 8)

        if self.show_field:
            p.setPen(QPen(QColor(90, 160, 230, 120), 1))
            for line in self._field:
                p.drawPolyline(QPolygonF([QPointF(*ts(x, y)) for x, y in line]))

        sL, sW = self.worker.cfg["mag_len"], self.worker.cfg["mag_wid"]
        for i in range(g.n):
            mx = g.r[i] * math.cos(g.alpha[i]); my = g.r[i] * math.sin(g.alpha[i])
            self._draw_mag(p, ts, mx, my, g.ang[i], sL, sW, scale)
        if self.worker.cfg.get("flux_gate") and getattr(g, "aux_n", 0) > 0:
            for i in range(g.aux_n):
                mx = g.aux_R * math.cos(g.aux_alpha[i]); my = g.aux_R * math.sin(g.aux_alpha[i])
                self._draw_mag(p, ts, mx, my, g.aux_ang[i], g.aux_L, g.aux_W, scale, aux=True)
        rL, rW = rotor_dims(self.worker.cfg)
        ang = self.vis_theta + (math.pi / 2.0 if self.worker.cfg["base_orient"] == "tangential" else 0.0)
        rx = R * math.cos(self.vis_theta); ry = R * math.sin(self.vis_theta)
        self._draw_mag(p, ts, rx, ry, ang, rL, rW, scale, rotor=True)

        f = self.worker.latest
        dirn = "↻" if f.omega >= 0 else "↺"      # CW / CCW
        p.setPen(QColor(220, 226, 238)); p.setFont(QFont("DejaVu Sans", 10, QFont.Bold))
        p.drawText(12, 20, f"t = {f.t:7.2f} s    rev = {f.revolutions}    RPM = {f.rpm:6.1f} {dirn}    "
                            f"{'REGAUGING' if self.worker.cfg['switching'] else 'passive'}"
                            f"{'+FLUX-GATE' if self.worker.cfg.get('flux_gate') else ''}    "
                            f"gate: {self.worker.cfg['pattern']}")
        p.setPen(QColor(180, 230, 180))
        p.drawText(12, 38, f"propulsion/rev = {f.prop_last*1e3:+8.4f} mJ   "
                            f"switching/rev = {f.switch_last*1e3:+8.4f} mJ   "
                            f"field energy = {f.field_energy_pct:5.1f}%")

        # --- real-world scale: scale bar + dimension summary (drawing is to scale)
        ppm = scale
        cfg = self.worker.cfg
        dia_mm = 2.0 * cfg["R_track"] * 1e3
        sL, sW = cfg["mag_len"], cfg["mag_wid"]
        target_m = (w * 0.18) / ppm
        m_ = 10.0 ** math.floor(math.log10(target_m)); nrm = target_m / m_
        nice = m_ * (1 if nrm < 1.5 else 2 if nrm < 3.5 else 5 if nrm < 7.5 else 10)
        bpx = nice * ppm; by = h - 14; bx = 14
        p.setFont(QFont("DejaVu Sans", 8))
        p.setPen(QPen(QColor(210, 218, 230), 2))
        p.drawLine(bx, by, bx + bpx, by)
        p.drawLine(bx, by - 5, bx, by + 5); p.drawLine(bx + bpx, by - 5, bx + bpx, by + 5)
        p.setPen(QColor(200, 208, 222))
        p.drawText(bx, by - 7, f"{nice * 1e3:.0f} mm  (to scale)")
        dim = (f"track Ø {dia_mm:.0f} mm   gap {cfg['gap']*1e3:.1f} mm   "
               f"stator {sL*1e3:.1f}x{sW*1e3:.1f} mm   {cfg['n_stator']} stators")
        p.drawText(w - 12 - p.fontMetrics().horizontalAdvance(dim), h - 10, dim)
        p.end()

    def _draw_mag(self, p, ts, mx, my, ang, L, W, scale, rotor=False, aux=False):
        sx, sy = ts(mx, my); halfL = (L / 2.0) * scale; halfW = (W / 2.0) * scale
        p.save(); p.translate(sx, sy); p.rotate(-math.degrees(ang))
        p.setPen(QPen(QColor(20, 20, 24), 1))
        if aux:
            p.setBrush(QBrush(QColor(60, 60, 68)))
            p.drawRoundedRect(_rectf(-halfW, -halfL, halfW * 2, halfL), 2, 2)
            p.setBrush(QBrush(QColor(34, 34, 40)))
            p.drawRoundedRect(_rectf(-halfW, 0, halfW * 2, halfL), 2, 2)
        else:
            p.setBrush(QBrush(QColor(214, 70, 70) if not rotor else QColor(255, 120, 100)))
            p.drawRoundedRect(_rectf(-halfW, -halfL, halfW * 2, halfL), 2, 2)
            p.setBrush(QBrush(QColor(70, 120, 214) if not rotor else QColor(120, 170, 255)))
            p.drawRoundedRect(_rectf(-halfW, 0, halfW * 2, halfL), 2, 2)
        if rotor:
            p.setPen(QPen(QColor(255, 240, 180), 1))
            p.drawEllipse(_rectf(-halfW - 2, -halfL - 2, halfW * 2 + 4, halfL * 2 + 4))
        p.restore()


# --------------------------------------------------------------------------- dashboard / graphs / controls
HP = 745.7   # watts per mechanical horsepower


class Calculators(QGroupBox):
    """Live work / energy / power calculators (per-revolution, cumulative, instantaneous)."""

    def __init__(self):
        super().__init__("Work & energy calculators")
        self._labels = {}
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)

        def header(row, text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#9cf;font-weight:bold;border:none;")
            grid.addWidget(lbl, row, 0, 1, 3)

        def row(r, key, title, fmt, unit):
            t = QLabel(title); t.setStyleSheet("color:#aab;")
            v = QLabel("—"); v.setStyleSheet("color:#eee;font-weight:bold;")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            u = QLabel(unit); u.setStyleSheet("color:#889;")
            grid.addWidget(t, r, 0); grid.addWidget(v, r, 1); grid.addWidget(u, r, 2)
            self._labels[key] = (v, fmt)

        r = 0
        header(r, "PER REVOLUTION (resets each turn)"); r += 1
        row(r, "prop_rev", "magnetic work", "{:+.2f}", "mJ"); r += 1
        row(r, "sw_rev", "switching work", "{:+.2f}", "mJ"); r += 1
        row(r, "diss_rev", "dissipated (loss)", "{:.2f}", "mJ"); r += 1
        row(r, "net_rev", "net work on rotor (ΔKE)", "{:+.2f}", "mJ"); r += 1
        row(r, "revt", "revolution time", "{:.3f}", "s"); r += 1
        r += 1
        header(r, "CUMULATIVE (since reset)"); r += 1
        row(r, "prop_cum", "total magnetic work", "{:.3f}", "J"); r += 1
        row(r, "prop_wh", "  in watt-hours", "{:.4f}", "W·h"); r += 1
        row(r, "sw_cum", "total switching work", "{:.3f}", "J"); r += 1
        row(r, "diss_cum", "total dissipated", "{:.3f}", "J"); r += 1
        row(r, "net_cum", "net energy on rotor", "{:+.3f}", "J"); r += 1
        row(r, "avgp", "average net power", "{:.4f}", "W"); r += 1
        row(r, "avgphp", "  in horsepower", "{:.5f}", "hp"); r += 1
        row(r, "avgr", "average RPM", "{:.1f}", ""); r += 1
        r += 1
        header(r, "INSTANTANEOUS"); r += 1
        row(r, "pmag", "magnetic power", "{:+.4f}", "W"); r += 1
        row(r, "pmaghp", "  in horsepower", "{:+.6f}", "hp"); r += 1
        row(r, "pdiss", "dissipated power", "{:.4f}", "W"); r += 1
        row(r, "pnet", "net power (dKE/dt)", "{:+.4f}", "W"); r += 1
        row(r, "pgen", "generator load out", "{:.4f}", "W"); r += 1
        row(r, "tau", "torque", "{:+.3f}", "mN·m"); r += 1
        row(r, "rpm", "RPM", "{:.1f}", ""); r += 1
        row(r, "ke", "kinetic energy", "{:.2f}", "mJ"); r += 1
        row(r, "fe", "magnet field energy", "{:.2f}  ({:.1f}%)", "J"); r += 1

        self.setLayout(grid)
        self.setStyleSheet("QGroupBox{color:#cde;font-weight:bold;border:1px solid #334;"
                           "border-radius:6px;margin-top:10px;padding:8px;} QLabel{color:#bbc;font:9pt 'DejaVu Sans';}")

    def set_values(self, f):
        v = {
            "prop_rev": f.prop_last * 1e3, "sw_rev": f.switch_last * 1e3,
            "diss_rev": f.diss_last * 1e3, "net_rev": f.net_last * 1e3, "revt": f.rev_time,
            "prop_cum": f.prop_cum, "prop_wh": f.wh, "sw_cum": f.switch_cum,
            "diss_cum": f.diss_cum, "net_cum": f.net_cum,
            "avgp": f.avg_power, "avgphp": f.avg_power / HP, "avgr": f.avg_rpm,
            "pmag": f.p_mag, "pmaghp": f.p_mag / HP, "pdiss": f.p_diss, "pnet": f.p_net,
            "pgen": f.p_gen,
            "tau": f.tau_mag * 1e3, "rpm": f.rpm, "ke": f.ke * 1e3,
            "fe": (f.field_energy_J, f.field_energy_pct),
        }
        for k, (lbl, fmt) in self._labels.items():
            val = v[k]
            lbl.setText(fmt.format(*val) if isinstance(val, tuple) else fmt.format(val))


class EngineeringReadouts(QGroupBox):
    """Live engineering readouts: headroom limits, geometry/fields, and motion.
    Static rows recompute only when the config changes; motion rows update each tick."""

    def __init__(self):
        super().__init__("Engineering & headroom readouts")
        self._labels = {}
        grid = QGridLayout(); grid.setColumnStretch(1, 1)

        def header(row, text):
            lbl = QLabel(text); lbl.setStyleSheet("color:#9cf;font-weight:bold;border:none;")
            grid.addWidget(lbl, row, 0, 1, 3)

        def row(r, key, title, fmt, unit):
            t = QLabel(title); t.setStyleSheet("color:#aab;")
            v = QLabel("—"); v.setStyleSheet("color:#eee;font-weight:bold;")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            u = QLabel(unit); u.setStyleSheet("color:#889;")
            grid.addWidget(t, r, 0); grid.addWidget(v, r, 1); grid.addWidget(u, r, 2)
            self._labels[key] = (v, fmt)

        r = 0
        header(r, "HEADROOM  (where it gives way)"); r += 1
        row(r, "demagH", "demag reverse field", "{:.0f}", "kA/m"); r += 1
        row(r, "demagM", "vs Hci (stator)", "{:.0f}%", ""); r += 1
        row(r, "demagG", "demag-limit gap", "{:.2f}", "mm"); r += 1
        row(r, "force", "peak force / stator", "{:.1f}", "N"); r += 1
        r += 1
        header(r, "GEOMETRY & FIELDS"); r += 1
        row(r, "circ", "track circumference", "{:.0f}", "mm"); r += 1
        row(r, "pitch", "tooth pitch", "{:.1f} deg / {:.1f} mm", ""); r += 1
        row(r, "q", "pole charge q", "{:.2f}", "A*m"); r += 1
        row(r, "fe", "field energy (total)", "{:.3f}  ({:.1f}%)", "J"); r += 1
        row(r, "Jin", "rotor inertia", "{:.2e}", "kg*m^2"); r += 1
        r += 1
        header(r, "MOTION  (live)"); r += 1
        row(r, "vsurf", "surface speed", "{:.2f}", "m/s"); r += 1
        row(r, "acent", "centripetal accel", "{:.0f}", "g"); r += 1
        row(r, "grate", "gate pass rate", "{:.0f}", "Hz"); r += 1
        row(r, "Lmom", "angular momentum", "{:.4f}", "kg*m^2/s"); r += 1

        self.setLayout(grid)
        self.setStyleSheet("QGroupBox{color:#cde;font-weight:bold;border:1px solid #334;"
                           "border-radius:6px;margin-top:10px;padding:8px;} QLabel{color:#bbc;font:9pt 'DejaVu Sans';}")
        self._sig = None

    def _set(self, key, val):
        lbl, fmt = self._labels[key]
        lbl.setText(fmt.format(*val) if isinstance(val, tuple) else fmt.format(val))

    def update_static(self, worker):
        cfg = worker.cfg; g = worker.gate
        sig = (cfg["grade_rotor"], cfg["grade_stator"], cfg["gap"], cfg["mag_len"],
               cfg["mag_wid"], cfg.get("mag_len_rotor"), cfg.get("mag_wid_rotor"),
               cfg["R_track"], cfg["n_stator"], cfg["pattern"])
        if sig == self._sig:
            return
        self._sig = sig
        H = reverse_field(cfg, cfg["gap"])
        Hci = HCI.get(cfg["grade_stator"], 7.5e5)
        self._set("demagH", H / 1e3)
        self._set("demagM", H / Hci * 100.0)
        self._set("demagG", demag_limit_gap(cfg, Hci) * 1e3)
        self._set("force", attraction_force(cfg, cfg["gap"]))
        self._set("circ", 2.0 * math.pi * cfg["R_track"] * 1e3)
        pdeg = math.degrees(2.0 * math.pi / g.n)
        self._set("pitch", (pdeg, math.radians(pdeg) * cfg["R_track"] * 1e3))
        self._set("q", (GRADES[cfg["grade_stator"]] / MU0) * cfg["mag_wid"] ** 2)
        self._set("Jin", worker.inertia)

    def set_values(self, f, worker):
        self.update_static(worker)                       # no-op unless config changed
        g = worker.gate
        fe = g.field_energy()
        fe0 = (g.br0 ** 2) / (2.0 * MU0) * g.volume_total
        self._set("fe", (fe, 100.0 * fe / max(fe0, 1e-12)))
        w = abs(f.omega); R = worker.cfg["R_track"]
        self._set("vsurf", w * R)
        self._set("acent", w * w * R / 9.81)
        self._set("grate", g.n * w / (2.0 * math.pi))
        self._set("Lmom", worker.inertia * f.omega)

    def snapshot(self, worker):
        cfg = worker.cfg; g = worker.gate; f = worker.latest
        H = reverse_field(cfg, cfg["gap"]); Hci = HCI.get(cfg["grade_stator"], 7.5e5)
        fe = g.field_energy(); fe0 = (g.br0 ** 2) / (2.0 * MU0) * g.volume_total
        w = abs(f.omega); R = cfg["R_track"]
        pdeg = math.degrees(2.0 * math.pi / g.n)
        return {
            "demag_reverse_field_kAm": H / 1e3,
            "demag_margin_pct": H / Hci * 100.0,
            "demag_limit_gap_mm": demag_limit_gap(cfg, Hci) * 1e3,
            "peak_force_per_stator_N": attraction_force(cfg, cfg["gap"]),
            "track_circumference_mm": 2.0 * math.pi * R * 1e3,
            "tooth_pitch_deg": pdeg, "tooth_pitch_mm": math.radians(pdeg) * R * 1e3,
            "pole_charge_q_Am": (GRADES[cfg["grade_stator"]] / MU0) * cfg["mag_wid"] ** 2,
            "field_energy_J": fe, "field_energy_pct": 100.0 * fe / max(fe0, 1e-12),
            "rotor_inertia_kgm2": worker.inertia,
            "surface_speed_ms": w * R, "centripetal_g": w * w * R / 9.81,
            "gate_pass_rate_Hz": g.n * w / (2.0 * math.pi),
            "angular_momentum_kgm2s": worker.inertia * f.omega,
        }


class RollingGraph(QWidget):
    def __init__(self, title, series, maxlen=260):
        super().__init__(); self.title = title; self.series = series
        self.data = {n: deque(maxlen=maxlen) for n, _ in series}; self.setMinimumHeight(130)

    def append(self, m):
        for n, _ in self.series: self.data[n].append(m.get(n, 0.0))
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height(); p.fillRect(self.rect(), QColor(20, 22, 28))
        p.setPen(QColor(170, 176, 188)); p.setFont(QFont("DejaVu Sans", 9, QFont.Bold))
        p.drawText(8, 14, self.title)
        ml, mb, mt = 44, 18, 20; pw, ph = w - ml - 8, h - mb - mt
        if pw <= 0 or ph <= 0: p.end(); return
        allv = [v for n, _ in self.series for v in self.data[n]]
        if not allv: p.end(); return
        lo, hi = min(allv), max(allv)
        if hi - lo < 1e-12: hi = lo + 1.0
        lo -= 0.1 * (hi - lo); hi += 0.1 * (hi - lo)
        p.setPen(QPen(QColor(48, 52, 60), 1, Qt.DotLine)); p.setFont(QFont("DejaVu Sans", 7))
        for frac in (0.0, 0.5, 1.0):
            yy = mt + ph * (1 - frac); p.drawLine(ml, yy, ml + pw, yy)
            p.setPen(QColor(110, 116, 128)); p.drawText(2, yy + 8, f"{lo+(hi-lo)*frac:.2g}")
            p.setPen(QPen(QColor(48, 52, 60), 1, Qt.DotLine))
        if 0 < hi and 0 > lo:
            zy = mt + ph * (1 - (0 - lo) / (hi - lo))
            p.setPen(QPen(QColor(120, 80, 80), 1, Qt.DotLine)); p.drawLine(ml, zy, ml + pw, zy)
        for name, color in self.series:
            d = self.data[name]
            if len(d) < 2: continue
            p.setPen(QPen(QColor(*color), 2)); n = len(d); prev = None
            for i, v in enumerate(d):
                x = ml + pw * i / (n - 1); y = mt + ph * (1 - (v - lo) / (hi - lo))
                if prev is not None: p.drawLine(prev, QPointF(x, y))
                prev = QPointF(x, y)
        p.end()


class ControlPanel(QGroupBox):
    changed = Signal(dict); speed_changed = Signal(float); pushed = Signal(float)
    export_stats = Signal(); export_preset = Signal(); import_preset = Signal()
    scaled = Signal(float)
    SPEED_LO, SPEED_HI, SPEED_DEFAULT = 0.01, 8.0, 0.2   # continuous log range + launch default

    def __init__(self):
        super().__init__("Controls"); self._state = dict(REF)
        grid = QGridLayout(); self._combo = {}

        def combo(row, label, key, items, default):
            grid.addWidget(QLabel(label), row, 0)
            cb = QComboBox(); cb.addItems(items); cb.setCurrentText(default)
            cb.currentTextChanged.connect(lambda v, k=key: self._on(k, v))
            grid.addWidget(cb, row, 1, 1, 2); self._combo[key] = cb

        combo(0, "Rotor magnet", "grade_rotor", list(GRADES), REF["grade_rotor"])
        combo(1, "Stator magnet", "grade_stator", list(GRADES), REF["grade_stator"])
        combo(2, "Gate pattern", "pattern", ["uniform", "alternating", "paired", "johnson_skew"], REF["pattern"])
        combo(3, "Magnet face", "base_orient", ["radial", "tangential"], REF["base_orient"])
        self._slider = {}; self._meta = {}

        def slider(row, label, key, lo, hi, init, fmt, isint=False, scale=1.0):
            grid.addWidget(QLabel(label), row, 0)
            sl = QSlider(Qt.Horizontal); sl.setMinimum(0); sl.setMaximum(1000)
            sl.setValue(int(1000 * (init - lo) / (hi - lo)))
            disp = QLabel()
            def onv(v, k=key, lo=lo, hi=hi, d=disp, f=fmt, ii=isint, sc=scale):
                val = lo + (hi - lo) * v / 1000
                if ii: val = int(round(val))
                d.setText(f.format(val * sc)); self._on(k, val)
            sl.valueChanged.connect(onv); grid.addWidget(sl, row, 1); grid.addWidget(disp, row, 2)
            self._slider[key] = sl
            self._meta[key] = dict(lo=lo, hi=hi, isint=isint, fmt=fmt, scale=scale, disp=disp)
            disp.setText(fmt.format(init * scale))

        def logslider(row, label, key, lo, hi, init, fmt, scale=1.0):
            grid.addWidget(QLabel(label), row, 0)
            sl = QSlider(Qt.Horizontal); sl.setMinimum(0); sl.setMaximum(1000)
            llo, lhi = math.log10(lo), math.log10(hi)
            sl.setValue(int(1000 * (math.log10(init) - llo) / (lhi - llo)))
            disp = QLabel()
            def onv(v, k=key, llo=llo, lhi=lhi, d=disp, f=fmt, sc=scale):
                val = 10.0 ** (llo + (lhi - llo) * v / 1000.0)
                d.setText(f.format(val * sc)); self._on(k, val)
            sl.valueChanged.connect(onv); grid.addWidget(sl, row, 1); grid.addWidget(disp, row, 2)
            self._slider[key] = sl
            self._meta[key] = dict(lo=lo, hi=hi, isint=False, fmt=fmt, scale=scale, disp=disp, log=True)
            disp.setText(fmt.format(init * scale))

        slider(4, "Stator count", "n_stator", 4, 36, REF["n_stator"], "{:.0f}", isint=True)
        slider(5, "Air gap", "gap", 0.003, 0.05, REF["gap"], "{:.1f} mm", scale=1000.0)
        slider(6, "Skew / magnet", "skew_deg", -20, 20, REF["skew_deg"], "{:+.1f}°")
        slider(7, "Engage arc", "engage_arc_deg", 4, 60, REF["engage_arc_deg"], "{:.0f}°")
        slider(8, "Retract distance", "retract_mm", 5, 60, REF["retract_mm"], "{:.0f} mm")
        slider(9, "Friction", "friction", 0.0, 1.0e-3, REF["friction"], "{:.1f} µ", scale=1.0e6)
        slider(10, "Rotor mass", "rotor_mass", 0.005, 0.20, REF["rotor_mass"], "{:.0f} g", scale=1000.0)
        slider(11, "Track radius", "R_track", 0.04, 0.20, REF["R_track"], "{:.0f} mm", scale=1000.0)
        logslider(12, "Demag timescale", "demag_tau", 1.0e2, 1.0e7, REF["demag_tau"], "{:.2f} h", scale=1.0/3600.0)

        self.switch_chk = QCheckBox("Regauging (auto flux-switching) -- loops continuously")
        self.switch_chk.setChecked(REF["switching"])
        self.switch_chk.stateChanged.connect(lambda s: self._on("switching", bool(s)))
        grid.addWidget(self.switch_chk, 13, 0, 1, 3)

        grid.addWidget(QLabel("Sim speed (slow-mo 0.01x <-> 8x)"), 14, 0)
        ssl = QSlider(Qt.Horizontal); ssl.setMinimum(0); ssl.setMaximum(1000)
        _llo, _lhi = math.log10(self.SPEED_LO), math.log10(self.SPEED_HI)
        ssl.setValue(int(1000 * (math.log10(self.SPEED_DEFAULT) - _llo) / (_lhi - _llo)))
        sd = QLabel(f"{self.SPEED_DEFAULT:g}x")
        def _on_speed(v, llo=_llo, lhi=_lhi, d=sd):
            s = 10.0 ** (llo + (lhi - llo) * v / 1000.0)
            d.setText(f"{s:g}x" if s >= 0.1 else f"{s:.3f}x")
            self.speed_changed.emit(s)
        ssl.valueChanged.connect(_on_speed)
        grid.addWidget(ssl, 14, 1); grid.addWidget(sd, 14, 2)
        self.speed_slider = ssl; self.speed_disp = sd

        pb_ccw = QPushButton("Push ⟲"); pb_cw = QPushButton("Push ⟳")
        pb_ccw.clicked.connect(lambda: self.pushed.emit(-30.0))
        pb_cw.clicked.connect(lambda: self.pushed.emit(30.0))
        grid.addWidget(pb_ccw, 15, 0); grid.addWidget(pb_cw, 15, 1)
        self.pause_btn = QPushButton("Pause"); self.reset_btn = QPushButton("Reset")
        self.field_chk = QCheckBox("Show stator field lines")
        grid.addWidget(self.pause_btn, 15, 2)
        grid.addWidget(self.field_chk, 16, 0, 1, 2); grid.addWidget(self.reset_btn, 16, 2)
        self.johnson_btn = QPushButton("Load Johnson geometry (patent 4,151,431)")
        self.johnson_btn.setToolTip("SmCo, ~4.5 mm gap, deliberate stagger/skew, regauging on.\n"
                                    "Loads his documented build figures; reads the real computed work/rev.")
        self.johnson_btn.clicked.connect(lambda: self.apply_preset(dict(JOHNSON_PRESET)))
        grid.addWidget(self.johnson_btn, 17, 0, 1, 3)

        self.export_stats_btn = QPushButton("Export stats")
        self.export_preset_btn = QPushButton("Export preset")
        self.import_preset_btn = QPushButton("Import preset")
        self.export_stats_btn.clicked.connect(self.export_stats)
        self.export_preset_btn.clicked.connect(self.export_preset)
        self.import_preset_btn.clicked.connect(self.import_preset)
        grid.addWidget(self.export_stats_btn, 18, 0)
        grid.addWidget(self.export_preset_btn, 18, 1)
        grid.addWidget(self.import_preset_btn, 18, 2)

        grid.addWidget(QLabel("Device scale (size)"), 19, 0)
        dscl = QSlider(Qt.Horizontal); dscl.setMinimum(0); dscl.setMaximum(1000)
        _cllo, _clhi = math.log10(0.25), math.log10(4.0)
        dscl.setValue(int(1000 * (0.0 - _cllo) / (_clhi - _cllo)))
        dsd = QLabel("1.00x")
        def _on_dscale(v, llo=_cllo, lhi=_clhi, d=dsd):
            s = 10.0 ** (llo + (lhi - llo) * v / 1000.0)
            d.setText(f"{s:.2f}x"); self.device_scale = s; self.scaled.emit(s)
        dscl.valueChanged.connect(_on_dscale)
        grid.addWidget(dscl, 19, 1); grid.addWidget(dsd, 19, 2)
        self.device_scale = 1.0

        self.flux_chk = QCheckBox("Flux-gate (null-zone aux magnets)")
        self.flux_chk.setChecked(REF["flux_gate"])
        self.flux_chk.setToolTip("Johnson's small weaker (rubber/ferrite) magnets placed at\n"
                                 "the null zones to warp the static field -- a passive flux\n"
                                 "gate, no actuator. Combine with regauging off for the pure\n"
                                 "passive case the patents describe.")
        self.flux_chk.stateChanged.connect(lambda s: self._on("flux_gate", bool(s)))
        grid.addWidget(self.flux_chk, 20, 0, 1, 3)
        slider(21, "Aux strength", "aux_br", 0.0, 1.2, REF["aux_br"], "{:.2f} T")
        slider(22, "Aux position", "aux_offset_deg", -30, 30, REF["aux_offset_deg"], "{:+.1f}°")
        slider(23, "Generator load", "load", 0.0, 5.0e-4, REF["load"], "{:.1f} µ", scale=1.0e6)
        self.glide_btn = QPushButton("Load flux-gate (tuned glide)")
        self.glide_btn.setToolTip("Passive flux-gate profile from fluxgate_sweep.py:\n"
                                  "aux 0.50 T at -20 deg + 8 mm gap flattens the cogging\n"
                                  "barrier 73% so the rotor glides instead of grabbing.")
        self.glide_btn.clicked.connect(lambda: self.apply_preset(dict(JOHNSON_GLIDE_PRESET)))
        grid.addWidget(self.glide_btn, 24, 0, 1, 3)
        self.setLayout(grid)
        self.setStyleSheet("""
            QGroupBox { color:#cde; font-weight:bold; border:1px solid #334;
                        border-radius:6px; margin-top:10px; padding:8px; }
            QLabel    { color:#bbc; }
            QCheckBox { color:#dde; spacing:6px; }
            QPushButton {
                background-color:#2b3344; color:#eef2f8;
                border:1px solid #4b5668; border-radius:4px;
                padding:6px 10px; font-weight:bold;
            }
            QPushButton:hover    { background-color:#3a4558; border:1px solid #6b7688; }
            QPushButton:pressed  { background-color:#1d2430; }
            QPushButton:disabled { color:#8899aa; background-color:#232834; }
        """)

    def _on(self, key, val):
        self._state[key] = val; self.changed.emit(dict(self._state))

    def _set_slider(self, key, val):
        m = self._meta[key]; sl = self._slider[key]
        if m.get("log"):
            llo, lhi = math.log10(m["lo"]), math.log10(m["hi"])
            pos = int(round(1000 * (math.log10(val) - llo) / (lhi - llo)))
        else:
            pos = int(round(1000 * (val - m["lo"]) / (m["hi"] - m["lo"])))
        sl.blockSignals(True); sl.setValue(pos); sl.blockSignals(False)
        m["disp"].setText(m["fmt"].format(val * m["scale"])); self._state[key] = val

    def apply_preset(self, state):
        """Bulk-apply a geometry preset (combos + sliders) and emit one reconfigure."""
        for cb in self._combo.values():
            cb.blockSignals(True)
        self.switch_chk.blockSignals(True)
        self.flux_chk.blockSignals(True)
        try:
            for k, v in state.items():
                if k == "switching":
                    self.switch_chk.setChecked(bool(v)); self._state[k] = bool(v)
                elif k == "flux_gate":
                    self.flux_chk.setChecked(bool(v)); self._state[k] = bool(v)
                elif k in self._combo:
                    self._combo[k].setCurrentText(str(v)); self._state[k] = v
                elif k in self._meta:
                    self._set_slider(k, float(v))
                else:
                    self._state[k] = v
        finally:
            for cb in self._combo.values():
                cb.blockSignals(False)
            self.switch_chk.blockSignals(False)
            self.flux_chk.blockSignals(False)
        self.changed.emit(dict(self._state))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Howard Johnson magnetic gate -- continuous circular motor")
        self.resize(1400, 900); self.setStyleSheet("background-color:#121419;")
        self.worker = SimWorker(REF)
        self.thread = QThread(); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.canvas = TrackCanvas(self.worker)
        self.calc = Calculators()
        self.eng = EngineeringReadouts()
        self.graph_p = RollingGraph("Power [W]",
                                    [("P magnetic", (130, 210, 130)),
                                     ("P dissipated", (230, 170, 90)),
                                     ("P net", (120, 170, 250))])
        self.graph_w = RollingGraph("Work per revolution [mJ]",
                                    [("propulsion", (130, 210, 130)), ("switching", (230, 170, 90))])
        self.controls = ControlPanel()
        self.controls.changed.connect(self._reconfigure)
        self.controls.speed_changed.connect(self._on_speed_scale)
        self.controls.pushed.connect(self._push)
        self.controls.pause_btn.clicked.connect(self._toggle_pause)
        self.controls.reset_btn.clicked.connect(self._reset)
        self.controls.field_chk.toggled.connect(self.canvas.set_field)
        self.controls.export_stats.connect(self._export_stats)
        self.controls.export_preset.connect(self._export_preset)
        self.controls.import_preset.connect(self._import_preset)
        self.controls.scaled.connect(self._on_scale)
        self.worker.set_time_scale(self.controls.SPEED_DEFAULT)   # launch at viewable slow-mo
        self._scale = 1.0
        self._paused = False; self._last_rev = 0

        ctrl_scroll = QScrollArea(); ctrl_scroll.setWidgetResizable(True); ctrl_scroll.setWidget(self.controls)
        left = QVBoxLayout(); left.addWidget(self.canvas, stretch=3); left.addWidget(ctrl_scroll, stretch=2)
        readouts = QWidget(); rlv = QVBoxLayout(readouts); rlv.setContentsMargins(0, 0, 0, 0)
        rlv.addWidget(self.calc); rlv.addWidget(self.eng)
        rd_scroll = QScrollArea(); rd_scroll.setWidgetResizable(True); rd_scroll.setWidget(readouts)
        right = QVBoxLayout(); right.addWidget(rd_scroll, stretch=2)
        right.addWidget(self.graph_p, stretch=1); right.addWidget(self.graph_w, stretch=1)
        central = QWidget(); h = QHBoxLayout(central); h.addLayout(left, stretch=5)
        line = QFrame(); line.setFrameShape(QFrame.VLine); line.setStyleSheet("color:#334;")
        h.addWidget(line); h.addLayout(right, stretch=4); self.setCentralWidget(central)
        self.eng.update_static(self.worker)

        self._clock = QElapsedTimer(); self._clock.start(); self._last_ns = self._clock.nsecsElapsed()
        self.timer = QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self._tick)
        self.timer.start(); self.worker.go(); self.thread.start()
        # auto-start: one push and the gate takes over (regauging sustains, or flux-gate glides)
        QTimer.singleShot(300, lambda: self.worker.push(30.0))

    def _tick(self):
        now = self._clock.nsecsElapsed(); dt = (now - self._last_ns) * 1e-9; self._last_ns = now
        f = self.worker.latest; ts = self.worker.time_scale
        if not self._paused:
            self.canvas.vis_theta = (self.canvas.vis_theta + f.omega * dt * ts) % (2.0 * math.pi)
            self.calc.set_values(f)
            self.eng.set_values(f, self.worker)
            self.graph_p.append({"P magnetic": f.p_mag, "P dissipated": f.p_diss, "P net": f.p_net})
            if f.revolutions != self._last_rev:
                self.graph_w.append({"propulsion": f.prop_last * 1e3, "switching": f.switch_last * 1e3})
                self._last_rev = f.revolutions
        self.canvas.update()

    def _reconfigure(self, state):
        self.worker.reconfigure(self._scale_cfg(state))
        self.canvas.recompute_field(); self.eng.update_static(self.worker)
    def _push(self, d): self.worker.push(d)
    def _toggle_pause(self):
        self._paused = not self._paused; self.worker.set_paused(self._paused)
        self.controls.pause_btn.setText("Resume" if self._paused else "Pause")
    def _reset(self):
        self.worker.reconfigure(self._scale_cfg(self.controls._state)); self._last_rev = 0
        for d in (self.graph_p.data, self.graph_w.data):
            for k in d: d[k].clear()

    def _on_scale(self, s):
        self._scale = s
        self._reconfigure(self.controls._state)

    def _on_speed_scale(self, s):
        # direct call (the worker thread blocks in run(), so a queued slot would never fire)
        self.worker.set_time_scale(s)

    def _scale_cfg(self, cfg):
        # geometric + dynamic similarity: lengths*S, mass*S^3, losses*S^3 (so steady
        # RPM is preserved while power ~ S^3, field energy ~ S^3, force ~ S^2).
        S = self._scale
        if abs(S - 1.0) < 1e-9:
            return dict(cfg)
        e = dict(cfg)
        for k in ("R_track", "mag_len", "mag_wid", "gap"):
            e[k] = cfg[k] * S
        e["mag_len_rotor"] = cfg.get("mag_len_rotor", cfg["mag_len"]) * S
        e["mag_wid_rotor"] = cfg.get("mag_wid_rotor", cfg["mag_wid"]) * S
        e["rotor_mass"] = cfg["rotor_mass"] * S ** 3
        e["retract_mm"] = cfg["retract_mm"] * S
        e["friction"] = cfg["friction"] * S ** 3
        e["drag"] = cfg["drag"] * S ** 3
        e["coulomb"] = cfg["coulomb"] * S ** 3
        e["demag_tau"] = cfg["demag_tau"] * S
        return e

    # --- export / import -----------------------------------------------------
    @staticmethod
    def _jsonable(v):
        return v.item() if isinstance(v, np.generic) else v

    def _default_path(self, kind):
        os.makedirs("exports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exports/mag_{kind}_{ts}.json"

    def _export_stats(self):
        f = self.worker.latest; cfg = self.worker.cfg
        snap = {
            "type": "magnetic_gate_stats",
            "sim_time_s": f.t, "revolutions": f.revolutions,
            "rpm": f.rpm, "omega_rad_s": f.omega,
            "config": {k: self._jsonable(v) for k, v in cfg.items()},
            "per_revolution_mJ": {
                "magnetic_work": f.prop_last * 1e3, "switching_work": f.switch_last * 1e3,
                "dissipated": f.diss_last * 1e3, "net_dKE": f.net_last * 1e3,
                "revolution_time_s": f.rev_time,
            },
            "cumulative": {
                "magnetic_work_J": f.prop_cum, "magnetic_work_Wh": f.wh,
                "switching_work_J": f.switch_cum, "dissipated_J": f.diss_cum,
                "net_energy_J": f.net_cum, "avg_power_W": f.avg_power,
                "avg_power_hp": f.avg_power / 745.7, "avg_rpm": f.avg_rpm,
            },
            "instantaneous": {
                "magnetic_power_W": f.p_mag, "dissipated_power_W": f.p_diss,
                "net_power_W": f.p_net, "torque_mNm": f.tau_mag * 1e3,
                "force_mN": f.force_mag * 1e3, "kinetic_energy_mJ": f.ke * 1e3,
            },
            "engineering": self.eng.snapshot(self.worker),
        }
        path, _ = QFileDialog.getSaveFileName(self, "Export stats (JSON)",
                                              self._default_path("stats"), "JSON (*.json)")
        if path:
            with open(path, "w") as fh: json.dump(snap, fh, indent=2)
            self.statusBar().showMessage(f"saved {path}", 5000)

    def _export_preset(self):
        state = {k: self._jsonable(v) for k, v in self.controls._state.items() if k != "time_scale"}
        path, _ = QFileDialog.getSaveFileName(self, "Export preset (JSON)",
                                              self._default_path("preset"), "JSON (*.json)")
        if path:
            with open(path, "w") as fh: json.dump(state, fh, indent=2)
            self.statusBar().showMessage(f"saved {path}", 5000)

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset (JSON)",
                                              "exports", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh: data = json.load(fh)
        except Exception as e:
            self.statusBar().showMessage(f"load failed: {e}", 6000); return
        cfg = data.get("config", data) if isinstance(data, dict) else {}
        cfg = {k: v for k, v in cfg.items() if k in REF}
        if not cfg:
            self.statusBar().showMessage("no preset keys found", 6000); return
        self.controls.apply_preset(cfg)
        self.statusBar().showMessage(f"loaded {path}", 5000)

    def closeEvent(self, ev):
        self.worker.stop(); self.thread.quit(); self.thread.wait(2000); super().closeEvent(ev)


def main():
    app = QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
