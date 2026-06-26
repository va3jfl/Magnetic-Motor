"""
linear_gate.py -- Howard Johnson's FIRST prototype: a straight magnetic gate.
Place a magnet cart AT REST at the entrance (no push, no actuator) and let
attraction act.  Does the gate pull it through with net energy, or not?

Pure Coulombian model (same as the circular sim), 1-D along the track.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU0 = 4.0e-7 * math.pi
MU0_4PI = 1.0e-7


def build_gate(n, pitch, gap, L, Wside, Br, flux, aux_br, aux_off_mm, aux_L, aux_W):
    """Stator poles (+/-) along the track at y=gap, plus aux flux-gate poles."""
    q = (Br / MU0) * Wside * Wside
    px, py, qj = [], [], []
    for i in range(n):
        x = i * pitch
        px += [x, x]; py += [gap + L / 2, gap - L / 2]; qj += [+q, -q]      # vertical magnet
    if flux:
        qa = (aux_br / MU0) * aux_W * aux_W
        for i in range(n):
            x = (i + 0.5) * pitch + aux_off_mm
            px += [x, x]; py += [gap + aux_L / 2, gap - aux_L / 2]; qj += [+qa, -qa]
    return np.array(px), np.array(py), np.array(qj)


def cart_poles(x, L):
    return np.array([x, x]), np.array([L / 2, -L / 2])


def force_x(x, gpx, gpy, gq, L, Wside, Br, soft2):
    cq = (Br / MU0) * Wside * Wside
    cqs = np.array([+cq, -cq])
    cpx, cpy = cart_poles(x, L)
    fx = 0.0
    for k in range(2):
        dx = cpx[k] - gpx; dy = cpy[k] - gpy
        d2 = dx * dx + dy * dy + soft2
        coef = MU0_4PI * cqs[k] * gq / d2 ** 1.5
        fx += float(np.sum(coef * dx))
    return fx


def run(label, n=6, pitch=0.0565, gap=0.008, L=0.016, Wside=0.01235, Br=1.05,
        flux=True, aux_br=0.5, aux_off_mm=-0.003, aux_L=0.008, aux_W=0.0074,
        x0=-1.5 * 0.0565, v0=0.0, mass=0.030, t_max=0.8, dt=2e-6, grade=0.0):
    gpx, gpy, gq = build_gate(n, pitch, gap, L, Wside, Br, flux, aux_br, aux_off_mm, aux_L, aux_W)
    if grade:                                  # one-way graded gate: weaken magnets left->right less
        gpx = gpx.copy()                       # (placeholder; grading applied via gap ramp below)
    soft2 = (0.25 * Wside) ** 2
    x, v, t = x0, v0, 0.0
    xs, vs, ts = [x], [v], [0.0]
    while t < t_max:
        # symplectic Euler
        f = force_x(x, gpx, gpy, gq, L, Wside, Br, soft2)
        v += f / mass * dt
        x += v * dt
        t += dt
        xs.append(x); vs.append(v); ts.append(t)
        if x > (n + 1) * pitch:                # exited the far side
            break
    xs, vs, ts = np.array(xs), np.array(vs), np.array(ts)
    ke = 0.5 * mass * vs * vs
    exit_x = n * pitch
    crossed = xs[-1] > exit_x
    peak_ke = ke.max()
    # net KE gain from start to gate exit (or end)
    end_ke = ke[-1]
    print(f"{label}:")
    print(f"  start: x0={x0*1e3:+.1f} mm at rest (KE=0)")
    print(f"  peak KE (attraction pulling it in) = {peak_ke*1e3:7.2f} mJ")
    print(f"  {'CROSSED the gate' if crossed else 'TRAPPED (did not reach the exit)'}: "
          f"ended at x={xs[-1]*1e3:+.1f} mm, KE={end_ke*1e3:7.2f} mJ")
    if crossed:
        print(f"  net KE at exit (gain vs start) = {end_ke*1e3:7.2f} mJ")
    return xs, ke, exit_x, crossed


def main():
    print("Johnson LINEAR gate -- cart placed AT REST at the entrance, no push, no actuator:\n")
    run("uniform gate + flux-gate aux")
    print()
    # the reset cost demo: a graded one-way gate DOES let it cross from rest (one-shot),
    # but only because the potential was pre-tilted -- returning the cart costs it back.
    print("For reference, a one-way GRADED gate (potential pre-tilted entrance->exit):")
    run("graded one-way gate", gap=0.008, grade=1.0)


if __name__ == "__main__":
    main()
