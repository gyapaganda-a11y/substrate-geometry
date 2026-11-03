"""Two small checks that accompany crosscheck_implicit_K.py (added for the v3 amendment).

1. Curvature at the point of least radius for Sloan's two stated bounds, compared with the exact values
   reported in the Wolfram MathWorld entry "Gomboc" (accessed 2026-11-03):
       Gomboc 1, beta = 0.15 : K = -(575/64) sqrt(5/2)
       Gomboc 2, beta = 0.17 : K = -45 (136 pi^2 - 25) / (2048 sqrt(2))
2. Behaviour of K near the poles (theta -> 0) as a function of phi, at beta = 0.03. A direction-dependent
   limit means the surface is not C^2 at the pole.

Usage: python scripts/crosscheck_poles_and_mathworld.py --out results/CROSSCHECK_poles_mathworld.json
(run as a file: it imports crosscheck_implicit_K from the same directory)
"""
import argparse, json, math
import numpy as np
import crosscheck_implicit_K as C

pi = math.pi
MW = {"g1": (0.15, -(575/64)*math.sqrt(5/2)),
      "g2": (0.17, -45*(136*pi**2-25)/(2048*math.sqrt(2)))}
LEAST_RADIUS_PHI = {"g1": 1.5*pi, "g2": pi}      # least-radius point is (theta, phi) = (pi/2, phi_min)

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/CROSSCHECK_poles_mathworld.json")
a = ap.parse_args()
fl = {s: C._build(C.SHAPES[s]["build"]()) for s in ("g1", "g2")}
out = {"mathworld_comparison": {}, "pole_limits_beta_0.03": {}}
for s, (beta, ref) in MW.items():
    k = float(C._K(fl[s], *C._pts(s, np.array(pi/2), np.array(LEAST_RADIUS_PHI[s]), beta), beta))
    out["mathworld_comparison"][s] = dict(beta=beta, K_computed=k, K_mathworld=ref, abs_diff=abs(k-ref))
    print(s, "beta", beta, "computed %.9f  MathWorld %.9f  diff %.2e" % (k, ref, abs(k-ref)))
phis = (0.0, pi/2, pi, 1.5*pi)
for s in ("g1", "g2"):
    rows = {}
    for th in (0.1, 0.01, 0.001, 0.0001):
        rows[str(th)] = [float(C._K(fl[s], *C._pts(s, np.array(th), np.array(p), 0.03), 0.03)) for p in phis]
    out["pole_limits_beta_0.03"][s] = dict(phi=list(phis), K_by_theta=rows)
    print(s, "K at theta=1e-4 for phi=0,pi/2,pi,3pi/2:", ["%.4f" % v for v in rows["0.0001"]])
json.dump(out, open(a.out, "w"), indent=1)
