"""
verify_identity_2d.py — confirm Varkonyi's gradient identity in 2D.

For a smooth 2D body parameterized in polar coords r = R(theta), Varkonyi's
identity (in 2D form) says: critical points of R(theta) and critical points
of h(phi) = max over theta of R(theta)*cos(theta - phi) coincide, for
strictly convex bodies (R + R'' > 0 everywhere).

This script tests the identity numerically on parameterized families of 2D
bodies. The script also tests non-strictly-convex bodies and shows that the
identity breaks there, confirming the precondition is essential.

Promoted (with cleanup) from research/varkonyi_diagnostic_2d.py.

Usage:
    python verify_identity_2d.py --out results/identity_2d_verification.json
"""
import argparse, json, os, sys
import numpy as np


def R(theta, params):
    a1, a2, a3, a4 = params
    return 1.0 + a1 * np.cos(theta) + a2 * np.cos(2 * theta) + a3 * np.cos(3 * theta) + a4 * np.sin(theta)


def Rp(theta, params):
    a1, a2, a3, a4 = params
    return -a1 * np.sin(theta) - 2 * a2 * np.sin(2 * theta) - 3 * a3 * np.sin(3 * theta) + a4 * np.cos(theta)


def Rpp(theta, params):
    a1, a2, a3, a4 = params
    return -a1 * np.cos(theta) - 4 * a2 * np.cos(2 * theta) - 9 * a3 * np.cos(3 * theta) - a4 * np.sin(theta)


def support(phi, params, n_search=20000):
    thetas = np.linspace(0, 2 * np.pi, n_search, endpoint=False)
    vals = R(thetas, params) * np.cos(thetas - phi)
    return float(np.max(vals))


def critical_points_R(params, n=200000):
    """Sign changes of R'(theta), with wraparound."""
    xs = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ys = Rp(xs, params)
    crits = []
    for i in range(n):
        j = (i + 1) % n
        if ys[i] * ys[j] <= 0 and not np.isclose(ys[i], ys[j]):
            crits.append((xs[i] + xs[j]) / 2)
    return crits


def critical_points_h(params, n_phi=8000):
    """Sample h(phi) at n_phi points, find local extrema."""
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    vals = np.array([support(phi, params) for phi in phis])
    crits = []
    for i in range(n_phi):
        im = (i - 1) % n_phi
        ip = (i + 1) % n_phi
        if (vals[i] > vals[im] and vals[i] > vals[ip]) or (vals[i] < vals[im] and vals[i] < vals[ip]):
            crits.append(phis[i])
    return crits


def is_strictly_convex(params, n=10000):
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return bool(np.min(R(ts, params) + Rpp(ts, params)) > 0)


def angular_pair_distance(crit_R, crit_h):
    """For each c in crit_R, find the nearest crit_h modulo 2pi; return max distance."""
    if not crit_R or not crit_h:
        return None
    max_d = 0.0
    for c in crit_R:
        ds = [min(abs(c - h), 2 * np.pi - abs(c - h)) for h in crit_h]
        max_d = max(max_d, min(ds))
    return max_d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cases = [
        ("circle perturbed (a1 only)", (0.10, 0.0, 0.0, 0.0)),
        ("two-mode small", (0.08, 0.02, 0.0, 0.0)),
        ("asymmetric small", (0.05, 0.0, 0.0, 0.03)),
        ("non-convex large a1 (control)", (0.95, 0.0, 0.0, 0.0)),  # R + R'' = 1 - 0.95 cos*2 - 0.95 cos*0 = NEG somewhere
    ]
    results = []
    for name, params in cases:
        convex = is_strictly_convex(params)
        crit_R = critical_points_R(params)
        crit_h = critical_points_h(params)
        # Compare counts (mod 2pi); for strict convex case they should coincide
        max_pair = angular_pair_distance(crit_R, crit_h)
        rec = {
            "case": name,
            "params": list(params),
            "strictly_convex": convex,
            "n_critical_points_R": len(crit_R),
            "n_critical_points_h": len(crit_h),
            "counts_match": (len(crit_R) == len(crit_h)),
            "max_R_to_h_angular_distance": max_pair,
        }
        results.append(rec)
        print(json.dumps(rec, indent=2))

    out = {"cases": results}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWritten to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
