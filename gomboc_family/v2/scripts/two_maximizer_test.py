"""
two_maximizer_test.py — demonstrate that non-strictly-convex bodies have
direction d for which multiple surface points achieve max v.d.

For a STRICTLY convex body, the support point in direction d is unique. The
script multi-starts the surface-point maximization from many initial (theta,
phi); for a strictly convex body all starts converge to the same surface
point. For a non-strictly-convex body, starts can converge to distinct
surface points with the same v.d value. The number of distinct converged
points is reported.

This is the cleanest behavioral demonstration that Sloan's beta>=0.05 bodies
violate the strict-convexity precondition of Varkonyi's identity.

Promoted from research/varkonyi_diagnostic_convexity.py (the multi-maximizer
portion).

Usage:
    python two_maximizer_test.py --family sloan --beta 0.05 \\
        --out results/two_maximizer_b005.json
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strict_convexity_gate import R_factory


def surface_xyz(t, p, R):
    r = R(t, p)
    return np.array([r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p), r * np.cos(t)])


def find_all_maximizers(d, R, n_starts=80, tie_tol=1e-4, sep_tol=0.01):
    """Multi-start search for points maximizing v.d. Returns distinct points
    that are within tie_tol of the global maximum."""
    rng = np.random.default_rng(0)
    found = []

    def neg(x):
        t, p = x
        if t <= 1e-6 or t >= np.pi - 1e-6:
            return 1e8
        return -np.dot(surface_xyz(t, p, R), d)

    for _ in range(n_starts):
        t0 = rng.uniform(0.1, np.pi - 0.1)
        p0 = rng.uniform(0, 2 * np.pi)
        res = minimize(neg, [t0, p0], method="Powell",
                       options={"xtol": 1e-12, "ftol": 1e-14, "maxiter": 10000})
        t, p = res.x
        p = p % (2 * np.pi)
        if t < 0.01 or t > np.pi - 0.01:
            continue
        v_d = -res.fun
        is_new = True
        for (tc, pc, vc) in found:
            if abs(t - tc) < sep_tol and (abs(p - pc) < sep_tol or abs(abs(p - pc) - 2 * np.pi) < sep_tol):
                is_new = False
                break
        if is_new:
            found.append((t, p, v_d))
    found.sort(key=lambda x: -x[2])
    if not found:
        return []
    top = [pt for pt in found if pt[2] > found[0][2] - tie_tol]
    return top


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["sloan", "sloan-phase", "sloan-radial"], required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--coef", type=float, default=0.0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--f-type", default="f4")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    R = R_factory(args.family, args.beta, args.coef, args.k, args.f_type)

    test_directions = {
        "(+1,0,0)": np.array([1.0, 0, 0]),
        "(-1,0,0)": np.array([-1.0, 0, 0]),
        "(0,0,1)": np.array([0, 0, 1.0]),
        "(0,0,-1)": np.array([0, 0, -1.0]),
        "(0.6,0.4,0.7)": np.array([0.6, 0.4, 0.7]) / np.linalg.norm([0.6, 0.4, 0.7]),
    }

    result = {
        "family": args.family,
        "beta": args.beta,
        "coefficient": args.coef,
        "by_direction": {},
    }

    for label, d in test_directions.items():
        tops = find_all_maximizers(d, R, n_starts=80)
        result["by_direction"][label] = {
            "v_d_max": tops[0][2] if tops else None,
            "n_distinct_maximizers": len(tops),
            "maximizers": [{"theta": float(t), "phi": float(p), "v_d": float(vd)} for (t, p, vd) in tops[:5]],
        }
        print(f"d={label}: v.d={tops[0][2]:.6f} -> {len(tops)} distinct maximizer(s)")

    multi_max_directions = [k for k, v in result["by_direction"].items() if v["n_distinct_maximizers"] > 1]
    result["strict_convexity_violated_in_directions"] = multi_max_directions
    result["strict_convexity_violated"] = bool(multi_max_directions)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWritten to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
