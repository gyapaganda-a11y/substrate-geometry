"""
count_R_critpoints.py — find all critical points of R(theta, phi) on S^2
for a radial body. Combined with strict_convexity_gate, this gives an
internal consistency check on Varkonyi's identity:

  IF body is strictly convex (K > 0 everywhere) AND R has N critical points,
  THEN h has exactly N critical points.

When n_critical_h != n_critical_R for a body whose K_min > 0 on a fine grid,
the conclusion is that strict convexity is violated in regions smaller than
the grid spacing — i.e., K_min on a coarse grid is not a sufficient gate.

Usage:
    python count_R_critpoints.py --family sloan-phase --beta 0.023149 \\
        --coef 0.234433 --k 1 --out results/phase1_R_critpoints.json
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strict_convexity_gate import R_factory


def grad_R(R, t, p, eps=1e-6):
    dR_dt = (R(t + eps, p) - R(t - eps, p)) / (2 * eps)
    dR_dp = (R(t, p + eps) - R(t, p - eps)) / (2 * eps)
    return dR_dt, dR_dp


def grad_norm_sq(x, R):
    t, p = x
    if t < 0.01 or t > np.pi - 0.01:
        return 1e6
    dt, dp = grad_R(R, t, p)
    # Spherical gradient: |dR/dt|^2 + |1/sin t * dR/dp|^2
    return float(dt * dt + (dp / max(np.sin(t), 1e-6)) ** 2)


def find_critical_points(R, n_starts=400, sep_tol=0.01, tol=1e-9):
    rng = np.random.default_rng(42)
    crits = []
    for _ in range(n_starts):
        t0 = rng.uniform(0.1, np.pi - 0.1)
        p0 = rng.uniform(0, 2 * np.pi)
        res = minimize(grad_norm_sq, [t0, p0], args=(R,),
                       method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-18, "maxiter": 5000})
        if res.fun > tol:
            continue
        t, p = res.x
        p = p % (2 * np.pi)
        # Deduplicate by spherical distance
        is_new = True
        for (tc, pc, _) in crits:
            d = np.arccos(np.clip(
                np.sin(t) * np.sin(tc) * np.cos(p - pc) + np.cos(t) * np.cos(tc), -1, 1))
            if d < sep_tol:
                is_new = False
                break
        if is_new:
            crits.append((t, p, float(R(t, p))))
    return crits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["sloan", "sloan-phase", "sloan-radial"], required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--coef", type=float, default=0.0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--f-type", default="f4")
    p.add_argument("--n-starts", type=int, default=400)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    R = R_factory(args.family, args.beta, args.coef, args.k, args.f_type)
    crits = find_critical_points(R, n_starts=args.n_starts)
    result = {
        "family": args.family,
        "beta": args.beta,
        "coefficient": args.coef,
        "k": args.k,
        "n_starts": args.n_starts,
        "n_critical_points": len(crits),
        "critical_points": [{"theta": float(t), "phi": float(p), "R": r} for (t, p, r) in crits],
    }
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
