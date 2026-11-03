"""
verify_R_critpoints_analytical.py — analytical R-critical-point counting
via symbolic gradient + scipy.optimize.root from a dense grid of starts.

Replaces the Nelder-Mead-on-||grad-R||^2 approach in count_R_critpoints.py
which may have spurious convergence issues. This script:

  1. Constructs partial derivatives of R(theta, phi) symbolically.
  2. Lambdifies the 2D gradient system [dR/dtheta, dR/dphi].
  3. Uses scipy.optimize.root (hybrid Powell or Levenberg-Marquardt) to find
     zeros of the gradient from a dense regular grid of (theta, phi) starts.
  4. Deduplicates by spherical distance.
  5. Reports critical points, their R values, and classifies each as
     max / min / saddle via the Hessian determinant.

Usage:
    python verify_R_critpoints_analytical.py --family sloan-radial \\
        --beta 0.005 --coef 0.05102 --f-type f3 \\
        --out results/V4_R_critpoints_minasym_b0005.json
"""
import argparse, json, os, sys, time
import numpy as np
import sympy as sp
from scipy.optimize import root


def build_gradient_lambdas(family, beta_val, coef_val, k_val=1, f_type="f4"):
    th, ph = sp.symbols("theta phi", real=True)
    beta_s = sp.Float(beta_val)
    coef_s = sp.Float(coef_val)
    eta = (sp.Rational(3, 2) * sp.pi) * (sp.cos(th) - sp.cos(th) ** 3 / 3)

    if family == "sloan":
        r4 = 1 + 4 * beta_s * sp.sin(th) * sp.cos(ph - eta)
    elif family == "sloan-phase":
        eta_p = eta + coef_s * sp.sin(k_val * eta)
        r4 = 1 + 4 * beta_s * sp.sin(th) * sp.cos(ph - eta_p)
    elif family == "sloan-radial":
        if f_type == "f1":   f_expr = sp.sin(th) * sp.sin(ph)
        elif f_type == "f2": f_expr = sp.sin(2 * th) * sp.cos(ph)
        elif f_type == "f3": f_expr = sp.sin(th) ** 2 * sp.cos(2 * ph)
        elif f_type == "f4": f_expr = sp.cos(th) * sp.sin(th) * sp.sin(ph)
        else: raise ValueError(f_type)
        r4 = 1 + 4 * beta_s * sp.sin(th) * sp.cos(ph - eta) + coef_s * f_expr
    else:
        raise ValueError(family)

    R = r4 ** sp.Rational(1, 4)
    dR_dth = sp.diff(R, th)
    dR_dph = sp.diff(R, ph)
    # Hessian for classification
    d2R_dthth = sp.diff(dR_dth, th)
    d2R_dthph = sp.diff(dR_dth, ph)
    d2R_dphph = sp.diff(dR_dph, ph)
    return {
        "R": sp.lambdify((th, ph), R, modules="numpy"),
        "grad": sp.lambdify((th, ph), [dR_dth, dR_dph], modules="numpy"),
        "hess": sp.lambdify((th, ph), [[d2R_dthth, d2R_dthph], [d2R_dthph, d2R_dphph]], modules="numpy"),
    }


def find_all_critical_points(lambdas, n_grid_t=24, n_grid_p=48, sep_tol=0.02, residual_tol=1e-10):
    """Solve grad R = 0 from a dense regular grid of starts, deduplicate."""
    grad = lambdas["grad"]
    def F(x):
        t, p = x
        # Keep theta away from poles; phi wrap handled implicitly
        if t < 1e-4 or t > np.pi - 1e-4:
            return [1e6, 1e6]
        return grad(t, p)

    # Dense regular start grid
    ts = np.linspace(0.05, np.pi - 0.05, n_grid_t)
    ps = np.linspace(0, 2 * np.pi, n_grid_p, endpoint=False)
    found = []
    for t0 in ts:
        for p0 in ps:
            try:
                sol = root(F, [t0, p0], method="hybr",
                           options={"xtol": 1e-12, "maxfev": 2000})
            except Exception:
                continue
            if not sol.success:
                continue
            t, p = sol.x
            res = np.linalg.norm(F([t, p]))
            if res > residual_tol:
                continue
            p = p % (2 * np.pi)
            if t < 0.01 or t > np.pi - 0.01:
                continue
            # Deduplicate by spherical distance
            is_new = True
            for (tc, pc, _) in found:
                d = float(np.arccos(np.clip(
                    np.sin(t) * np.sin(tc) * np.cos(p - pc) + np.cos(t) * np.cos(tc), -1, 1)))
                if d < sep_tol:
                    is_new = False
                    break
            if is_new:
                R_val = float(lambdas["R"](t, p))
                found.append((float(t), float(p), R_val))
    return found


def classify_critical_point(t, p, lambdas):
    H = np.asarray(lambdas["hess"](t, p), dtype=float)
    det = float(np.linalg.det(H))
    trace = float(np.trace(H))
    if det > 0:
        return "max" if trace < 0 else "min"
    elif det < 0:
        return "saddle"
    else:
        return "degenerate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["sloan", "sloan-phase", "sloan-radial"])
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--coef", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--f-type", default="f4")
    ap.add_argument("--n-grid-t", type=int, default=24)
    ap.add_argument("--n-grid-p", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Building symbolic gradient for {args.family} beta={args.beta} coef={args.coef} ...", file=sys.stderr)
    t0 = time.time()
    L = build_gradient_lambdas(args.family, args.beta, args.coef, args.k, args.f_type)
    print(f"  symbolic build: {time.time()-t0:.2f}s", file=sys.stderr)

    print(f"Searching from {args.n_grid_t}x{args.n_grid_p} grid of starts ({args.n_grid_t * args.n_grid_p} total)...",
          file=sys.stderr)
    t0 = time.time()
    crits = find_all_critical_points(L, n_grid_t=args.n_grid_t, n_grid_p=args.n_grid_p)
    print(f"  root-finding: {time.time()-t0:.2f}s", file=sys.stderr)

    classified = []
    for (t, p, r) in crits:
        kind = classify_critical_point(t, p, L)
        classified.append({"theta": t, "phi": p, "R": r, "kind": kind})

    n_min = sum(1 for c in classified if c["kind"] == "min")
    n_max = sum(1 for c in classified if c["kind"] == "max")
    n_sad = sum(1 for c in classified if c["kind"] == "saddle")
    n_deg = sum(1 for c in classified if c["kind"] == "degenerate")
    euler_check = n_min - n_sad + n_max  # should be 2 for S^2 if all crit-pts found

    result = {
        "family": args.family,
        "beta": args.beta,
        "coef": args.coef,
        "k": args.k,
        "f_type": args.f_type if args.family == "sloan-radial" else None,
        "method": "analytical_gradient_+_scipy_root_+_hessian_classification",
        "n_starts_grid": [args.n_grid_t, args.n_grid_p],
        "n_critical_points_total": len(classified),
        "n_minima": n_min,
        "n_maxima": n_max,
        "n_saddles": n_sad,
        "n_degenerate": n_deg,
        "euler_min_minus_saddle_plus_max": euler_check,
        "critical_points": sorted(classified, key=lambda c: -c["R"]),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {args.out}", file=sys.stderr)
    print(json.dumps({k: v for k, v in result.items() if k != "critical_points"}, indent=2))
    print("Critical points (sorted by R desc):")
    for c in result["critical_points"][:20]:
        print(f"  theta={c['theta']:.4f}  phi={c['phi']:.4f}  R={c['R']:.6f}  kind={c['kind']}")


if __name__ == "__main__":
    main()
