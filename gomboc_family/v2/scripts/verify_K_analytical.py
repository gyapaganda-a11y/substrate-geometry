"""
verify_K_analytical.py — analytical (symbolic) Gaussian curvature on
Sloan-family bodies.

Replaces the finite-difference K computation in strict_convexity_gate.py
(which has unknown precision near possible curvature singularities) with
sympy-derived exact partial derivatives, lambdified to numpy for high-
speed evaluation on a dense grid.

This is Verification 1 / 2 / 3 of the post-reviewer-audit suite. The
question being answered: does Sloan beta=0.05 (and Phase-1, Phase-2,
Phase-3) genuinely have K<0 regions, or are the K<0 reports from the
numerical pipeline artifacts of finite-difference precision?

For radial parameterization r = R(theta, phi):
    x(theta, phi) = R(theta, phi) * u(theta, phi)
    u = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))

The Gaussian curvature is computed from first/second fundamental forms:
    E = x_theta . x_theta,  F = x_theta . x_phi,  G = x_phi . x_phi
    L = x_theta_theta . n,  M = x_theta_phi . n,  N = x_phi_phi . n
    K = (LN - M^2) / (EG - F^2)
where n is the outward unit normal.

All partial derivatives computed symbolically via sympy. Results converted
to numpy lambdas and evaluated on a 320x640 grid at multiple resolutions.

Usage:
    python verify_K_analytical.py --family sloan --beta 0.05 \\
        --out results/analytical_K_sloan_b005.json
"""
import argparse, json, os, sys, time
import numpy as np
import sympy as sp


def build_K_lambda(family, beta_val, coef_val=0.0, k_val=1, f_type="f4"):
    """Construct a sympy expression for K(theta, phi) and lambdify it.

    Returns: (K_callable, R_callable, gradR_callable)
        K_callable(theta, phi) -> Gaussian curvature
        R_callable(theta, phi) -> radial function
        gradR_callable(theta, phi) -> (dR_dtheta, dR_dphi)
    """
    th, ph = sp.symbols("theta phi", real=True)
    beta_s = sp.Float(beta_val)
    coef_s = sp.Float(coef_val)
    eta = (sp.Rational(3, 2) * sp.pi) * (sp.cos(th) - sp.cos(th) ** 3 / 3)

    if family == "sloan":
        r4 = 1 + 4 * beta_s * sp.sin(th) * sp.cos(ph - eta)
    elif family == "sloan-phase":
        eta_perturbed = eta + coef_s * sp.sin(k_val * eta)
        r4 = 1 + 4 * beta_s * sp.sin(th) * sp.cos(ph - eta_perturbed)
    elif family == "sloan-radial":
        if f_type == "f1":
            f_expr = sp.sin(th) * sp.sin(ph)
        elif f_type == "f2":
            f_expr = sp.sin(2 * th) * sp.cos(ph)
        elif f_type == "f3":
            f_expr = sp.sin(th) ** 2 * sp.cos(2 * ph)
        elif f_type == "f4":
            f_expr = sp.cos(th) * sp.sin(th) * sp.sin(ph)
        else:
            raise ValueError(f_type)
        r4 = 1 + 4 * beta_s * sp.sin(th) * sp.cos(ph - eta) + coef_s * f_expr
    else:
        raise ValueError(family)

    R = r4 ** sp.Rational(1, 4)
    u = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
    x = R * u

    x_th = sp.diff(x, th)
    x_ph = sp.diff(x, ph)
    x_thth = sp.diff(x_th, th)
    x_thph = sp.diff(x_th, ph)
    x_phph = sp.diff(x_ph, ph)

    # Normal = (x_th x x_ph) / ||...||
    cross = x_th.cross(x_ph)
    n_norm = sp.sqrt(cross.dot(cross))
    n_hat = cross / n_norm

    E_expr = x_th.dot(x_th)
    F_expr = x_th.dot(x_ph)
    G_expr = x_ph.dot(x_ph)
    L_expr = x_thth.dot(n_hat)
    M_expr = x_thph.dot(n_hat)
    N_expr = x_phph.dot(n_hat)

    K_expr = (L_expr * N_expr - M_expr ** 2) / (E_expr * G_expr - F_expr ** 2)
    # Don't simplify — too slow; lambdify directly.
    K_fn = sp.lambdify((th, ph), K_expr, modules="numpy")
    R_fn = sp.lambdify((th, ph), R, modules="numpy")
    dR_dth = sp.lambdify((th, ph), sp.diff(R, th), modules="numpy")
    dR_dph = sp.lambdify((th, ph), sp.diff(R, ph), modules="numpy")
    return K_fn, R_fn, (dR_dth, dR_dph)


def evaluate_K_grid(K_fn, n_t, n_p, eps_t=0.04):
    t = np.linspace(eps_t, np.pi - eps_t, n_t)
    p = np.linspace(0, 2 * np.pi, n_p, endpoint=False)
    T, P = np.meshgrid(t, p, indexing="ij")
    K = K_fn(T, P)
    # K_fn may return scalar at constant points; promote
    K = np.broadcast_to(K, T.shape).astype(float)
    return K, T, P


def summarize_K(K):
    valid = K[np.isfinite(K)]
    return {
        "K_min": float(np.min(valid)),
        "K_max": float(np.max(valid)),
        "K_mean": float(np.mean(valid)),
        "fraction_K_negative": float(np.mean(valid < 0)),
        "n_finite": int(valid.size),
        "n_nan": int(K.size - valid.size),
    }


def find_min_K_location(K, T, P):
    valid_mask = np.isfinite(K)
    idx = np.argmin(np.where(valid_mask, K, np.inf))
    i, j = np.unravel_index(idx, K.shape)
    return float(T[i, j]), float(P[i, j]), float(K[i, j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["sloan", "sloan-phase", "sloan-radial"])
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--coef", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--f-type", default="f4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Constructing symbolic K for {args.family} beta={args.beta} coef={args.coef} ...", file=sys.stderr)
    t0 = time.time()
    K_fn, R_fn, _ = build_K_lambda(args.family, args.beta, args.coef, args.k, args.f_type)
    t_build = time.time() - t0
    print(f"  symbolic build + lambdify: {t_build:.2f}s", file=sys.stderr)

    # Resolution sweep
    sweep = []
    for n_t in [160, 320, 640]:
        n_p = 2 * n_t
        t0 = time.time()
        K, T, P = evaluate_K_grid(K_fn, n_t, n_p)
        t_eval = time.time() - t0
        summary = summarize_K(K)
        theta_min, phi_min, Kmin_val = find_min_K_location(K, T, P)
        rec = {
            "n_t": n_t, "n_p": n_p,
            "compute_seconds": t_eval,
            **summary,
            "K_min_location": {"theta": theta_min, "phi": phi_min, "K": Kmin_val},
        }
        sweep.append(rec)
        print(f"  {n_t}x{n_p}: K_min={summary['K_min']:+.6f}  K_max={summary['K_max']:+.6f}  "
              f"frac<0={summary['fraction_K_negative']:.6f}  ({t_eval:.1f}s)", file=sys.stderr)

    result = {
        "family": args.family,
        "beta": args.beta,
        "coef": args.coef,
        "k": args.k,
        "f_type": args.f_type if args.family == "sloan-radial" else None,
        "method": "sympy_analytical",
        "symbolic_build_seconds": t_build,
        "resolution_sweep": sweep,
        "K_min_converged": sweep[-1]["K_min"],
        "strict_convexity_pass": bool(sweep[-1]["K_min"] > 0),
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWritten to {args.out}", file=sys.stderr)
    print(json.dumps({k: v for k, v in result.items() if k != "resolution_sweep"}, indent=2))
    print("Resolution sweep summary:")
    for rec in sweep:
        print(f"  {rec['n_t']:>4}x{rec['n_p']:<5}  K_min={rec['K_min']:+.6f}  frac<0={rec['fraction_K_negative']:.4%}")


if __name__ == "__main__":
    main()
