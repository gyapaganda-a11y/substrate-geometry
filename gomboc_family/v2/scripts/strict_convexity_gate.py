"""
strict_convexity_gate.py — Gaussian-curvature based strict-convexity test for
radial-parameterized bodies.

This replaces the v1 convexity metric (mesh_vol / hull_vol > 0.999), which is
satisfied by bodies with small K<0 regions and therefore does not detect loss
of strict convexity — the precondition of Varkonyi's gradient identity and of
Sloan's mono-monostatic argument.

For a smooth surface parameterized by r = R(theta, phi), Gaussian curvature
at each point is K = (LN - M^2) / (EG - F^2), where E, F, G are coefficients
of the first fundamental form and L, M, N are coefficients of the second.
Both are computed by finite differences on a dense (theta, phi) grid.

A body passes the strict-convexity gate iff min(K) > 0 over the entire grid.

Usage:
    python strict_convexity_gate.py --family sloan-phase --beta 0.023149 \\
        --coef 0.234433 --k 1 --out results/phase1_convexity.json

Outputs JSON with K_min, K_max, K_mean, fraction_K_negative, gate_pass.
"""
import argparse, json, os, sys
import numpy as np


def eta_base(t):
    return (3 * np.pi / 2) * (np.cos(t) - np.cos(t) ** 3 / 3)


def sloan_R(t, p, beta):
    r4 = 1.0 + 4.0 * beta * np.sin(t) * np.cos(p - eta_base(t))
    return np.where(r4 > 0, np.abs(r4) ** 0.25, 1e-6)


def sloan_phase_R(t, p, beta, coef, k=1):
    eta = eta_base(t) + coef * np.sin(k * eta_base(t))
    r4 = 1.0 + 4.0 * beta * np.sin(t) * np.cos(p - eta)
    return np.where(r4 > 0, np.abs(r4) ** 0.25, 1e-6)


RADIAL_PERTURBATIONS = {
    "f1": lambda T, P: np.sin(T) * np.sin(P),
    "f2": lambda T, P: np.sin(2 * T) * np.cos(P),
    "f3": lambda T, P: np.sin(T) ** 2 * np.cos(2 * P),
    "f4": lambda T, P: np.cos(T) * np.sin(T) * np.sin(P),
}


def sloan_radial_R(t, p, beta, coef, f_type="f4"):
    base = 1.0 + 4.0 * beta * np.sin(t) * np.cos(p - eta_base(t))
    r4 = base + coef * RADIAL_PERTURBATIONS[f_type](t, p)
    return np.where(r4 > 0, np.abs(r4) ** 0.25, 1e-6)


def R_factory(family, beta, coef, k=1, f_type="f4"):
    if family == "sloan":
        return lambda t, p: sloan_R(t, p, beta)
    if family == "sloan-phase":
        return lambda t, p: sloan_phase_R(t, p, beta, coef, k)
    if family == "sloan-radial":
        return lambda t, p: sloan_radial_R(t, p, beta, coef, f_type)
    raise ValueError(f"unknown family: {family}")


def surface(R, t, p):
    """Vector-aware surface position x(t, p) = R(t,p) * u(t,p)."""
    r = R(t, p)
    return np.stack([r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p), r * np.cos(t)], axis=-1)


def gaussian_curvature_grid(R, n_t=160, n_p=320, eps_t=0.04):
    """Compute K on a (theta, phi) grid via numerical first/second fundamental
    forms. eps_t bounds theta away from the poles where the polar parameterization
    is singular but the body itself is smooth (the pole singularity is a
    coordinate artifact in this case; we skip 4% of theta range to avoid it)."""
    t = np.linspace(eps_t, np.pi - eps_t, n_t)
    p = np.linspace(0, 2 * np.pi, n_p, endpoint=False)
    dt = t[1] - t[0]
    dp = p[1] - p[0]

    # 5-point stencil for derivatives via vectorized evaluation
    T, P = np.meshgrid(t, p, indexing="ij")  # (n_t, n_p)

    def s(dT, dP):
        return surface(R, T + dT, (P + dP) % (2 * np.pi))

    xc = s(0, 0)
    xt = (s(dt, 0) - s(-dt, 0)) / (2 * dt)
    xp = (s(0, dp) - s(0, -dp)) / (2 * dp)
    xtt = (s(dt, 0) - 2 * xc + s(-dt, 0)) / (dt * dt)
    xpp = (s(0, dp) - 2 * xc + s(0, -dp)) / (dp * dp)
    xtp = (s(dt, dp) - s(dt, -dp) - s(-dt, dp) + s(-dt, -dp)) / (4 * dt * dp)

    # Cross product element-wise
    n_vec = np.cross(xt, xp)  # (n_t, n_p, 3)
    n_norm = np.linalg.norm(n_vec, axis=-1)
    safe = n_norm > 1e-12
    n_hat = np.zeros_like(n_vec)
    n_hat[safe] = n_vec[safe] / n_norm[safe, None]

    E = np.einsum("...i,...i->...", xt, xt)
    F = np.einsum("...i,...i->...", xt, xp)
    G = np.einsum("...i,...i->...", xp, xp)
    L = np.einsum("...i,...i->...", xtt, n_hat)
    M = np.einsum("...i,...i->...", xtp, n_hat)
    N = np.einsum("...i,...i->...", xpp, n_hat)

    denom = E * G - F * F
    K = np.where(np.abs(denom) > 1e-14, (L * N - M * M) / denom, np.nan)
    return K[~np.isnan(K)]


def compute(family, beta, coef, k, n_t, n_p, eps_t, f_type="f4"):
    R = R_factory(family, beta, coef, k, f_type)
    K = gaussian_curvature_grid(R, n_t=n_t, n_p=n_p, eps_t=eps_t)
    K_min = float(np.min(K))
    K_max = float(np.max(K))
    K_mean = float(np.mean(K))
    frac_neg = float(np.mean(K < 0))
    return {
        "family": family,
        "beta": beta,
        "coefficient": coef,
        "k": k,
        "f_type": f_type if family == "sloan-radial" else None,
        "grid": [n_t, n_p],
        "theta_eps": eps_t,
        "K_min": K_min,
        "K_max": K_max,
        "K_mean": K_mean,
        "fraction_K_negative": frac_neg,
        "gate_pass": bool(K_min > 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["sloan", "sloan-phase", "sloan-radial"], required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--coef", type=float, default=0.0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--f-type", choices=list(RADIAL_PERTURBATIONS), default="f4")
    p.add_argument("--n-t", type=int, default=160)
    p.add_argument("--n-p", type=int, default=320)
    p.add_argument("--theta-eps", type=float, default=0.04)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = compute(args.family, args.beta, args.coef, args.k, args.n_t, args.n_p, args.theta_eps, args.f_type)
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Result written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
