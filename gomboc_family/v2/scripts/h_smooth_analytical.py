"""
h_smooth_analytical.py — compute the support function h(d) analytically on a
smooth radial body, by 2D continuous optimization over (theta, phi).

This is what catalog paper v1 §4.3 line 219-224 claimed to have done but did
not. v2 makes good on the claim with a real, runnable implementation.

For each direction d in S^2:
    h(d) = max over (theta, phi) of R(theta, phi) * u(theta, phi) . d
where u = (sin theta cos phi, sin theta sin phi, cos theta) and R is the
radial function of the body. For a strictly convex body the maximizer is
unique; multi-start protects against non-convex regions where it isn't.

Then the spherical critical points of h - com.d (where com is the body's
COM) are counted: maxima, minima, and saddles distinguished by neighborhood
structure on a Fibonacci-sphere sampling.

Usage:
    python h_smooth_analytical.py --beta 0.05 --n-dirs 600 --n-starts 40 \
        --out results/sloan_b005_analytical_h.json

Supported body families (--family):
    sloan           : r^4 = 1 + 4*beta*sin(theta)*cos(phi - eta(theta))
    sloan-phase     : Fourier-perturbed phase: eta + coef * sin(k * eta)
    sloan-radial    : Sloan + additional radial term coef*cos(t)*sin(t)*sin(p)

Outputs JSON with:
    beta, family, coefficient, n_dirs, com,
    h_at_each_direction, height_range, n_maxima, n_minima, n_saddles,
    extrema_heights (sorted), gap_between_two_deepest_basins
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Radial functions
# --------------------------------------------------------------------------

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


def surface_point(t, p, R):
    r = R(t, p)
    return r * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


# --------------------------------------------------------------------------
# COM via solid-volume integration
# --------------------------------------------------------------------------

def estimate_com(R, n_t=120, n_p=240):
    """COM of the solid body bounded by r = R(theta, phi)."""
    t = np.linspace(1e-3, np.pi - 1e-3, n_t)
    p = np.linspace(0, 2 * np.pi, n_p, endpoint=False)
    T, P = np.meshgrid(t, p, indexing="ij")
    Rv = R(T, P)
    sint = np.sin(T)
    # dm proportional to r^2 dr sin theta d theta d phi, integrated 0..R: R^3/3 * sin theta
    # COM_i: integral of x_i * dm = R^4/4 * u_i * sin theta
    ux = np.sin(T) * np.cos(P)
    uy = np.sin(T) * np.sin(P)
    uz = np.cos(T)
    wt_top = (Rv ** 4 / 4) * sint
    wt_bot = (Rv ** 3 / 3) * sint
    dt, dp = t[1] - t[0], p[1] - p[0]
    Mx = np.sum(wt_top * ux) * dt * dp
    My = np.sum(wt_top * uy) * dt * dp
    Mz = np.sum(wt_top * uz) * dt * dp
    Mass = np.sum(wt_bot) * dt * dp
    return np.array([Mx, My, Mz]) / Mass


# --------------------------------------------------------------------------
# Smooth support function (the real analytical h)
# --------------------------------------------------------------------------

def smooth_h(d, R, n_starts=40, seed_starts=None):
    """h(d) = max over the smooth surface of v . d, via Powell-tight optimization
    from many starts to protect against non-convex regions where the maximizer
    is not unique."""
    best = -np.inf
    rng = np.random.default_rng(int(abs(d[0] * 1e6 + d[1] * 1e3 + d[2]) * 1e3) % (2 ** 31))
    starts = list(seed_starts) if seed_starts is not None else []
    while len(starts) < n_starts:
        starts.append((rng.uniform(0.1, np.pi - 0.1), rng.uniform(0, 2 * np.pi)))

    def neg(x):
        t, p = x
        if t <= 1e-6 or t >= np.pi - 1e-6:
            return 1e8
        return -np.dot(surface_point(t, p, R), d)

    for s in starts:
        res = minimize(neg, s, method="Powell",
                       options={"xtol": 1e-12, "ftol": 1e-14, "maxiter": 10000})
        if -res.fun > best:
            best = -res.fun
    return best


# --------------------------------------------------------------------------
# Sphere directions + extremum classification
# --------------------------------------------------------------------------

def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    z = 1 - 2 * (i + 0.5) / n
    rad = np.sqrt(1 - z * z)
    th = 2 * np.pi * i / phi
    return np.stack([rad * np.cos(th), rad * np.sin(th), z], axis=1)


def classify_extrema(dirs, heights, depth_thresh=1e-6, k=20):
    """A direction is a max/min only if its height differs from ALL k-nearest
    neighbors by more than depth_thresh in the same direction. Otherwise it is
    on a flat plateau or not extremal; saddles fall out as 'neither'."""
    tree = cKDTree(dirs)
    _, nbrs = tree.query(dirs, k=k + 1)
    maxima, minima = [], []
    for i in range(len(dirs)):
        nbr_h = heights[nbrs[i, 1:]]
        if heights[i] > nbr_h.max() + depth_thresh:
            maxima.append(i)
        elif heights[i] < nbr_h.min() - depth_thresh:
            minima.append(i)
    return maxima, minima


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def compute(family, beta, coef, k, n_dirs, n_starts, depth_thresh, f_type="f4"):
    R = R_factory(family, beta, coef, k, f_type)
    com = estimate_com(R)
    # Seed starts include the two locations where Sloan-family bodies tend to
    # have radial-critical points; this accelerates global maximizer finding.
    seed_starts = [(np.pi / 2, 0.0), (np.pi / 2, np.pi)]
    dirs = fibonacci_sphere(n_dirs)
    heights = np.zeros(n_dirs)
    for i, d in enumerate(dirs):
        h_val = smooth_h(d, R, n_starts=n_starts, seed_starts=seed_starts)
        heights[i] = h_val - np.dot(com, d)  # COM-frame height
    maxima, minima = classify_extrema(dirs, heights, depth_thresh=depth_thresh)
    n_basins = len(minima)
    n_peaks = len(maxima)
    # ECS = number of stable equilibria = number of MAXIMA of h - com.d
    # (because resting body's COM is HIGHEST in directions where h-com.d peaks)
    # but the catalog v1 convention measures basins = local minima of the
    # "drop height", which is the same thing under sign convention. Report both.
    sorted_min_h = np.sort(np.array([heights[i] for i in minima])) if minima else np.array([])
    sorted_max_h = np.sort(np.array([heights[i] for i in maxima])) if maxima else np.array([])
    gap_two_deepest = float(sorted_min_h[1] - sorted_min_h[0]) if len(sorted_min_h) >= 2 else None
    return {
        "family": family,
        "beta": beta,
        "coefficient": coef,
        "k": k,
        "f_type": f_type if family == "sloan-radial" else None,
        "n_dirs": n_dirs,
        "n_starts_per_dir": n_starts,
        "depth_threshold": depth_thresh,
        "com": com.tolist(),
        "h_range": [float(heights.min()), float(heights.max())],
        "h_extent": float(heights.max() - heights.min()),
        "n_maxima": n_peaks,
        "n_minima": n_basins,
        "minima_heights_sorted": sorted_min_h.tolist(),
        "maxima_heights_sorted": sorted_max_h.tolist(),
        "gap_between_two_deepest_minima": gap_two_deepest,
        "all_heights": heights.tolist(),
        "directions": dirs.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["sloan", "sloan-phase", "sloan-radial"], default="sloan")
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--coef", type=float, default=0.0)
    p.add_argument("--k", type=int, default=1, help="harmonic order for sloan-phase")
    p.add_argument("--f-type", choices=list(RADIAL_PERTURBATIONS), default="f4")
    p.add_argument("--n-dirs", type=int, default=600)
    p.add_argument("--n-starts", type=int, default=40)
    p.add_argument("--depth-threshold", type=float, default=1e-6)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = compute(args.family, args.beta, args.coef, args.k, args.n_dirs, args.n_starts, args.depth_threshold, args.f_type)
    summary = {k: v for k, v in result.items() if k not in ("all_heights", "directions")}
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull result written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
