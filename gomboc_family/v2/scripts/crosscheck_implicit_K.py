"""Independent cross-check of the strict-convexity gate (added for the v3 amendment).

Does NOT import any other script in this directory. Method differs from verify_K_analytical.py:
Gaussian curvature of the implicit level set F(p) = |p|^4 - 1 - 4*beta*s(p/|p|) = 0,
    K = (gradF . adj(Hess F) . gradF) / |gradF|^4,
with the Hessian from symbolic differentiation of F in Cartesian coordinates (no (theta, phi) chart,
so no pole singularity) and the adjugate formed numerically.

Shapes (Sloan, arXiv:2306.14914):
  g2 : r^4 = 1 + 4 beta sin(th) cos(ph - (3 pi/2)(cos th - cos^3 th / 3))     [Eq. 16, "Gomboc 2"]
  g1 : r^4 = 1 + 4 beta sin(th) cos(ph - 5 th)                                  [Eq. 12, "Gomboc 1"]

For each shape and beta: K_min on a dense grid, then Nelder-Mead refinement from the best grid points
(so a minimum lying between grid points is not missed). The convexity-loss boundary is the root of
refined K_min(beta) = 0. For g2 the critical points of s on S^2 are also counted (beta-independent,
since R^4 = 1 + 4 beta s) and classified, with the Morse-Euler check M - S + m = 2.

Usage: python scripts/crosscheck_implicit_K.py --out results/CROSSCHECK_implicit_K.json
"""
import argparse, json, platform
import numpy as np, scipy, sympy as sp
from scipy.optimize import minimize, brentq, fsolve

x, y, z, b = sp.symbols("x y z beta", real=True)
r = sp.sqrt(x**2 + y**2 + z**2)
w = z / r


def _s_g2():
    eta = sp.Rational(3, 2) * sp.pi * (w - w**3 / 3)
    return (x * sp.cos(eta) + y * sp.sin(eta)) / r        # = sin(th) cos(ph - eta(th))


def _s_g1():
    rho = sp.sqrt(x**2 + y**2)
    sth = rho / r
    c5 = sp.expand(sp.chebyshevt(5, w))                    # cos(5 th)
    s5 = sth * sp.expand(sp.chebyshevu(4, w))              # sin(5 th) = sin(th) U4(cos th)
    return sth * ((x / rho) * c5 + (y / rho) * s5)         # = sin(th) cos(ph - 5 th)


def _build(s):
    F = r**4 - 1 - 4 * b * s
    V = (x, y, z)
    g = [sp.diff(F, v) for v in V]
    H = [[sp.diff(g[i], V[j]) for j in range(3)] for i in range(3)]
    flat = g + [H[0][0], H[0][1], H[0][2], H[1][1], H[1][2], H[2][2]]
    return sp.lambdify((x, y, z, b), flat, modules="numpy", cse=True)


def _K(fl, X, Y, Z, beta):
    g1, g2, g3, h11, h12, h13, h22, h23, h33 = fl(X, Y, Z, beta)
    a11 = h22*h33-h23*h23; a22 = h11*h33-h13*h13; a33 = h11*h22-h12*h12
    a12 = -(h12*h33-h13*h23); a13 = h12*h23-h13*h22; a23 = -(h11*h23-h12*h13)
    num = g1*(a11*g1+a12*g2+a13*g3)+g2*(a12*g1+a22*g2+a23*g3)+g3*(a13*g1+a23*g2+a33*g3)
    n2 = g1*g1+g2*g2+g3*g3
    return num/(n2*n2)


SHAPES = {
    "g2": dict(s0=lambda T, P: np.sin(T)*np.cos(P-1.5*np.pi*(np.cos(T)-np.cos(T)**3/3)), build=_s_g2),
    "g1": dict(s0=lambda T, P: np.sin(T)*np.cos(P-5*T), build=_s_g1),
}


def _pts(shape, T, P, beta):
    R = (1+4*beta*SHAPES[shape]["s0"](T, P))**0.25
    return R*np.sin(T)*np.cos(P), R*np.sin(T)*np.sin(P), R*np.cos(T)


def kmin(shape, fl, beta, nt=240, nph=480, nstart=10):
    ths = np.linspace(0.01, np.pi-0.01, nt); phs = np.linspace(0, 2*np.pi, nph, endpoint=False)
    T, P = np.meshgrid(ths, phs, indexing="ij")
    K = _K(fl, *_pts(shape, T, P, beta), beta)
    gmin = float(np.nanmin(K))
    order = np.dstack(np.unravel_index(np.argsort(K, axis=None)[:400], K.shape))[0]
    best, loc, seen = gmin, None, []
    for i, j in order:
        t0, p0 = ths[i], phs[j]
        if any(abs(t0-a)+abs(p0-c) < 0.15 for a, c in seen):
            continue
        seen.append((t0, p0))
        if len(seen) > nstart:
            break
        res = minimize(lambda q: float(_K(fl, *_pts(shape, q[0], q[1], beta), beta)), [t0, p0],
                       method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 4000})
        if res.fun < best:
            best, loc = float(res.fun), [float(res.x[0]), float(res.x[1])]
    return gmin, best, loc


def critical_points_g2():
    th, ph = sp.symbols("th ph", real=True)
    sf = sp.sin(th)*sp.cos(ph-sp.Rational(3, 2)*sp.pi*(sp.cos(th)-sp.cos(th)**3/3))
    gt, gp = sp.lambdify((th, ph), sp.diff(sf, th), "numpy"), sp.lambdify((th, ph), sp.diff(sf, ph), "numpy")
    Hf = sp.lambdify((th, ph), sp.Matrix([[sp.diff(sf, th, th), sp.diff(sf, th, ph)],
                                          [sp.diff(sf, th, ph), sp.diff(sf, ph, ph)]]), "numpy")
    sols = []
    for t0 in np.linspace(0.05, np.pi-0.05, 40):
        for p0 in np.linspace(0, 2*np.pi, 80, endpoint=False):
            q, _, ier, _ = fsolve(lambda q: [gt(q[0], q[1]), gp(q[0], q[1])/max(np.sin(q[0]), 1e-9)], [t0, p0],
                                  full_output=True, xtol=1e-13)
            if ier != 1 or not (1e-3 < q[0] < np.pi-1e-3):
                continue
            t, p = q[0], q[1] % (2*np.pi)
            if max(abs(gt(t, p)), abs(gp(t, p))) > 1e-9:
                continue
            if any(abs(t-a) < 1e-6 and min(abs(p-c), 2*np.pi-abs(p-c)) < 1e-6 for a, c, _ in sols):
                continue
            Hm = np.array(Hf(t, p), dtype=float)
            Ht = np.array([[Hm[0, 0], Hm[0, 1]/np.sin(t)], [Hm[0, 1]/np.sin(t), Hm[1, 1]/np.sin(t)**2]])
            ev = np.linalg.eigvalsh(Ht)
            sols.append((float(t), float(p), "max" if ev.max() < 0 else ("min" if ev.min() > 0 else "saddle")))
    M = sum(k == "max" for *_, k in sols); m = sum(k == "min" for *_, k in sols); S = sum(k == "saddle" for *_, k in sols)
    return dict(points=sols, maxima=M, minima=m, saddles=S, euler=M-S+m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/CROSSCHECK_implicit_K.json")
    a = ap.parse_args()
    out = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                             sympy=sp.__version__), shapes={})
    betas = {"g2": [0.010, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.080],
             "g1": [0.010, 0.020, 0.030, 0.050, 0.080, 0.100, 0.120, 0.150]}
    for shape in ("g2", "g1"):
        fl = _build(SHAPES[shape]["build"]())
        rows = []
        for beta in betas[shape]:
            gmin, rmin, loc = kmin(shape, fl, beta)
            rows.append(dict(beta=beta, K_min_grid=gmin, K_min_refined=rmin, location_theta_phi=loc))
            print(shape, beta, "grid %+.5f refined %+.5f" % (gmin, rmin), flush=True)
        f = lambda bb: kmin(shape, fl, bb, 160, 320, 8)[1]
        grid = np.arange(0.005, 0.151, 0.0025)
        lo = next(g for g in grid if f(g) < 0)
        bstar = brentq(f, lo-0.0025, lo, xtol=1e-7)
        out["shapes"][shape] = dict(rows=rows, beta_star=bstar,
                                    sloan_stated_bound=0.15 if shape == "g1" else 0.17,
                                    overshoot_factor=(0.15 if shape == "g1" else 0.17)/bstar)
        print(shape, "beta* =", bstar, "factor", out["shapes"][shape]["overshoot_factor"], flush=True)
    out["g2_critical_points_of_s"] = critical_points_g2()
    print(out["g2_critical_points_of_s"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
