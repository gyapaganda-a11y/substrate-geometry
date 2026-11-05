"""
Strict-convexity check of the published bodies (Table 4 of arXiv:2604.17095).

For each instance the surface is r^4 = 1 + 4 beta sin(theta) cos(phi - P(theta)),
P(theta) = eta(theta) + a sin(k eta(theta)), eta = (3 pi / 2)(cos theta - cos^3 theta / 3).

The minimum Gaussian curvature K_min is computed two independent ways:
  1. implicit-surface formula on a dense (theta, phi) grid (sympy-generated, numpy-evaluated);
  2. finite differences of the parametrization, refined around the grid minimizer.
A body with K_min > 0 everywhere is strictly convex. A mesh convex-hull volume test cannot
detect curvature violations of the size found here, which is why this check exists.

Usage:  python check_strict_convexity.py
Needs:  numpy, sympy
"""

import math
import numpy as np
import sympy as sp

INSTANCES = [
    ("primary", 1, 0.023149, 0.234433),
    ("second", 2, 0.0321, 0.137606),
    ("third", 3, 0.051692, -0.055157),
]

x, y, z, be, a = sp.symbols("x y z be a", real=True)


def implicit_curvature_function(k):
    r = sp.sqrt(x**2 + y**2 + z**2)
    w = z / r
    eta = sp.Rational(3, 2) * sp.pi * (w - w**3 / 3)
    P = eta + a * sp.sin(k * eta)
    s = (x * sp.cos(P) + y * sp.sin(P)) / r
    F = r**4 - 1 - 4 * be * s
    V = (x, y, z)
    g = [sp.diff(F, v) for v in V]
    H = [[sp.diff(g[i], V[j]) for j in range(3)] for i in range(3)]
    flat = g + [H[0][0], H[0][1], H[0][2], H[1][1], H[1][2], H[2][2]]
    f = sp.lambdify((x, y, z, be, a), flat, "numpy", cse=True)

    def K(px, py, pz, b, aa):
        g1, g2, g3, h11, h12, h13, h22, h23, h33 = f(px, py, pz, b, aa)
        a11 = h22 * h33 - h23 * h23
        a22 = h11 * h33 - h13 * h13
        a33 = h11 * h22 - h12 * h12
        a12 = -(h12 * h33 - h13 * h23)
        a13 = h12 * h23 - h13 * h22
        a23 = -(h11 * h23 - h12 * h13)
        num = (g1 * (a11 * g1 + a12 * g2 + a13 * g3) + g2 * (a12 * g1 + a22 * g2 + a23 * g3)
               + g3 * (a13 * g1 + a23 * g2 + a33 * g3))
        n2 = g1 * g1 + g2 * g2 + g3 * g3
        return num / (n2 * n2)

    return K


def eta_np(t):
    return 1.5 * math.pi * (np.cos(t) - np.cos(t) ** 3 / 3)


def radius(t, p, k, b, aa):
    P = math.pi * 1.5 * (math.cos(t) - math.cos(t) ** 3 / 3)
    P = P + aa * math.sin(k * P)
    return (1 + 4 * b * math.sin(t) * math.cos(p - P)) ** 0.25


def point(t, p, k, b, aa):
    r = radius(t, p, k, b, aa)
    return np.array([r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t)])


def fd_curvature(t, p, k, b, aa, h=1e-4):
    f = lambda tt, pp: point(tt, pp, k, b, aa)
    Xt = (f(t + h, p) - f(t - h, p)) / (2 * h)
    Xp = (f(t, p + h) - f(t, p - h)) / (2 * h)
    Xtt = (f(t + h, p) - 2 * f(t, p) + f(t - h, p)) / h**2
    Xpp = (f(t, p + h) - 2 * f(t, p) + f(t, p - h)) / h**2
    Xtp = (f(t + h, p + h) - f(t + h, p - h) - f(t - h, p + h) + f(t - h, p - h)) / (4 * h * h)
    N = np.cross(Xt, Xp)
    N = N / np.linalg.norm(N)
    E, F, G = Xt @ Xt, Xt @ Xp, Xp @ Xp
    return ((Xtt @ N) * (Xpp @ N) - (Xtp @ N) ** 2) / (E * G - F * F)


def main():
    th = np.linspace(1e-3, math.pi - 1e-3, 500)
    ph = np.linspace(0, 2 * math.pi, 1000, endpoint=False)
    T, Pp = np.meshgrid(th, ph, indexing="ij")
    print(f"{'instance':8s} {'beta':>9s} {'coeff':>10s} {'K_min grid':>11s} {'K_min FD':>9s} {'% surface K<0':>14s}  strictly convex")
    for name, k, b, aa in INSTANCES:
        K = implicit_curvature_function(k)
        P = eta_np(T) + aa * np.sin(k * eta_np(T))
        r = (1 + 4 * b * np.sin(T) * np.cos(Pp - P)) ** 0.25
        Kv = K(r * np.sin(T) * np.cos(Pp), r * np.sin(T) * np.sin(Pp), r * np.cos(T), b, aa)
        i = np.unravel_index(np.argmin(Kv), Kv.shape)
        t0, p0 = T[i], Pp[i]
        best = min(fd_curvature(t0 + dt, p0 + dp, k, b, aa)
                   for dt in np.linspace(-0.03, 0.03, 31) for dp in np.linspace(-0.03, 0.03, 31))
        frac = 100 * float(np.mean(Kv < 0))
        print(f"{name:8s} {b:9.6f} {aa:10.6f} {Kv.min():11.4f} {best:9.4f} {frac:13.2f}%  {'yes' if Kv.min() > 0 else 'no'}")


if __name__ == "__main__":
    main()
