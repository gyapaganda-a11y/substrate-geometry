"""
diagnostic_phase1_sensitivity.py — investigate whether Phase-1's h_minima=3
finding is (a) a real geometric feature (sub-grid K<0 pockets), or (b) a
numerical artifact of the analytical h finder.

Tests:
  1. Refine K grid 160x320 -> 320x640 -> 640x1280 and report K_min at each.
     If K_min stays positive at all refinements, body is strictly convex and
     the h_minima > 1 must come from numerical noise (or my interpretation
     of the identity is wrong).
  2. Run analytical h with depth_thresh in {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}
     and n_dirs in {300, 600, 1200} to see how n_minima scales. If the
     extra minima depend on threshold/density, they are numerical features
     of the discretization. If they are stable, they are real.
  3. Compare the heights of the reported minima to see whether they cluster
     into a single deep minimum + numerical-floor flutter.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strict_convexity_gate as scg
import h_smooth_analytical as hsa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/phase1_sensitivity.json")
    args = p.parse_args()

    beta, coef = 0.023149, 0.234433
    family, k = "sloan-phase", 1

    out = {"phase1_params": {"beta": beta, "coef": coef}}

    # 1. K refinement
    print("=== K refinement ===")
    K_refine = []
    for (n_t, n_p) in [(160, 320), (320, 640), (640, 1280)]:
        r = scg.compute(family, beta, coef, k, n_t, n_p, eps_t=0.04)
        rec = {"n_t": n_t, "n_p": n_p, "K_min": r["K_min"], "K_max": r["K_max"], "frac_neg": r["fraction_K_negative"]}
        K_refine.append(rec)
        print(f"  {n_t}x{n_p}: K_min={r['K_min']:+.6f}  K_max={r['K_max']:+.6f}  frac<0={r['fraction_K_negative']:.6f}")
    out["K_refinement"] = K_refine

    # 2. h sensitivity to threshold and direction density
    print("\n=== h sensitivity ===")
    h_sens = []
    for n_dirs in [300, 600, 1200]:
        r_h = hsa.compute(family, beta, coef, k, n_dirs=n_dirs, n_starts=40,
                          depth_thresh=1e-12)  # collect at very tight, then post-filter
        heights = np.array(r_h["all_heights"])
        dirs = np.array(r_h["directions"])
        from scipy.spatial import cKDTree
        tree = cKDTree(dirs)
        _, nbrs = tree.query(dirs, k=21)
        for thresh in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            n_min = 0
            n_max = 0
            for i in range(len(dirs)):
                nbr_h = heights[nbrs[i, 1:]]
                if heights[i] < nbr_h.min() - thresh:
                    n_min += 1
                if heights[i] > nbr_h.max() + thresh:
                    n_max += 1
            rec = {"n_dirs": n_dirs, "depth_thresh": thresh, "n_minima": n_min, "n_maxima": n_max,
                   "h_range": [float(heights.min()), float(heights.max())]}
            h_sens.append(rec)
            print(f"  n_dirs={n_dirs}  thresh={thresh:.0e}: minima={n_min}  maxima={n_max}")
    out["h_sensitivity"] = h_sens

    # 3. Distribution of minima heights at one config
    print("\n=== minima heights at default config ===")
    r_default = hsa.compute(family, beta, coef, k, n_dirs=600, n_starts=40, depth_thresh=1e-5)
    out["default_run"] = {
        "n_minima": r_default["n_minima"],
        "n_maxima": r_default["n_maxima"],
        "minima_heights_sorted": r_default["minima_heights_sorted"],
        "maxima_heights_sorted": r_default["maxima_heights_sorted"],
        "h_extent": r_default["h_extent"],
    }
    if r_default["minima_heights_sorted"]:
        mh = r_default["minima_heights_sorted"]
        print(f"  n_minima={r_default['n_minima']}, heights={[f'{h:.8f}' for h in mh[:10]]}")
        if len(mh) >= 2:
            print(f"  gap min1->min2: {mh[1]-mh[0]:.2e}")
            print(f"  gap min1->minN: {mh[-1]-mh[0]:.2e}")
            print(f"  h_extent:      {r_default['h_extent']:.4e}")
            print(f"  gap/extent ratio for min1->min2: {(mh[1]-mh[0])/r_default['h_extent']:.4%}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
