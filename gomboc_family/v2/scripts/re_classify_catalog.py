"""
re_classify_catalog.py — re-screen all 13 catalog members from v1 under the
strict-convexity gate, with analytical h(d) basin count as the equilibrium
metric (rather than the v1 mesh-vertex min basin counter).

Three classification buckets:
    'strictly-convex'      : K_min > 0 AND analytical ECS = 1
    'convexity-marginal'   : K_min > 0 BUT analytical ECS > 1
                             (or K_min close to 0; threshold configurable)
    'not-strictly-convex'  : K_min <= 0
                             (these CANNOT be mono-monostatic under Varkonyi
                              identity in its precondition-satisfied form; the
                              "ECS=1" reportage in v1 is an artifact of the
                              min-over-mesh-vertices recipe)

Reads from L:/Research Programs/Substrate-Geometry/gomboc_family/results/catalog.json
Emits   to ./results/catalog_v2_classification.json

Usage:
    python re_classify_catalog.py
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strict_convexity_gate as scg
import h_smooth_analytical as hsa


V1_CATALOG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "results", "catalog.json"))
OUT_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "results", "catalog_v2_classification.json"))


def family_for(entry):
    """Map v1 catalog 'family' + 'perturbation_type' to v2 R_factory family."""
    fam_v1 = entry["family"]
    ptype = entry["perturbation_type"]
    if fam_v1 == "phase":
        # Phase-1: 'phase' (k=1), Phase-2: 'phase2', Phase-3: 'phase3'
        if ptype == "phase":
            return ("sloan-phase", 1, None)
        if ptype == "phase2":
            return ("sloan-phase", 2, None)
        if ptype == "phase3":
            return ("sloan-phase", 3, None)
    if fam_v1 == "radial":
        # ptype is one of f1, f2, f3, f4
        return ("sloan-radial", 1, ptype)
    raise ValueError(f"unknown entry family/ptype: {fam_v1}/{ptype}")


def classify_one(entry, n_t_K=160, n_p_K=320, n_dirs_h=400, n_starts_h=24, depth_thresh=1e-5):
    family, k, f_type = family_for(entry)
    beta = entry["beta"]
    coef = entry["coefficient"]

    # Strict-convexity gate
    cv = scg.compute(family, beta, coef, k, n_t_K, n_p_K, eps_t=0.04,
                     f_type=f_type or "f4")

    # Analytical h(d) basin count
    h_res = hsa.compute(family, beta, coef, k, n_dirs_h, n_starts_h, depth_thresh,
                        f_type=f_type or "f4")

    # Classification bucket
    if cv["K_min"] > 0:
        if h_res["n_minima"] == 1:
            bucket = "strictly-convex-mono-monostatic"
        else:
            bucket = "convexity-marginal"
    else:
        bucket = "not-strictly-convex"

    return {
        "name": entry["name"],
        "v1_family": entry["family"],
        "v1_perturbation_type": entry["perturbation_type"],
        "beta": beta,
        "coefficient": coef,
        "v1_ecs": entry["ecs"],
        "v1_convexity": entry["convexity"],
        "v2_K_min": cv["K_min"],
        "v2_K_max": cv["K_max"],
        "v2_fraction_K_negative": cv["fraction_K_negative"],
        "v2_strict_convexity_pass": cv["gate_pass"],
        "v2_analytical_h_minima": h_res["n_minima"],
        "v2_analytical_h_maxima": h_res["n_maxima"],
        "v2_h_extent": h_res["h_extent"],
        "v2_min_gap": h_res["gap_between_two_deepest_minima"],
        "v2_bucket": bucket,
    }


def main():
    with open(V1_CATALOG) as f:
        v1 = json.load(f)
    results = []
    t0 = time.time()
    for i, entry in enumerate(v1["catalog"], 1):
        print(f"[{i}/{len(v1['catalog'])}] {entry['name']} ...", file=sys.stderr)
        try:
            r = classify_one(entry)
            print(f"   -> K_min={r['v2_K_min']:+.4f}  frac<0={r['v2_fraction_K_negative']:.4f}  "
                  f"h_minima={r['v2_analytical_h_minima']}  bucket={r['v2_bucket']}", file=sys.stderr)
            results.append(r)
        except Exception as e:
            print(f"   FAILED: {e}", file=sys.stderr)
            results.append({"name": entry["name"], "error": str(e)})
    dt = time.time() - t0
    # Summary
    buckets = {}
    for r in results:
        b = r.get("v2_bucket", "error")
        buckets[b] = buckets.get(b, 0) + 1
    summary = {
        "n_total": len(results),
        "by_bucket": buckets,
        "compute_seconds": dt,
    }
    out = {"summary": summary, "entries": results}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {OUT_PATH}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
