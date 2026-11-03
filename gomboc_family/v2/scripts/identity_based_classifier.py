"""
identity_based_classifier.py — the rigorous v2 ECS classifier, based on
Varkonyi's identity rather than mesh-vertex basin counting.

Logic:
  For a smooth radial body with COM at origin, IF the body is strictly
  convex (K(theta,phi) > 0 everywhere) AND R has exactly N critical points
  on S^2, THEN by Varkonyi's gradient identity h has exactly N critical
  points on S^2.

  Mono-monostatic <=> ECS = 1 (one local minimum + one local maximum) <=>
  R has exactly 2 critical points <=> N_critical(R) = 2 AND body is
  strictly convex.

Classification:
  - 'strictly-convex-mono-monostatic': K_min > 0 AND N_critical(R) = 2
  - 'strictly-convex-multi-equilibrium': K_min > 0 AND N_critical(R) != 2
  - 'not-strictly-convex': K_min <= 0
                           (mono-monostatic status indeterminate by identity)

An additional analytical h(d) computation is run as a SANITY check: if the
identity predicts N_critical = 2 but h_analytical finds something else, the
disagreement flags either (a) sub-grid K<0 pockets (refine K grid), or (b)
numerical issues in h finder (more starts, finer dirs). Where the identity
and h_analytical agree, ECS is reported confidently.

This replaces the v1 min-over-mesh-vertices ECS count entirely.

Usage:
    python identity_based_classifier.py
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strict_convexity_gate as scg
import count_R_critpoints as crp
import h_smooth_analytical as hsa


V1_CATALOG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "results", "catalog.json"))
OUT_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "results", "catalog_v2_identity_based.json"))


def family_for(entry):
    fam_v1 = entry["family"]
    ptype = entry["perturbation_type"]
    if fam_v1 == "phase":
        if ptype == "phase":   return ("sloan-phase", 1, None)
        if ptype == "phase2":  return ("sloan-phase", 2, None)
        if ptype == "phase3":  return ("sloan-phase", 3, None)
    if fam_v1 == "radial":
        return ("sloan-radial", 1, ptype)
    raise ValueError(f"unknown entry: {fam_v1}/{ptype}")


def classify_one(entry, n_t_K=320, n_p_K=640, n_starts_R=500, n_dirs_h=600, n_starts_h=40, depth_thresh=1e-5):
    family, k, f_type = family_for(entry)
    beta = entry["beta"]
    coef = entry["coefficient"]

    # 1. K gate on a fine grid
    cv = scg.compute(family, beta, coef, k, n_t_K, n_p_K, eps_t=0.04, f_type=f_type or "f4")

    # 2. R critical-point count
    R_fn = scg.R_factory(family, beta, coef, k, f_type or "f4")
    R_crits = crp.find_critical_points(R_fn, n_starts=n_starts_R)
    n_crit_R = len(R_crits)

    # 3. Analytical h sanity-check
    h_res = hsa.compute(family, beta, coef, k, n_dirs_h, n_starts_h, depth_thresh,
                        f_type=f_type or "f4")

    # Classification
    if cv["K_min"] > 0:
        if n_crit_R == 2:
            bucket = "strictly-convex-mono-monostatic"
        else:
            bucket = "strictly-convex-multi-equilibrium"
    else:
        bucket = "not-strictly-convex"

    # Identity sanity check: under strict convexity, R-crit-count == h-crit-count
    predicted_h_crits = n_crit_R if cv["K_min"] > 0 else None
    observed_h_crits = h_res["n_minima"] + h_res["n_maxima"]
    identity_disagreement = (
        predicted_h_crits is not None
        and predicted_h_crits != observed_h_crits
    )

    return {
        "name": entry["name"],
        "v1_family": entry["family"],
        "v1_perturbation_type": entry["perturbation_type"],
        "beta": beta,
        "coefficient": coef,
        "v1_ecs": entry["ecs"],
        "v2_K_min": cv["K_min"],
        "v2_K_max": cv["K_max"],
        "v2_fraction_K_negative": cv["fraction_K_negative"],
        "v2_n_critical_R": n_crit_R,
        "v2_h_analytical_n_minima": h_res["n_minima"],
        "v2_h_analytical_n_maxima": h_res["n_maxima"],
        "v2_h_extent": h_res["h_extent"],
        "v2_predicted_h_crits_under_identity": predicted_h_crits,
        "v2_observed_h_crits": observed_h_crits,
        "v2_identity_disagreement": identity_disagreement,
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
            print(f"   -> K_min={r['v2_K_min']:+.4f}  N_R_crit={r['v2_n_critical_R']}  "
                  f"h_min={r['v2_h_analytical_n_minima']}+max={r['v2_h_analytical_n_maxima']}  "
                  f"bucket={r['v2_bucket']}  disagree={r['v2_identity_disagreement']}",
                  file=sys.stderr)
            results.append(r)
        except Exception as e:
            print(f"   FAILED: {e}", file=sys.stderr)
            results.append({"name": entry["name"], "error": str(e)})
    dt = time.time() - t0
    buckets = {}
    for r in results:
        b = r.get("v2_bucket", "error")
        buckets[b] = buckets.get(b, 0) + 1
    summary = {
        "n_total": len(results),
        "by_bucket": buckets,
        "n_identity_disagreements": sum(1 for r in results if r.get("v2_identity_disagreement")),
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
