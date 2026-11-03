# Preconditions Check — Catalog Paper v2

Every theorem the paper applies has preconditions; this document enumerates them, names the body classes the theorem is applied to, and points at the script that tests pass/fail on each.

This artifact is the institutional fix for the v1 failure mode where Sloan's strict-convexity precondition was inherited silently. Technical correspondence with P. L. Várkonyi (2026-05-15) surfaced the gap.

---

## Theorem A: Sloan (2023, arXiv:2306.14914) — radial-critical-point analysis

**Statement:** For the analytical family `r^4 = 1 + 4β sin θ cos(φ − η(θ))`, the radial function R(θ, φ) has exactly two critical points on S².

**Preconditions (as stated in Sloan's analysis):**
1. The function R is smooth (C² sufficient).
2. The parameterization is well-defined (r > 0 everywhere; i.e., 1 + 4β sin θ cos(...) > 0 for all (θ, φ)).
3. The body is strictly convex — required for the conclusion "two R-critical-points implies two h-critical-points" to follow via Gauss-map bijection. **(This precondition is implicit in Sloan and was the silently-violated one in v1.)**

**Bodies the paper applies Theorem A to:** every Sloan-family body the paper studies, including the original Sloan β=0.05 and β=0.08 reference cases, plus all 13 v1 catalog members which are extensions of Sloan's parameterization.

**Test scripts:**

| Precondition | Script | Pass criterion |
|---|---|---|
| Smoothness (C²) | analytical (closed form) | R is real-analytic where r⁴ > 0 |
| r > 0 globally | `scripts/strict_convexity_gate.py` (asserts internally) | r⁴ > 0 on grid; reports failure if not |
| **Strict convexity** | `scripts/strict_convexity_gate.py` | `min(K) > 0` on dense (θ, φ) grid |

**Status of bodies under test (see `results/catalog_v2_classification.json`):**
- Sloan β = 0.02: PASS (K_min ≥ 0)
- Sloan β = 0.03: PASS (K_min ≥ 0)
- Sloan β = 0.05: **FAIL** (K_min = −0.57, K<0 over 4.0% of surface)
- Sloan β = 0.08: **FAIL** (K_min = −2.20, K<0 over 8.7% of surface)
- v1 catalog: see classification table; bucket `strictly-convex-mono-monostatic` passes, `convexity-marginal` or `not-strictly-convex` fail.

---

## Theorem B: Várkonyi gradient identity (textbook convex-body geometry)

**Statement:** For a smooth strictly convex compact body K ⊂ ℝ³ with COM at origin, the radial function R(u) and support function H(n) satisfy
∇_S H(n) = − R(u) ∇_S R(u) / √(R² + ‖∇_S R‖²)
under the Gauss map n = n(u). Therefore ∇_S H = 0 iff ∇_S R = 0.

**Preconditions:**
1. Smooth (C² sufficient).
2. **Strictly convex** (Gauss map is a diffeomorphism on S²).
3. COM at origin (the identity is stated in COM-frame; if COM is elsewhere, replace H with H − ⟨COM, n⟩).

**Bodies the paper applies Theorem B to:** ALL bodies in the paper, both the Sloan reference cases and every v1 catalog member.

**Test scripts:**

| Precondition | Script | Pass criterion |
|---|---|---|
| Smoothness | analytical | C∞ where defined |
| Strict convexity | `scripts/strict_convexity_gate.py` | `min(K) > 0` |
| COM at origin | `scripts/h_smooth_analytical.py` `estimate_com()` | `‖com‖ < 1e-6` |

---

## Theorem C: Domokos-Várkonyi (2006) — mono-monostatic existence

**Statement:** A smooth strictly convex body with exactly one stable rest position and exactly one unstable rest position exists; the original Gömböc construction is one such.

**Preconditions:**
1. Smooth.
2. **Strictly convex.**
3. COM-frame critical-point count of H equals 2 (one minimum + one maximum on S²).

**Bodies the paper claims to satisfy Theorem C:** any catalog member labeled `strictly-convex-mono-monostatic` after v2 re-classification. Membership in this bucket requires BOTH the strict-convexity gate AND the analytical h(d) basin count of 1.

**Test scripts:**

| Precondition | Script | Pass criterion |
|---|---|---|
| Smooth | analytical | C∞ |
| Strict convexity | `scripts/strict_convexity_gate.py` | `min(K) > 0` |
| ECS = 1 on h-analytical | `scripts/h_smooth_analytical.py` | `n_minima == 1 AND n_maxima == 1` |

---

## What the paper does NOT do anymore (v1 failure modes retired)

- v1 used `convexity = mesh_vol / hull_vol > 0.999` as the convexity check. This is satisfied by bodies with small K<0 regions. v2 retires this metric for theorem-precondition purposes (it remains useful as a sanity check on mesh quality, but is not the strict-convexity gate).
- v1 §4.3 claimed "we computed h(d) analytically at 2000 directions via multi-start" with no script implementing it. v2 implements it as `scripts/h_smooth_analytical.py` and points every quantitative claim at that script via `claims_to_scripts.md`.
- v1 inferred mono-monostatic status from a mesh-vertex basin counter on a structured (θ, φ) mesh. Structured meshes have pole/stripe artifacts. v2 uses Fibonacci-sphere direction sampling for the analytical h(d) computation, which is artifact-free.
