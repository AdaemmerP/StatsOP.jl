
"""
    cl_sacf_bp(sp_dgp::ICSTS, lam, L0, cl_init, w::Int;
               reps_final, reps_bracket, bracket_step,
               arl_truncation_factor, verbose, cl_tol, seed)

Compute the critical limit `cl` for a BP-EWMA control chart based on the spatial
autocorrelation function (SACF) such that the in-control ARL equals `L0`.

The search proceeds in two phases:
1. **Bracketing**: A coarse Monte Carlo estimate (using `reps_bracket` replications)
   steps away from `cl_init` in increments of `bracket_step` until an interval
   `[a, b]` is found where the ARL crosses `L0`.
2. **Refinement**: The ITP root-finding algorithm narrows the bracket to
   within `cl_tol` using full Monte Carlo estimates (`reps_final` replications).

Using a fixed `seed` ensures the ARL function behaves smoothly across
evaluations within a single call. With large `reps_final`, the result is accurate
regardless of which seed is used. Pass `seed=nothing` (default) for independent
results across calls, or fix the seed for reproducibility.

# Arguments
- `sp_dgp::ICSTS`: Data-generating process under the in-control distribution.
- `lam`: EWMA smoothing parameter λ ∈ (0, 1].
- `L0`: Target in-control ARL.
- `cl_init`: Initial guess for the critical limit.
- `w::Int`: Window size for the BP-statistic.

# Keyword Arguments
- `reps_final=10_000`: Replications used during the ITP refinement phase.
- `reps_bracket=1_000`: Replications used during the bracketing phase.
- `bracket_step=0.01`: Step size for the bracketing search.
- `arl_truncation_factor=50`: Individual simulation runs are capped at
  `arl_truncation_factor * L0` steps during bracketing.
- `cl_tol=1e-4`: Absolute convergence tolerance on `cl` for the ITP phase.
- `seed=nothing`: Random seed for reproducibility.
- `verbose=false`: If `true`, prints progress information at each evaluation.

# Returns
- `cl::Float64`: The critical limit achieving an in-control ARL of `L0`.

# Example
```julia
sp_dgp = ICSTS(20, 20, Normal(0, 1))
cl = cl_sacf_bp(sp_dgp, 0.1, 370.0, 0.5, 3; reps_final=50_000, seed=42)
```
"""
function cl_sacf_bp(
    sp_dgp::ICSTS, lam, L0, cl_init, w::Int;
    reps_final=10_000,
    reps_bracket=1_000,
    bracket_step=0.01,
    arl_truncation_factor=50,
    verbose=false,
    cl_tol=1e-4,
    seed=nothing
)
    # Evaluates ARL(cl) via MC simulation;
    # fixed seed ensures comparable noise across calls.
    function get_arl(cl, current_reps, current_truncate)
        Random.seed!(seed)
        res = arl_sacf_bp_ic(sp_dgp, lam, cl, w, current_reps; rl_max=current_truncate)
        if verbose
            println("  cl = ", round(cl, digits=6), " | ARL = ", round(res[1], digits=4))
        end
        return res[1]
    end

    # Ensure we have a seed for reproducibility.
    seed = isnothing(seed) ? rand(Int) : seed

    # Cap ARL runs during bracketing to avoid wasting reps far from the root.
    trunc_val = arl_truncation_factor * L0
    cap_val = trunc_val / 10

    # Step 1: Find an interval [a, b] that brackets the root using coarse MC.
    if verbose
        println("\n" * "="^60)
        println(" STEP 1: Bracket Search")
        println(" reps = $reps_bracket | truncation cap = $trunc_val | step size = $bracket_step")
        println("="^60)
    end

    a = cl_init
    f_a = get_arl(a, reps_bracket, trunc_val) - L0

    # If cl_init hits the truncation cap, shift down until we get a real ARL estimate.
    if f_a >= cap_val
        if verbose
            println("  [!] cl_init hits truncation cap — shifting down:")
        end
        while f_a >= cap_val
            a -= bracket_step
            f_a = get_arl(a, reps_bracket, trunc_val) - L0
            if verbose
                println("      → cl = ", round(a, digits=6), " | ARL - L0 = ", round(f_a, digits=4), " (cap = $cap_val)")
            end
        end
    end

    direction = (f_a < 0) ? 1.0 : -1.0
    b = a + (direction * bracket_step)
    f_b = get_arl(b, reps_bracket, trunc_val) - L0

    search_iter = 0
    while f_a * f_b > 0
        b += (direction * bracket_step)
        f_b = get_arl(b, reps_bracket, trunc_val) - L0
        search_iter += 1
        if search_iter > 100
            error("Could not find a bracket. Check if cl_init is reasonable.")
        end
    end

    bracket = (min(a, b), max(a, b))
    if verbose
        println("-"^60)
        println(" Bracket found: [", round(bracket[1], digits=6), ", ", round(bracket[2], digits=6), "]")
        println("-"^60)
    end

    # Step 2: Refine to the root via ITP using full reps and no truncation.
    if verbose
        println("\n" * "="^60)
        println(" STEP 2: Root Finding via ITP")
        println(" reps = $reps_final | no truncation | tolerance = $cl_tol")
        println("="^60)
    end
    final_cl = find_zero(
        cl -> get_arl(cl, reps_final, typemax(Int)) - L0,
        bracket,
        Roots.ITP(),
        xatol=cl_tol,
        verbose=verbose
    )

    return final_cl
end
