
"""
    cl_sop(sop_dgp, lam, L0, cl_init, d1, d2;
           reps_final, reps_bracket, bracket_step,
           arl_truncation_factor, chart_choice, refinement,
           verbose, cl_tol, seed)

Compute the critical limit `cl` for a EWMA control chart such that the
in-control Average Run Length (ARL) equals `L0`.

The search proceeds in two phases:
1. **Bracketing**: A coarse Monte Carlo estimate (using `reps_bracket` replications)
   steps away from `cl_init` in increments of `bracket_step` until an interval
   `[a, b]` is found where the ARL crosses `L0`.
2. **Refinement**: The ITP root-finding algorithm narrows the bracket to
   within `cl_tol` using full Monte Carlo estimates (`reps_final` replications).

Using a fixed `seed` ensures the ARL function behaves smoothly across
evaluations within a single call, which is required for the root finder to
work reliably. With large `reps_final`, the result is accurate regardless of
which seed is used. Pass `seed=nothing` (default) for independent results
across calls, or fix the seed for reproducibility.

# Arguments
- `sop_dgp::ICSTS`: Data-generating process under the in-control distribution.
- `lam`: EWMA smoothing parameter λ ∈ (0, 1].
- `L0`: Target in-control ARL.
- `cl_init`: Initial guess for the critical limit. Does not need to be precise —
  a rough value in the right ballpark is sufficient.
- `d1`, `d2`: Integer parameters passed to the ARL simulation.

# Keyword Arguments
- `reps_final=10_000`: Replications used during the ITP refinement phase.
  Increase for a more accurate result.
- `reps_bracket=1_000`: Replications used during the bracketing phase.
  Fewer replications are sufficient here since only the sign of `ARL - L0` matters.
- `bracket_step=0.01`: Step size for the bracketing search.
- `arl_truncation_factor=50`: Individual simulation runs are capped at
  `arl_truncation_factor * L0` steps. Prevents excessive compute when `cl` is
  far from the root during bracketing.
- `chart_choice=TauTilde()`: Control chart statistic to use.
- `refinement=false`: Whether to apply a refined chart computation.
- `cl_tol=1e-4`: Absolute convergence tolerance on `cl` for the ITP phase.
- `seed=nothing`: Random seed for reproducibility. If `nothing`, a fresh seed
  is drawn each call. Fix to an integer (e.g. `seed=42`) to get the same
  result across runs.
- `verbose=false`: If `true`, prints `cl` and ARL at each function evaluation.

# Returns
- `cl::Float64`: The critical limit achieving an in-control ARL of `L0`.

# Example
```julia
cl = cl_sop(dgp, 0.1, 370.0, 2.5, 3, 5; reps_final=50_000, seed=42)
```
"""
function cl_sop(
    sop_dgp::ICSTS, lam, L0, cl_init, d1::Int, d2::Int;
    reps_final=10_000,
    reps_bracket=1_000,
    bracket_step=0.001,
    arl_truncation_factor=50,
    chart_choice=TauTilde(),
    refinement=false,
    verbose=false,
    cl_tol=1e-4,
    seed=nothing
)
    # Evaluates ARL(cl) via MC simulation; 
    # fixed seed ensures comparable noise across calls.
    function get_arl(cl, current_reps, current_truncate)
        Random.seed!(seed)
        res = arl_sop_ic(
            sop_dgp, lam, cl, d1, d2, current_reps;
            chart_choice=chart_choice,
            refinement=refinement,
            rl_max=current_truncate
        )
        if verbose
            println("  cl = ", round(cl, digits=6), " | ARL = ", round(res[1], digits=4))
        end
        return res[1]
    end

    # Ensure we have a seed for reproducibility.
    seed = isnothing(seed) ? rand(Int) : seed

    # Cap ARL runs during bracketing to avoid wasting reps far from the root.
    trunc_val = arl_truncation_factor * L0 # upper limit for ARL estimates
    cap_val = trunc_val / 10 # if ARL hits this during bracketing, we consider it "too large" and shift cl downwards

    # Step 1: Find an interval [a, b] that brackets the root using coarse MC.
    if verbose
        println("\n" * "="^60)
        println(" STEP 1: Bracket Search")
        println(" reps = $reps_bracket | truncation cap = $trunc_val | step size = $bracket_step")
        println("="^60)
    end

    # Start bracket search from cl_init.
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




# """
#     cl_sop(
#   sop_dgp::ICSTS, lam, L0, cl_init, d1::Int, d2::Int, reps=10_000;
#   chart_choice::InformationMeasure=TauTilde(), jmin=4, jmax=6, verbose=false
# )

# Compute the control limit for a given in-control process. The input parameters are:

# - `sop_dgp`: The in-control spatial process (ICSTS) to use for the control limit.
# - `lam::Float64`:  A scalar value for lambda for the EWMA chart.
# - `L0::Float64`: The desired average run length.
# - `cl_init::Float64`: The initial value for the control limit.
# - `d1::Int`: The first (row) delay for the spatial process.
# - `d2::Int`: The second (column) delay for the spatial process.
# - `reps::Int`: The number of replications to compute the ARL.
# - `chart_choice::Int`: The chart choice for the SOP chart.
# - `jmin`: The minimum number of values to change after the decimal point in the control limit.
# - `jmax`: The maximum number of values to change after the decimal point in the control limit.
# - `verbose::Bool`: A boolean to indicate whether to print the control limit and ARL for each iteration.
# """
# function cl_sop(
#     sop_dgp::ICSTS, lam, L0, cl_init, d1::Int, d2::Int, reps=10_000;
#     chart_choice=TauTilde(), refinement::Union{Bool,RefinedType}=false,
#     jmin=4, jmax=6,
#     verbose=false,
#     truncate_val::Int=typemax(Int)
# )

#     L1 = 0.0

#     for j in jmin:jmax
#         for dh in 1:80
#             cl_init = cl_init + (-1)^j * dh / 10^j
#             L1 = arl_sop_ic(
#                 sop_dgp, lam, cl_init, d1, d2, reps;
#                 chart_choice=chart_choice, refinement=refinement, truncate_val=truncate_val
#             )[1]
#             if verbose
#                 println("cl = ", cl_init, "\t", "ARL = ", L1)
#             end
#             if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
#                 break
#             end
#         end
#         cl_init = cl_init
#     end

#     if L1 < L0
#         cl_init = cl_init + 1 / 10^jmax
#     end

#     return cl_init
# end
