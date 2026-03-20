"""
    cl_op(op_dgp, lam, L0, cl_init;
          reps_final, reps_bracket, bracket_step,
          arl_truncation_factor, chart_choice, d, m, ced, ad,
          verbose, cl_tol, seed)

Compute the critical limit `cl` for an EWMA control chart based on ordinal patterns
such that the in-control Average Run Length (ARL) equals `L0`.

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
- `op_dgp::Union{ContinuousDGPIC,DiscreteDGPIC}`: Data-generating process under the
  in-control distribution.
- `lam`: EWMA smoothing parameter λ ∈ (0, 1].
- `L0`: Target in-control ARL.
- `cl_init`: Initial guess for the critical limit. Does not need to be precise —
  a rough value in the right ballpark is sufficient.

# Keyword Arguments
- `reps_final=10_000`: Replications used during the ITP refinement phase.
- `reps_bracket=1_000`: Replications used during the bracketing phase.
- `bracket_step=0.01`: Step size for the bracketing search.
- `arl_truncation_factor=50`: Individual simulation runs are capped at
  `arl_truncation_factor * L0` steps during bracketing.
- `chart_choice`: Control chart statistic to use. Required.
- `d::Union{Int,Vector{Int}}=1`: Delay value or vector.
- `m::Int=3`: Pattern length.
- `ced::Bool=false`: Use conditional expected delay?
- `ad::Int=100`: Number of iterations for ced.
- `cl_tol=1e-4`: Absolute convergence tolerance on `cl` for the ITP phase.
- `seed=nothing`: Random seed for reproducibility. Fix to an integer (e.g. `seed=42`)
  to get the same result across runs.
- `verbose=false`: If `true`, prints progress information at each evaluation.

# Returns
- `cl::Float64`: The critical limit achieving an in-control ARL of `L0`.

# Example
```julia
op_dgp = ContinuousDGPIC(Normal(0, 1))
cl = cl_op(op_dgp, 0.1, 370.0, 2.5; chart_choice=TauTilde(), reps_final=50_000, seed=42)
```
"""
function cl_op(
    op_dgp, lam, L0, cl_init;
    reps_final=10_000,
    reps_bracket=1_000,
    bracket_step=0.01,
    arl_truncation_factor=50,
    chart_choice,
    d=1,
    m=3,
    ced=false,
    ad=100,
    verbose=false,
    cl_tol=1e-4,
    seed=nothing
)
    # For lower-sided charts (Shannon, ShannonExtropy), ARL is decreasing in cl.
    is_lower_sided = chart_choice isa Shannon || chart_choice isa ShannonExtropy

    # Evaluates ARL(cl) via MC simulation;
    # fixed seed ensures comparable noise across calls.
    function get_arl(cl, current_reps, current_truncate)
        Random.seed!(seed)
        res = arl_op_ic(
            op_dgp, lam, cl, current_reps;
            chart_choice=chart_choice, d=d, m=m, ced=ced, ad=ad,
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
    trunc_val = arl_truncation_factor * L0
    cap_val = trunc_val / 10

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

    # For upper-sided charts: if cl_init yields very high ARL, shift down.
    if !is_lower_sided && f_a >= cap_val
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

    # For lower-sided charts (Shannon), ARL decreases with cl → reverse direction.
    direction = if is_lower_sided
        (f_a < 0) ? -1.0 : 1.0
    else
        (f_a < 0) ? 1.0 : -1.0
    end

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
#     cl_op(op_dgp, lam, L0, cl_init, reps=10_000; chart_choice, jmin=4, jmax=6, verbose=false, d=1, m=3, ced=false, ad=100)
#
# Function to compute the control limit using in-control processes using ordinal patterns.
#
# - `op_dgp::ICTS.
# - `lam::Float64`: Smoothing parameter for the EWMA statistic.
# - `L0::Float64`: In-control ARL.
# - `cl_init::Float64`: Initial guess for the control limit.
# - `reps::Int64`: Number of replications.
# - `chart_choice::Int`
#   1. ``\\widehat{H}^{(d)}=-\\sum_{k=1}^{m!} \\hat{p}_k{ }^{(d)} \\ln \\hat{p}_k{ }^{(d)}``
#   2. ``\\widehat{H}_{\\mathrm{ex}}^{(d)}=-\\sum_{k=1}^{m!}\\left(1-\\hat{p}_k{ }^{(d)}\\right) \\ln \\left(1-\\hat{p}_k{ }^{(d)}\\right)``
#   3. ``\\widehat{\\Delta}^{(d)}=\\sum_{k=1}^{m!}\\left(\\hat{p}_k^{(d)}-1 / m!\\right)^2``
#   4. ``\\hat{\\beta}^{(d)}=\\hat{p}_6^{(d)}-\\hat{p}_1^{(d)}``
#   5. ``\\hat{\\tau}^{(d)}=\\hat{p}_6^{(d)}+\\hat{p}_1^{(d)}-\\frac{1}{3}``
#   6. ``\\hat{\\delta}^{(d)}=\\hat{p}_4^{(d)}+\\hat{p}_5^{(d)}-\\hat{p}_3^{(d)}-\\hat{p}_2^{(d)}``
#
#   The patterns are categorized as follows:
#
#   ``
#   \\qquad p_1 = (3,2,1);  \\quad p_2=(3,1,2);  \\quad p_3 = (2,3,1);
#   ``
#
#   ``
#   \\qquad p_4 = (1,3,2);  \\quad p_5 = (2,1,3);  \\quad p_ 6 = (1,2,3)
#   ``
# - `jmin::Int` Minimum number of decimals for final control limit to optimize.
# - `jmax::Int` Maximum number of decimals for final control limit to optimize.
# - `verbose::Bool=false` Print intermediate results?
# - `d::Union{Int,Vector{Int}}=1`: Delay value or vector.
# - `ced::Bool=false`: Use conditional expected delay? Default is false.
# - `ad::Int=100`: Number of iterations for ced.
# """
# function cl_op(
#   op_dgp, lam, L0, cl_init, reps=10_000;
#   chart_choice, jmin=4, jmax=6, verbose=false, d=1, m=3, ced=false, ad=100
# )
#
#   L1 = zeros(2)
#
#   for j in jmin:jmax
#     for dh in 1:80
#
#       if (chart_choice isa Shannon || chart_choice isa ShannonExtropy)
#         cl_init = cl_init - (-1)^j * dh / 10^j
#       else
#         cl_init = cl_init + (-1)^j * dh / 10^j
#       end
#       L1 = arl_op_ic(
#         op_dgp, lam, cl_init, reps; chart_choice=chart_choice, d=d, m=m, ced=ced, ad=ad
#       )
#
#       if verbose
#         println("cl = ", cl_init, "\t", "ARL = ", L1[1])
#       end
#       if (j % 2 == 1 && L1[1] < L0) || (j % 2 == 0 && L1[1] > L0)
#         break
#       end
#     end
#     cl_init = cl_init
#   end
#
#   if L1[1] < L0
#     cl_init = cl_init + 1 / 10^jmax
#   end
#   return cl_init
# end
