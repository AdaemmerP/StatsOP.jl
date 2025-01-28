"""
    arl_sacf_bp_ic(lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000)

Compute the in-control average run length (ARL) using the spatial autocorrelation function (SACF) for the BP-statistic. The input arguments are:  

- `lam`: The smoothing parameter for the exponentially weighted moving average 
(EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `sp_dgp`: The in-control spatial data generating process (DGP) to use for the 
SACF function.
- `d1_vec::Vector{Int}`: The first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The second (column) delays for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf_bp_ic(sp_dgp::ICSP, lam, cl, w::Int, reps=10_000)

    # Extract distribution        
    dist_error = sp_dgp.dist

    # Check whether to use threading or multi processing
    if nprocs() == 1 # Threading

        # Make chunks for separate tasks (based on number of threads)        
        chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads()))

        # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
        par_results = map(chunks) do i
            Threads.@spawn rl_sacf_bp_ic(sp_dgp, lam, cl, w, i, dist_error)
        end

    elseif nprocs() > 1 # Multi Processing

        # Make chunks for separate tasks (based on number of workers)    
        chunks = Iterators.partition(1:reps, div(reps, nworkers()))

        par_results = pmap(chunks) do i
            rl_sacf_bp_ic(lam, cl, sp_dgp, d1_vec, d2_vec, i, dist_error)
        end

    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end

"""
    rl_sacf_bp_ic(
  lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, p_reps::UnitRange, dist_error::UnivariateDistribution
)

Compute the out-of-control run length using the spatial autocorrelation function 
(SACF) for the BP-statistic. The function returns the run length for a given control 
  limit `cl` and a given number of repetitions `reps`. The input arguments are:

 - `sp_dgp::ICSP`: The in-control spatial data generating process (DGP) to use for 
the SACF function. 
- `lam`: The smoothing parameter for the exponentially weighted moving average 
(EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `d1_vec::Vector{Int}`: The first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The second (column) delays for the spatial process.
- `p_reps::UnitRange`: The number of repetitions to compute the run length. This 
has to be a unit range of integers to allow for parallel processing, since the 
  function is called by `arl_sacf()`.
- `dist_error::UnivariateDistribution`: The distribution to use for the error term 
in the spatial process. This can be any univariate distribution from the `Distributions.jl` package.
"""
function rl_sacf_bp_ic(
    sp_dgp::ICSP, lam, cl, w::Int, p_reps::UnitRange, dist_error::UnivariateDistribution
)

    # Extract matrix size and pre-allocate data matrices
    M = sp_dgp.M_rows
    N = sp_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # Compute all relevant h1-h2 combinations
    set_1 = Iterators.product(1:w, 0:w)
    set_2 = Iterators.product(-w:0, 1:w)
    h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))
    rho_hat_all = zeros(length(h1_h2_combinations))

    for r in axes(p_reps, 1)

        fill!(rho_hat_all, 0.0)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl
            rl += 1

            # fill data matrix 
            rand!(dist_error, data)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # compute BP-statistic using all h1-h2 combinations
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (h1, h2)) in enumerate(h1_h2_combinations)

                rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, h1, h2)
                bp_stat += 2 * rho_hat_all[i]^2

            end

        end

        rls[r] = rl

    end

    return rls

end
