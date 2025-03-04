
"""
   arl_sacf_ic(lam, cl, spatial_dgp::ICSTS, d1::Int, d2::Int, reps=10_000)

Compute the in-control average run length (ARL), using the spatial autocorrelation 
function (SACF) for a delay (d1, d2) combination. The function returns the ARL 
for a given control limit `cl` and a given number of repetitions `reps`. 
    
The input arguments are:

- `spatial_dgp`: The in-control spatial data generating process (DGP) to use for the SACF function.
- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf_ic(spatial_dgp::ICSTS, lam, cl, d1::Int, d2::Int, reps=10_000)

    # Extract        
    dist_error = spatial_dgp.dist

    # Check whether to use threading or multi processing
    if nprocs() == 1 # Threading

        # Make chunks for separate tasks (based on number of threads)        
        chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

        # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
        par_results = map(chunks) do i
            Threads.@spawn rl_sacf_ic(spatial_dgp, lam, cl, d1, d2, i, dist_error)
        end

    elseif nprocs() > 1 # Multi Processing
        
        # Make chunks for separate tasks (based on number of workers)
        chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

        par_results = pmap(chunks) do i
            rl_sacf_ic(spatial_dgp, lam, cl, d1, d2, i, dist_error)
        end

    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sacf_ic(
    spatial_dgp::ICSP, lam, cl, d1::Int, d2::Int, p_reps::UnitRange, dist_error::UnivariateDistribution
)

Compute the in-control run length using the spatial autocorrelation function (SACF) for a delay (d1, d2) combination. The function returns the run length for a given control limit `cl`.

The input arguments are:
  
- `spatial_dgp::ICSP`: The in-control spatial data generating process (DGP) to use for the SACF function.
- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `p_reps::UnitRange`: The number of repetitions to compute the run length.
- `dist_error::UnivariateDistribution`: The distribution to use for the error term.
"""
function rl_sacf_ic(
    spatial_dgp::ICSTS, lam, cl, d1::Int, d2::Int, p_reps::UnitRange, dist_error::UnivariateDistribution
)

    # Extract matrix size and pre-allocate matrices
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    for r in axes(p_reps, 1)

        rho_hat = 0.0
        rl = 0

        while abs(rho_hat) < cl
            rl += 1

            # fill matrix with iid values
            rand!(dist_error, data)
            X_centered .= data .- mean(data)

            # compute ρ(d1,d2)-EWMA
            rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

        end

        rls[r] = rl

    end

    return rls

end
