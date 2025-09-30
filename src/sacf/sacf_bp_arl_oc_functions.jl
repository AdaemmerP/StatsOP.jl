""" 

    arl_sacf_bp_oc(spatial_dgp::SpatialDGP, lam, cl, w::Int, reps=10_000)

Compute the out-of-control average run length (ARL) using the spatial autocorrelation 
function (SACF) for the BP-statistic. The function returns the ARL for a given control limit `cl` and a given number of repetitions `reps`. The input arguments are:

- `spatial_dgp::SpatialDGP`: The spatial data generating process (DGP) to use for 
the SACF function. This can be one of the following: `SAR1`, `SAR11`, `SAR22`, 
  `SINAR11`, `SQMA11`, `SQINMA11`, or `BSQMA11`.    
- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) 
control chart.
- `cl`: The control limit for the EWMA control chart.
- `d1_vec::Vector{Int}`: The first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The second (column) delays for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf_bp_oc(spatial_dgp::SpatialDGP, lam, cl, w::Int, reps = 10_000)

    # Extract distribution          
    dist_error = spatial_dgp.dist
    dist_ao = spatial_dgp.dist_ao

    # Check whether to use threading or multi processing
    if nprocs() == 1 # Threading

        # Make chunks for separate tasks (based on number of threads)        
        chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads()))

        # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
        par_results = map(chunks) do i
            Threads.@spawn rl_sacf_bp(spatial_dgp, lam, cl, w, i, dist_error, dist_ao)
        end

    elseif nprocs() > 1 # Multi Processing

        # Make chunks for separate tasks (based on number of workers)    
        chunks = Iterators.partition(1:reps, div(reps, nworkers()))

        par_results = pmap(chunks) do i
            rl_sacf_bp(spatial_dgp, lam, cl, w, i, dist_error, dist_ao)
        end

    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end


# ------------------------------------------------------------------------------#
# -------------------           Run length method for SAR1           -----------#
# ------------------------------------------------------------------------------#
function rl_sacf_bp(
    spatial_dgp::SAR1,
    lam,
    cl,
    w::Int,
    p_reps::UnitRange,
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # Extract matrix sizes and pre-allocate
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # Compute all relevant h1-h2 combinations
    set_1 = Iterators.product(1:w, 0:w)
    set_2 = Iterators.product((-w):0, 1:w)
    h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))
    rho_hat_all = zeros(length(h1_h2_combinations))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    # vec_ar: vector for SAR(1) model
    # vec_ar2: vector for in-place multiplication for SAR(1) model
    mat = build_sar1_matrix(spatial_dgp) # will be done only once
    mat_ao = zeros((M + 2 * spatial_dgp.margin), (N + 2 * spatial_dgp.margin))
    vec_ar = zeros((M + 2 * spatial_dgp.margin) * (N + 2 * spatial_dgp.margin))
    vec_ar2 = similar(vec_ar)
    mat2 = similar(mat_ao)

    # Loop over repetitions
    for r in axes(p_reps, 1)

        fill!(rho_hat_all, 0.0)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl # BP-statistic can only be positive

            rl += 1

            # Fill matrix with dgp 
            data .= fill_mat_dgp_sop!(
                spatial_dgp,
                dist_error,
                dist_ao,
                mat,
                mat_ao,
                vec_ar,
                vec_ar2,
                mat2,
            )

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute BP-statistic using all d1-d2 combinations
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (h1, h2)) in enumerate(h1_h2_combinations)

                # compute ρ(d1,d2)-EWMA
                rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, h1, h2)
                bp_stat += 2 * rho_hat_all[i]^2

            end

        end

        rls[r] = rl

    end
    return rls
end

# ------------------------------------------------------------------------------#
# -----------   Run length method for SAR11, SAR22 and SINAR11        ----------#
# ------------------------------------------------------------------------------#
function rl_sacf_bp(
    spatial_dgp::Union{SAR11,SINAR11,SAR22},
    lam,
    cl,
    w::Int,
    p_reps::UnitRange,
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # Extract matrix sizes and pre-allocate
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # Compute all relevant h1-h2 combinations
    set_1 = Iterators.product(1:w, 0:w)
    set_2 = Iterators.product((-w):0, 1:w)
    h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))
    rho_hat_all = zeros(length(h1_h2_combinations))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
    init_mat!(spatial_dgp, dist_error, mat)

    # Loop over repetitions
    for r in axes(p_reps, 1)

        fill!(rho_hat_all, 0.0)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl # BP-statistic can only be positive

            rl += 1

            # Fill matrix with dgp 
            data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute BP-statistic using all d1-d2 combinations
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (h1, h2)) in enumerate(h1_h2_combinations)

                # compute ρ(d1,d2)-EWMA
                rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, h1, h2)
                bp_stat += 2 * rho_hat_all[i]^2

            end

            # Re-initialize matrix 
            fill!(mat, 0.0)
            if typeof(spatial_dgp) ∈ (SAR11, SINAR11, SAR22)
                init_mat!(spatial_dgp, dist_error, mat)
            end

        end

        rls[r] = rl
    end
    return rls
end

# ------------------------------------------------------------------------------#
# -----------   Run length method for SQMA11 and SQINMA11              ---------#
# ------------------------------------------------------------------------------#
function rl_sacf_bp(
    spatial_dgp::Union{SQMA11,SQINMA11},
    lam,
    cl,
    w::Int,
    p_reps::UnitRange,
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # Extract matrix sizes and pre-allocate
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # Compute all relevant h1-h2 combinations
    set_1 = Iterators.product(1:w, 0:w)
    set_2 = Iterators.product((-w):0, 1:w)
    h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))
    rho_hat_all = zeros(length(h1_h2_combinations))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + 1, N + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Loop over repetitions
    for r in axes(p_reps, 1)

        fill!(rho_hat_all, 0.0)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl # BP-statistic can only be positive

            rl += 1

            # Fill matrix with dgp 
            data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute BP-statistic using all d1-d2 combinations
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (h1, h2)) in enumerate(h1_h2_combinations)

                # compute ρ(d1,d2)-EWMA
                rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, h1, h2)
                bp_stat += 2 * rho_hat_all[i]^2

            end

            # Re-set matrix 
            fill!(mat, 0.0)

        end

        rls[r] = rl
    end
    return rls
end

# ------------------------------------------------------------------------------#
# -----------        Run length method for SQMA22                     ----------#
# ------------------------------------------------------------------------------#
function rl_sacf_bp(
    spatial_dgp::SQMA22,
    lam,
    cl,
    w::Int,
    p_reps::UnitRange,
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # Extract matrix sizes and pre-allocate
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # Compute all relevant h1-h2 combinations
    set_1 = Iterators.product(1:w, 0:w)
    set_2 = Iterators.product((-w):0, 1:w)
    h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))
    rho_hat_all = zeros(length(h1_h2_combinations))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + 2, N + 2)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Loop over repetitions
    for r in axes(p_reps, 1)

        fill!(rho_hat_all, 0.0)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl # BP-statistic can only be positive

            rl += 1

            # Fill matrix with dgp 
            data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute BP-statistic using all d1-d2 combinations
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (h1, h2)) in enumerate(h1_h2_combinations)

                # compute ρ(d1,d2)-EWMA
                rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, h1, h2)
                bp_stat += 2 * rho_hat_all[i]^2

            end

            # Re-set matrix 
            fill!(mat, 0.0)

        end

        rls[r] = rl
    end
    return rls
end

#------------------------------------------------------------------------------#
# -----------        Run length method for BSQMA11                   ----------#
#------------------------------------------------------------------------------#
function rl_sacf_bp(
    spatial_dgp::BSQMA11,
    lam,
    cl,
    w::Int,
    p_reps::UnitRange,
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # Extract matrix sizes and pre-allocate
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # Compute all relevant h1-h2 combinations
    set_1 = Iterators.product(1:w, 0:w)
    set_2 = Iterators.product((-w):0, 1:w)
    h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))
    rho_hat_all = zeros(length(h1_h2_combinations))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + 1, N + 1)
    mat_ma = zeros(M + 2, N + 2) # one extra row and column for "forward looking"
    mat_ao = similar(mat)

    # Loop over repetitions
    for r in axes(p_reps, 1)

        fill!(rho_hat_all, 0.0)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl # BP-statistic can only be positive

            rl += 1

            # Fill matrix with dgp 
            data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute BP-statistic using all d1-d2 combinations
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (h1, h2)) in enumerate(h1_h2_combinations)

                # compute ρ(d1,d2)-EWMA
                rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, h1, h2)
                bp_stat += 2 * rho_hat_all[i]^2

            end

            # Re-set matrix 
            fill!(mat, 0.0)

        end

        rls[r] = rl
    end
    return rls
end
