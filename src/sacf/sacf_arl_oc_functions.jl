
"""
    arl_sacf_oc(sp_dgp::SpatialDGP, lam, cl, d1::Int, d2::Int, reps=10_000)

Compute the in-control average run length (ARL) using the spatial autocorrelation 
function (SACF) for a delay (d1, d2) combination and an out-of-control process. 
  
The input arguments are: 

- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `sp_dgp`: The spatial data generating process (DGP) to use for the SACF function. 
This can be one of the following: `SAR1`, `SAR11`, `SAR22`, `SINAR11`, `SQMA11`, 
`SQINMA11`, or `BSQMA11`.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf_oc(sp_dgp::SpatialDGP, lam, cl, d1::Int, d2::Int, reps = 10_000)

    # extract m and n from spatial_dgp
    dist_error = sp_dgp.dist
    dist_ao = sp_dgp.dist_ao

    # Check whether to use threading or multi processing
    if nprocs() == 1 # Threading

        # Make chunks for separate tasks (based on number of threads)        
        chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

        # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
        par_results = map(chunks) do i
            Threads.@spawn rl_sacf_oc(sp_dgp, lam, cl, d1, d2, i, dist_error, dist_ao)
        end

    elseif nprocs() > 1 # Multi Processing

        # Make chunks for separate tasks (based on number of workers)
        chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

        par_results = pmap(chunks) do i
            rl_sacf_oc(sp_dgp, lam, cl, d1, d2, i, dist_error, dist_ao)
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
function rl_sacf_oc(
    spatial_dgp::SAR1,
    lam,
    cl,
    d1::Int,
    d2::Int,
    p_reps::UnitRange{Int},
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # pre-allocate  
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

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


    for r in axes(p_reps, 1)

        rho_hat = 0.0
        rl = 0

        while abs(rho_hat) < cl
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

            # Compute ρ(d1,d2)-EWMA
            rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

        end

        rls[r] = rl
    end
    return rls
end

# ------------------------------------------------------------------------------#
# -----------   Run length method for SAR11, SAR22 and SINAR11        ----------#
# ------------------------------------------------------------------------------#
function rl_sacf_oc(
    spatial_dgp::Union{SAR11,SINAR11,SAR22},
    lam,
    cl,
    d1::Int,
    d2::Int,
    p_reps::UnitRange{Int},
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # pre-allocate  
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
    init_mat!(spatial_dgp, dist_error, mat)

    for r in axes(p_reps, 1)

        rho_hat = 0.0
        rl = 0

        while abs(rho_hat) < cl
            rl += 1

            # Fill matrix with dgp
            data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute ρ(d1,d2)-EWMA
            rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

            # Re-set matrix 
            fill!(mat, 0.0)
            init_mat!(spatial_dgp, dist_error, mat)

        end

        rls[r] = rl
    end
    return rls
end


# ------------------------------------------------------------------------------#
# -----------   Run length method for SQMA11, SQINMA11                ----------#
# ------------------------------------------------------------------------------#
function rl_sacf_oc(
    spatial_dgp::Union{SQMA11,SQINMA11},
    lam,
    cl,
    d1::Int,
    d2::Int,
    p_reps::UnitRange{Int},
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # pre-allocate  
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + 1, N + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    for r in axes(p_reps, 1)

        rho_hat = 0.0
        rl = 0

        while abs(rho_hat) < cl
            rl += 1

            # Fill matrix with dgp
            if spatial_dgp isa SAR1
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
            else
                data .=
                    fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)
            end

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute ρ(d1,d2)-EWMA
            rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

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
function rl_sacf_oc(
    spatial_dgp::SQMA22,
    lam,
    cl,
    d1::Int,
    d2::Int,
    p_reps::UnitRange{Int},
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # pre-allocate  
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + 2, N + 2)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    for r in axes(p_reps, 1)

        rho_hat = 0.0
        rl = 0

        while abs(rho_hat) < cl
            rl += 1

            # Fill matrix with dgp
            data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute ρ(d1,d2)-EWMA
            rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

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
function rl_sacf_oc(
    spatial_dgp::BSQMA11,
    lam,
    cl,
    d1::Int,
    d2::Int,
    p_reps::UnitRange{Int},
    dist_error::UnivariateDistribution,
    dist_ao::Union{UnivariateDistribution,Nothing},
)

    # pre-allocate  
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    mat = zeros(M + 1, N + 1)
    mat_ma = zeros(M + 2, N + 2) # one extra row and column for "forward looking"
    mat_ao = similar(mat)

    for r in axes(p_reps, 1)

        rho_hat = 0.0
        rl = 0

        while abs(rho_hat) < cl
            rl += 1

            # Fill matrix with dgp
            data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # Compute ρ(d1,d2)-EWMA
            rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

            # Re-set matrix 
            fill!(mat, 0.0)

        end

        rls[r] = rl
    end
    return rls
end
