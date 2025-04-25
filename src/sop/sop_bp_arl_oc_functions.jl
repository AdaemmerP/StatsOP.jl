"""
  arl_sop_bp_oc(
  spatial_dgp::SpatialDGP, lam, cl, w::Int, reps=10_000; chart_choice=3
)

Compute the average run length (ARL) for a given out-of-control spatial DGP, using 
the EWMA-BP-SOP statistic.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `spatial_dgp::SpatialDGP`: A struct for the out-of-control spatial DGP.
- `w::Int`: An integer value for the window size for the BP-statistic.
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""

function arl_sop_bp_oc(
    spatial_dgp::SpatialDGP, lam, cl, w::Int, reps=1_000; chart_choice=3, refinement::Int=0
)

    # Check input parameters
    @assert 1 <= chart_choice <= 7 "chart_choice must be between 1 and 7"
    if chart_choice in 1:4
        @assert refinement == 0 "refinement must be 0 for chart_choice 1-4"
    elseif chart_choice in 5:7
        @assert 1 <= refinement <= 3 "refinement must be 1-3 for chart_choices 5-7"
    end

    # Compute m and n
    dist_error = spatial_dgp.dist
    dist_ao = spatial_dgp.dist_ao

    # Compute lookup array to finde SOPs
    lookup_array_sop = compute_lookup_array_sop()

    # Check whether to use threading or multi processing --> only one process threading, else distributed
    if nprocs() == 1

        # Make chunks for separate tasks        
        chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect
        # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
        par_results = map(chunks) do i

            Threads.@spawn rl_sop_bp_oc(
                spatial_dgp, lam, cl, w, lookup_array_sop, i, dist_error, dist_ao, chart_choice, refinement
            )

        end

    elseif nprocs() > 1 # Multi Processing

        chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
        par_results = pmap(chunks) do i

            rl_sop_bp_oc(
                spatial_dgp, lam, cl, w, lookup_array_sop, i, dist_error, dist_ao, chart_choice, refinement
            )

        end

    end
    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_bp_oc(
    spatial_dgp::SpatialDGP, lam, cl, w::Int, lookup_array_sop, p_reps::UnitRange,
    dist_error::UnivariateDistribution, dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice
)

Computes the run length for a given out-of-control DGP, using the EWMA-BP-SOP statistic.

The input parameters are:

- `spatial_dgp::SpatialDGP`: A struct for the out-of-control spatial DGP.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `w::Int`: An integer value for the window size for the BP-statistic.
- `lookup_array_sop::Array{Int, 2}`: A 2D array with the lookup table for the SOPs.
- `p_reps::UnitRange`: A range of integers for the number of repetitions.
- `dist_error::UnivariateDistribution`: A distribution for the error term.
- `dist_ao::Union{Nothing,UnivariateDistribution}`: A distribution for the additive outlier.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""
function rl_sop_bp_oc(
    spatial_dgp::SpatialDGP, lam, cl, w::Int, lookup_array_sop, p_reps::UnitRange,
    dist_error::UnivariateDistribution, dist_ao::Union{Nothing,UnivariateDistribution},
    chart_choice, refinement
)

    # find maximum values of d1 and d2 for construction of matrices
    M = spatial_dgp.M_rows
    N = spatial_dgp.N_cols

    # Compute all possible combinations of d1 and d2  
    d1_d2_combinations = Iterators.product(1:w, 1:w)

    # pre-allocate
    if refinement == 0
        # classical approach
        p_hat = zeros(3)
        p_ewma_all = zeros(3, 1, length(d1_d2_combinations))
    else
        # refined approach
        p_hat = zeros(6)
        p_ewma_all = zeros(6, 1, length(d1_d2_combinations))
    end
    sop_freq = zeros(Int, 24)
    win = zeros(Int, 4)
    data = zeros(M, N)
    rls = zeros(Int, length(p_reps))
    sop = zeros(4)

    # indices for sum of frequencies
    index_sop = create_index_sop(refinement=refinement)
    #s_1 = index_sop[1]
    #s_2 = index_sop[2]
    #s_3 = index_sop[3]

    # pre-allocate
    # mat:    matrix for the final values of the spatial DGP
    # mat_ao: matrix for additive outlier 
    # mat_ma: matrix for moving averages
    # vec_ar: vector for SAR(1) model
    # vec_ar2: vector for in-place multiplication for SAR(1) model
    if spatial_dgp isa SAR1
        mat = build_sar1_matrix(spatial_dgp) # will be done only once
        mat_ao = zeros((M + 2 * spatial_dgp.margin), (N + 2 * spatial_dgp.margin))
        vec_ar = zeros((M + 2 * spatial_dgp.margin) * (N + 2 * spatial_dgp.margin))
        vec_ar2 = similar(vec_ar)
        mat2 = similar(mat_ao)
    elseif typeof(spatial_dgp) ∈ (SAR11, SINAR11, SAR22)
        mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
        mat_ma = similar(mat)
        mat_ao = similar(mat)
        # Function to initialize matrix only for SAR(1,1) and SINAR(1,1) and SAR(2,2) 
        init_mat!(spatial_dgp, dist_error, mat)
    elseif typeof(spatial_dgp) ∈ (SQMA11, SQINMA11)
        mat = zeros(M + 1, N + 1)
        mat_ma = similar(mat)
        mat_ao = similar(mat)
    elseif spatial_dgp isa SQMA22
        mat = zeros(M + 2, N + 2)
        mat_ma = similar(mat)
        mat_ao = similar(mat)
    elseif spatial_dgp isa BSQMA11
        mat = zeros(M + 1, N + 1)
        mat_ma = zeros(M + 2, N + 2) # one extra row and column for "forward looking"
        mat_ao = similar(mat)
    end

    for r in axes(p_reps, 1)

        fill!(p_ewma_all, 1 / 3)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl # BP-statistic can only be positive
            rl += 1

            # Fill matrix via dgp 
            if spatial_dgp isa SAR1
                data .= fill_mat_dgp_sop!(
                    spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2, mat2
                )
            else
                data .= fill_mat_dgp_sop!(
                    spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma
                )
            end

            # Add noise when using count data
            if dist_error isa DiscreteUnivariateDistribution
                for j in axes(data, 2)
                    for i in axes(data, 1)
                        data[i, j] = data[i, j] + rand()
                    end
                end
            end

            # -----------------------------------------------------------------------#
            # ----------------     Loop for BP-Statistik                     --------#
            # -----------------------------------------------------------------------#
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (d1, d2)) in enumerate(d1_d2_combinations)

                # Determine row and column of SOP matrix
                m = spatial_dgp.M_rows - d1
                n = spatial_dgp.N_cols - d2

                # Compute sum of frequencies for each pattern group
                sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

                # Fill 'p_hat' with sop-frequencies and compute relative frequencies
                fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, index_sop)# s_1, s_2, s_3)

                # Apply EWMA
                @views @. p_ewma_all[:, :, i] = (1 - lam) * p_ewma_all[:, :, i] + lam * p_hat

                # Compute test statistic for one d1-d2 combination
                @views stat = chart_stat_sop(p_ewma_all[:, :, i], chart_choice)

                # Compute BP-statistic
                bp_stat += stat^2

                # Reset win, sop_freq and p_hat
                fill!(win, 0)
                fill!(sop_freq, 0)
                fill!(p_hat, 0)
            end
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
        end

        rls[r] = rl
    end
    return rls
end

