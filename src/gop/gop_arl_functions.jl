export arl_gop,
    rl_gop



# Function to compute average run length for ordinal patterns
function arl_gop(lam, cl, reps, gop_dgp, chart_choice; d=1)

    # Compute lookup array and number of ops
    lookup_array_gop = compute_lookup_array_gop()

    # No threading or multiprocessing
    if nprocs() == 1 && reps <= Threads.nthreads()
        results = rl_gop(
            lam, cl, lookup_array_gop, 1:reps, gop_dgp, gop_dgp.dist, chart_choice, d
        )

        return (mean(results), std(results) / sqrt(reps))

        # Threading
    elseif nprocs() == 1 && reps > Threads.nthreads()

        # Make chunks for separate tasks (based on number of threads)        
        chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

        # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
        par_results = map(chunks) do i

            Threads.@spawn rl_gop(lam, cl, lookup_array_gop, i, gop_dgp, gop_dgp.dist, chart_choice, d)

        end

        # Multiprocessing    
    elseif nprocs() > 1 && reps >= nworkers()

        # Make chunks for separate tasks (based on number of workers)
        chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

        par_results = pmap(chunks) do i
            rl_gop(lam, cl, lookup_array_gop, i, gop_dgp, gop_dgp.dist, chart_choice, d)
        end

    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps), median(rlvec))
end

#--- Run-length method for D-Chart
function rl_gop(
    lam, cl, lookup_array_gop, p_reps, gop_dgp, gop_dgp_dist, chart_choice::Union{D_Chart,Persistence}, d::Int
)

    # value of patterns (can become variable in future versions)
    m = 3

    # Pre-allocate variables
    rls = zeros(Int64, length(p_reps))
    bin = zeros(Int, 13)
    win = zeros(Int, m)
    ix = similar(win)
    p = zeros(13)
    p0 = zeros(13)
    p_p0 = zeros(13) # for "p - p0"
    fill_p0!(p0, gop_dgp_dist)

    # compute length of 'x_seq' vector based on d
    x_seq = zeros(Int, 1 + (m - 1) * d)

    for r in axes(p_reps, 1) # p_reps is a range

        # initialize run length at zero
        rl = 0
        # initialze EWMA statistic, Equation (17), in the paper
        p .= p0
        # Initialize observations
        seq = StatsOP.init_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)

        # initial statistic
        @. p_p0 = p - p0
        stat = chart_stat_gop(p_p0, chart_choice)

        while !abort_criterium_gop(stat, cl, chart_choice)
            # increase run length
            rl += 1
            bin .= 0

            # compute ordinal pattern based on permutations    
            competerank!(win, seq, ix)

            # Binarization of ordinal pattern
            j, k, l = win
            bin[lookup_array_gop[j, k, l]] = 1

            # Compute EWMA statistic, Equation (17), in the paper
            @. p = lam * bin + (1 - lam) * p
            # statistic based on smoothed p-estimate
            @. p_p0 = p - p0
            stat = chart_stat_gop(p_p0, chart_choice)

            # update sequence depending on DGP
            seq = StatsOP.update_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
            fill!(win, 0)
        end

        rls[r] = rl
    end
    return rls
end


#--- Run-length method for G-Chart
function rl_gop(
    lam, cl, lookup_array_gop, p_reps, gop_dgp, gop_dgp_dist, chart_choice::G_Chart, d::Int
)

    # value of patterns (can become variable in future versions)
    m = 3

    # Pre-allocate variables
    rls = zeros(Int64, length(p_reps))
    bin = zeros(Int, 13)
    win = zeros(Int, m)
    ix = similar(win)
    p = zeros(13)
    p0 = similar(p)
    p_p0 = similar(p) # for "p - p0"
    G = [
        1 0 0 0 0 0 0 1 0 1 0 0 0;
        0 0 0 0 0 1 0 0 0 0 1 0 1;
        0 1 1 1 1 0 1 0 1 0 0 1 0
    ]
    G1G = G' * G
    fill_p0!(p0, gop_dgp_dist)

    # compute length of 'x_seq' vector based on d
    x_seq = zeros(Int, 1 + (m - 1) * d)

    for r in axes(p_reps, 1) # p_reps is a range

        # initialize run length at zero
        rl = 0
        # initialze EWMA statistic, Equation (17), in the paper
        p .= p0
        # Initialize observations
        seq = StatsOP.init_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
        # initial statistic
        @. p_p0 = p - p0
        stat = chart_stat_gop(p_p0, G1G, chart_choice)
        @assert all(isfinite, p_p0)


        while !abort_criterium_gop(stat, cl, chart_choice)
            # increase run length
            rl += 1
            bin .= 0

            # compute ordinal pattern based on permutations    
            competerank!(win, seq, ix)

            @assert isfinite(stat)

            # Binarization of ordinal pattern
            j, k, l = win
            bin[lookup_array_gop[j, k, l]] = 1

            # Compute EWMA statistic, Equation (17), in the paper
            @. p = lam * bin + (1 - lam) * p
            # statistic based on smoothed p-estimate
            @. p_p0 = p - p0
            stat = chart_stat_gop(p_p0, G1G, chart_choice)

            # update sequence depending on DGP
            seq = StatsOP.update_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
            fill!(win, 0)

        end

        rls[r] = rl
    end
    return rls
end

