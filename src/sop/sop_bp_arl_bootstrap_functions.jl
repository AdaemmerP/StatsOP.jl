
"""
    arl_sop_bp(
        p_array::Array{T,3}, lam, cl, w, reps; chart_choice=3
    ) where {T<:Real}

Compute the average run length for the BP-EWMA-SOP for a given control limit 
  using bootstraping

The input parameters are:

- `p_array::Array{Float64, 3}`: A 3D array with the with the relative frequencies 
of each d1-d2 (delay) combination. The first dimension (rows) is the picture, the 
second dimension refers to the patterns group (s₁, s₂, or s₃) and the third dimension 
denotes each d₁-d₂ combination. This matrix will be used for re-sampling.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `w::Int`: An integer value for the number of workers.
- `reps::Int`: An integer value for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""
function arl_sop_bp_bootstrap(
    p_array::Array{T,3}, lam, cl, w, reps; chart_choice=3
) where {T<:Real}

    # Check whether to use threading or multi processing --> only one process threading, else distributed
    if nprocs() == 1
        # Make chunks for separate tasks (based on number of threads)        
        chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

        # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
        par_results = map(chunks) do i
            Threads.@spawn rl_sop_bp_bootstrap(p_array, lam, cl, i, chart_choice)
        end

    elseif nprocs() > 1

        # Make chunks for separate tasks (based on number of workers)
        chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

        par_results = pmap(chunks) do i
            rl_sop_bp_bootstrap(p_array, lam, cl, i, chart_choice)
        end

    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec::Vector{Int} = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_bp_bootstrap(p_array::Array{T,3}, lam, cl, reps_range, chart_choice, ) where {T<:Real}

Compute the EWMA-BP-SOP run length for a given control limit using bootstraping.

The input parameters are:

- `p_array::Array{Float64,3}`: A 3D array with the with the relative frequencies for 
each d1-d2 (delay) combination. The first dimension (rows) is the picture, the second
dimension refers to the patterns group (s₁, s₂, or s₃) and the third dimension denotes
each d₁-d₂ combination. This array will be used for re-sampling.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""
function rl_sop_bp_bootstrap(
    p_array::Array{T,3}, lam, cl, reps_range::UnitRange, chart_choice
) where {T<:Real}

    # Pre-allocate
    if chart_choice in 1:4
        # classical approach
        p_hat = zeros(3)
    else
        # refined approach
        p_hat = zeros(6)
    end
    rls = zeros(Int, length(reps_range))
    p_array_mean = mean(p_array, dims=1)
    range_index = axes(p_array, 1) # Range for number of images
    p_ewma = similar(p_array_mean) # will be dimension 1 x 3 x 'size(p_array, 3)'
    stat_ic = zeros(size(p_array, 3))

    # Compute in-control values
    for i in axes(p_array, 3)
        @views stat_ic[i] = chart_stat_sop(p_array_mean[:, :, i], chart_choice)
    end

    # Loop over repetitions
    for r in axes(reps_range, 1)
        p_ewma .= p_array_mean
        bp_stat = 0.0 # in-control value
        rl = 0

        while bp_stat < cl
            rl += 1

            # sample from p_vec
            index = rand(range_index)

            # initialize sum
            bp_stat = 0.0
            for i in axes(p_array, 3)

                @views p_hat .= p_array[index, :, i]

                # Apply EWMA
                @views @. p_ewma[:, :, i] = (1 - lam) * p_ewma[:, :, i] + lam * p_hat'

                # Compute test statistic
                @views stat = chart_stat_sop(p_ewma[:, :, i], chart_choice)
                bp_stat += (stat - stat_ic[i])^2

            end

        end

        rls[r] = rl

    end
    return rls
end
