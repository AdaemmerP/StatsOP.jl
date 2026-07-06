
"""
    arl_sop_bp_bootstrap(p_array, lam, cl, w, reps; chart_choice=TauTilde(),
      rl_max=typemax(Int))

Compute the average run length (ARL) of the EWMA chart based on the Box-Pierce type
statistic for spatial ordinal patterns (SOPs) by bootstrapping from pre-computed SOP
type frequencies. The computation is multithreaded.

- `p_array::Array{Float64, 3}`: 3-dimensional array with the relative SOP type
  frequencies for each delay combination. The first dimension is the picture, the second
  the pattern group (s₁, s₂, s₃), and the third the d₁-d₂ combination. This array is
  used for resampling; it can be computed with [`compute_p_array_bp`](@ref).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `w::Int`: window size of the BP statistic.
- `reps::Int`: number of replications.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_sop_bp_bootstrap(
    p_array::Array{T,3}, lam, cl, w, reps; chart_choice=TauTilde(), rl_max::Int=typemax(Int)
) where {T<:Real}

    # Number of chunks for load balancing
    n_chunks = Threads.nthreads() * 4

    # Make chunks for separate tasks (based on number of threads)
    chunks = Iterators.partition(1:reps, div(reps, n_chunks))

    par_results = map(chunks) do i
        Threads.@spawn rl_sop_bp_bootstrap(p_array, lam, cl, i, chart_choice, rl_max)
    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec::Vector{Int} = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_bp_bootstrap(p_array, lam, cl, reps_range, chart_choice,
      rl_max=typemax(Int))

Compute run lengths of the EWMA chart based on the Box-Pierce type statistic for spatial
ordinal patterns (SOPs) by bootstrapping, for a chunk of replications. This is the
single-threaded worker used by [`arl_sop_bp_bootstrap`](@ref).

- `p_array::Array{Float64,3}`: 3-dimensional array with the relative SOP type
  frequencies for each delay combination (see [`arl_sop_bp_bootstrap`](@ref)).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `reps_range::UnitRange{Int}`: range of replication indices to process.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_sop_bp_bootstrap(
    p_array::Array{T,3}, lam, cl, reps_range::UnitRange, chart_choice, rl_max::Int=typemax(Int)
) where {T<:Real}

    # Pre-allocate
    if chart_choice isa Union{TauHat, KappaHat, TauTilde, KappaTilde}
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

            # Break while loop when rl exceeds rl_max
            if rl > rl_max
                break
            end
        end

        rls[r] = rl

    end
    return rls
end
