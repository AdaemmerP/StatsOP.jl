
"""
    arl_sop_bp_ic(spatial_dgp, lam, cl, w, reps=1_000; chart_choice=TauTilde(),
      refinement=false, rl_max=typemax(Int))

Compute the in-control average run length (ARL) of the EWMA chart based on the
Box-Pierce type statistic for spatial ordinal patterns (SOPs) via simulation. The
computation is multithreaded.

- `spatial_dgp::ICSTS`: in-control spatial DGP.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `w::Int`: An integer value for the window size for the BP-statistic.
- `reps::Int`: An integer value for the number of repetitions. The default value is 1,000.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `refinement`: `false` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_sop_bp_ic(
  spatial_dgp::ICSTS, lam, cl, w::Int, reps=1_000; chart_choice=TauTilde(), refinement::Union{Bool,RefinedType}=false, rl_max::Int=typemax(Int)
)

  # Compute m and n  
  dist_error = spatial_dgp.dist

  # Compute lookup array to finde SOPs
  lookup_array_sop = compute_lookup_array_sop()

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_sop_bp_ic(
      spatial_dgp, lam, cl, w, lookup_array_sop, i, dist_error, chart_choice, refinement, rl_max
    )
  end
  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_bp_ic(spatial_dgp, lam, cl, w, lookup_array_sop, reps_range, dist_error,
      chart_choice, refinement, rl_max=typemax(Int))

Compute in-control run lengths of the EWMA chart based on the Box-Pierce type statistic
for spatial ordinal patterns (SOPs), for a chunk of replications. This is the
single-threaded worker used by [`arl_sop_bp_ic`](@ref).

- `spatial_dgp::ICSTS`: in-control spatial DGP.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `w::Int`: An integer value for the window size for the BP-statistic.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops,
which will be computed using `lookup_array_sop = StatsOP.compute_lookup_array_sop()`.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions.
- `dist_error::Distribution`: A distribution for the error term. Here you can use any
univariate distribution from the `Distributions.jl` package.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `refinement`: `false` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_sop_bp_ic(
  spatial_dgp::ICSTS, lam, cl, w::Int, lookup_array_sop, reps_range::UnitRange,
  dist_error, chart_choice,
  refinement,
  rl_max::Int=typemax(Int)
)

  # Pre-allocate
  n_size = _n_sop_types(refinement)
  p_hat = zeros(n_size)
  p_ewma = zeros(n_size)

  sop = zeros(4)
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)
  rls = zeros(Int, length(reps_range))

  # Extract matrix sizes and pre-allocate data matrix
  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols
  data = zeros(M, N)

  # Compute all possible combinations of d1 and d2
  d1_d2_combinations = Iterators.product(1:w, 1:w)
  # Pre-allocate array for p_ewma for all d1-d2 combinations
  p_ewma_all = zeros(n_size, 1, length(d1_d2_combinations))

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement=refinement)

  for r in axes(reps_range, 1)

    fill!(p_ewma_all, 1 / 3)
    bp_stat = 0.0
    rl = 0

    while bp_stat < cl
      rl += 1

      # Fill data 
      rand!(dist_error, data)

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

        m = spatial_dgp.M_rows - d1
        n = spatial_dgp.N_cols - d2

        # Compute sum of frequencies for each pattern group
        sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

        # Fill 'p_hat' with sop-frequencies and compute relative frequencies
        fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)

        # Apply EWMA to p-vectors
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
      # -------------------------------------------------#

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end

    rls[r] = rl
  end
  return rls
end
