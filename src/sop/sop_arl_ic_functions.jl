
"""
    arl_sop_ic(sop_dgp, lam, cl, d1, d2, reps=10_000; chart_choice=TauTilde(),
      refinement=false, rl_max=typemax(Int))

Compute the in-control average run length (ARL) of the EWMA chart based on spatial
ordinal patterns (SOPs) via simulation. The computation is multithreaded.

- `sop_dgp::ICSTS`: in-control spatial DGP.
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `d1::Int`, `d2::Int`: row and column delays.
- `reps::Int=10_000`: number of replications.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `refinement`: `false` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_sop_ic(
  sop_dgp::ICSTS, lam, cl, d1::Int, d2::Int, reps=10_000;
  chart_choice=TauTilde(),
  refinement::Union{Bool,RefinedType}=false,
  rl_max::Int=typemax(Int)
)

  # Extract values    
  m = sop_dgp.M_rows - d1
  n = sop_dgp.N_cols - d2
  dist = sop_dgp.dist

  # Compute lookup array and number of sops
  lookup_array_sop = compute_lookup_array_sop()

  # Number of chunks
  n_chunks = Threads.nthreads() * 4

  # Assert that reps is bigger than
  @assert reps > n_chunks "Number of repetitions must be greater than number of chunks, which equal number of threads times 4. Current number of repetitions: $reps, number of chunks: $n_chunks."

  # Make chunks for separate tasks (based on number of threads)        
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
  par_results = map(chunks) do i
    Threads.@spawn rl_sop_ic(lam, cl, lookup_array_sop, i, dist, chart_choice, refinement, m, n, d1, d2, rl_max)
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_ic(lam, cl, lookup_array_sop, reps_range, dist, chart_choice, refinement,
      m, n, d1, d2, rl_max=typemax(Int))

Compute in-control run lengths of the EWMA chart based on spatial ordinal patterns
(SOPs) for a chunk of replications. This is the single-threaded worker used by
[`arl_sop_ic`](@ref).

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops,
which will be computed using `lookup_array_sop = compute_lookup_array_sop()`.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions.
- `dist::Distribution`: A distribution for the error term. Here you can use any
univariate distribution from the `Distributions.jl` package.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `refinement`: `false` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `m::Int`: An integer value for the number of rows for the final "SOP" matrix.
- `n::Int`: An integer value for the number of columns for the final "SOP" matrix.
- `d1::Int`: An integer value for the first delay (d₁).
- `d2::Int`: An integer value for the second delay (d₂).
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.
"""
function rl_sop_ic(
  lam, cl, lookup_array_sop, reps_range::UnitRange{Int}, dist, chart_choice, refinement, m, n, d1::Int, d2::Int, rl_max::Int=typemax(Int)
)


  # Pre-allocate
  n_size = refinement ? 6 : 3
  p_hat = zeros(n_size)
  p_ewma = zeros(n_size)

  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)
  data_tmp = zeros(m + d1, n + d2)
  rls = zeros(Int, length(reps_range))
  sop_vec = zeros(4)

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement=refinement)

  for r in 1:length(reps_range)
    fill!(p_ewma, 1.0 / 3.0)
    stat = chart_stat_sop(p_ewma, chart_choice)

    rl = 0

    while abs(stat) < cl
      rl += 1

      # Fill data 
      rand!(dist, data_tmp)

      # Add noise when using count data
      if dist isa DiscreteUnivariateDistribution
        for j in axes(data_tmp, 2)
          for i in axes(data_tmp, 1)
            data_tmp[i, j] = data_tmp[i, j] + rand()
          end
        end
      end

      # Compute frequencies of SOPs
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop_vec, win, sop_freq)

      # Fill 'p_hat' with sop-frequencies and compute relative frequencies
      fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)

      # Apply EWMA to p-vectors
      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      # Compute test statistic
      stat = chart_stat_sop(p_ewma, chart_choice)

      # Reset win and freq_sop
      fill!(win, 0)
      fill!(sop_freq, 0)
      fill!(p_hat, 0)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end

    rls[r] = rl
  end
  return rls
end

