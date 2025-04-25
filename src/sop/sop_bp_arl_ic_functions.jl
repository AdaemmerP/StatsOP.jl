
"""
    arl_sop_bp_ic(
  spatial_dgp::ICSP, lam, cl, w::Int, reps=10_000; chart_choice=3
)

Computes the average run length (ARL) for a given in-control spatial DGP, using 
EWMA-BP-SOP statistics.

The input parameters are:

- `spatial_dgp::ICSP`: A struct for the in-control spatial DGP.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `w::Int`: An integer value for the window size for the BP-statistic.
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""
function arl_sop_bp_ic(
  spatial_dgp::ICSTS, lam, cl, w::Int, reps=1_000; chart_choice=3, refinement::Int=0
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

  # Compute lookup array to finde SOPs
  lookup_array_sop = compute_lookup_array_sop()

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect
    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_sop_bp_ic(
        spatial_dgp, lam, cl, w, lookup_array_sop, i, dist_error, chart_choice, refinement
      )

    end

  elseif nprocs() > 1 # Multi Processing

    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
    par_results = pmap(chunks) do i

      rl_sop_bp_ic(
        spatial_dgp, lam, cl, w, lookup_array_sop, i, dist_error, chart_choice, refinement
      )

    end

  end
  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_bp_ic(
  spatial_dgp::ICSP, lam, cl, lookup_array_sop, reps_range, dist, chart_choice, 
  d1_vec::Vector{Int}, d2_vec::Vector{Int}
)

Computes the run length for a given in-control spatial DGP, using the EWMA-BP-SOP statistic.

The input parameters are:

- `spatial_dgp::ICSP`: A struct for the in-control spatial DGP.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops,
which will be computed computed using `lookup_array_sop = compute_lookup_array_sop()`.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions.
- `dist::Distribution`: A distribution for the error term. Here you can use any
univariate distribution from the `Distributions.jl` package.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `d1_vec::Vector{Int}`: A vector with integer values for the first delay (d₁).
- `d2_vec::Vector{Int}`: A vector with integer values for the second delay (d₂).
"""
function rl_sop_bp_ic(
  spatial_dgp::ICSTS, lam, cl, w::Int, lookup_array_sop, reps_range::UnitRange,
  dist_error, chart_choice, refinement
)

  # Pre-allocate    
  if refinement == 0
    # classical approach
    p_hat = zeros(3)
  else
    # refined approach
    p_hat = zeros(6)
  end
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
  p_ewma_all = zeros(3, 1, length(d1_d2_combinations))

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement=refinement)
  #s_1 = index_sop[1]
  #s_2 = index_sop[2]
  #s_3 = index_sop[3]

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
        fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, index_sop) # s_1, s_2, s_3)

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
    end

    rls[r] = rl
  end
  return rls
end
