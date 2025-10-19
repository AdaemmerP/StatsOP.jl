
"""
    arl_sop_ic(sop_dgp::ICSP, lam, cl, d1::Int, d2::Int, reps=10_000; chart_choice=3)

Compute the average run length (ARL) for a given in-control spatial DGP. 
  
The input parameters are:

- `sop_dgp::ICSP`: A struct for the in-control spatial DGP.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `d1::Int`: An integer value for the first delay (d₁).
- `d2::Int`: An integer value for the second delay (d₂).
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. 
The default value is 3.
"""
function arl_sop_ic(
  sop_dgp::ICSTS, lam, cl, d1::Int, d2::Int, reps=10_000; chart_choice=TauTilde(), refinement::Union{Nothing,RefinedType}=nothing
)

  # Check input parameters
  @assert chart_choice in (Shannon, ShannonExtropy, DistanceToWhiteNoise, TauHat, KappaHat, TauTilde, KappaTilde) "chart_choice must be one of the defined chart types from type InformationMeasure"
  # Extract values    
  m = sop_dgp.M_rows - d1
  n = sop_dgp.N_cols - d2
  dist = sop_dgp.dist

  # Compute lookup array and number of sops
  lookup_array_sop = compute_lookup_array_sop()

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sop_ic(lam, cl, lookup_array_sop, i, dist, chart_choice, refinement, m, n, d1, d2)
    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sop_ic(lam, cl, lookup_array_sop, i, dist, chart_choice, refinement, m, n, d1, d2)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_ic(lam, cl, lookup_array_sop, reps_range, dist, chart_choice, m, n, d1::Int, d2::Int)

Compute the run length for a given in-control spatial DGP. 
  
The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops, 
which will be computed computed using `lookup_array_sop = compute_lookup_array_sop()`. 
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions.
- `dist::Distribution`: A distribution for the error term. Here you can use any 
univariate distribution from the `Distributions.jl` package.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `m::Int`: An integer value for the number of rows for the final "SOP" matrix.
- `n::Int`: An integer value for the number of columns for the final "SOP" matrix.
- `d1::Int`: An integer value for the first delay (d₁).
- `d2::Int`: An integer value for the second delay (d₂).
"""
function rl_sop_ic(
  lam, cl, lookup_array_sop, reps_range::UnitRange{Int}, dist, chart_choice, refinement, m, n, d1::Int, d2::Int
)

  # Pre-allocate
  if isnothing(refinement)    # classical approach
    p_hat = zeros(3)
    p_ewma = zeros(3)
  elseif refinement isa RefinedType
    # refined approach
    p_hat = zeros(6)
    p_ewma = zeros(6)
  end
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
    end

    rls[r] = rl
  end
  return rls
end

