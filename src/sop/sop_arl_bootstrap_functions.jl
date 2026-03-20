
"""
    arl_sop_bootstrap(p_mat::Array{Float64,2}, lam, cl, reps=10_000; chart_choice=chart_choice)

Compute the average run length (ARL) using a bootstrap approach  for a particular
delay (d₁-d₂) combination. 

The input parameters are:

- `p_mat::Array{Float64,2}`: A matrix with the values of the relative type frequencies.
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
The default value is 3.
"""
function arl_sop_bootstrap(
  p_mat::Array{Float64,2}, lam, cl, reps=10_000; chart_choice=TauTilde(), refinement=false, rl_max::Int=typemax(Int)
)

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_sop_bootstrap(p_mat, lam, cl, i, chart_choice, refinement, rl_max)
  end

  # Collect results from tasks   
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop_bootstrap(lam, cl, reps_range, chart_choice, p_mat::Array{Float64,2})

Compute the run length for a given control limit using bootstraping instead 
of a theoretical in-control distribution.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions. 
This has to be a range to be compatible with `arl_sop()` which uses threading and multi-processing.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `p_mat::Array{Float64,2}`: A matrix with the values of the relative frequencies 
of each d1-d2 (delay) combination. This matrix will be used for re-sampling.
"""
function rl_sop_bootstrap(p_mat::Array{Float64,2}, lam, cl, reps_range::UnitRange{Int}, chart_choice, refinement=false, rl_max::Int=typemax(Int))

  # Pre-allocate
  n_size = refinement ? 6 : 3
  p_hat = zeros(n_size)

  rls = zeros(Int, length(reps_range))
  p_vec_mean = vec(mean(p_mat, dims=1))
  p_ewma = similar(p_vec_mean)
  p_ewma .= p_vec_mean

  # Set initial value for test statistic
  stat = chart_stat_sop(p_ewma, chart_choice)
  stat0 = stat

  # Compute index to sample from (1 to number of rows ("pictures") in p_mat)
  range_index = axes(p_mat, 1)

  # Loop over repetitions
  for r in axes(reps_range, 1)
    p_ewma .= p_vec_mean
    stat = stat0
    rl = 0

    while abs(stat - stat0) < cl
      rl += 1

      # sample from p_vec
      index = rand(range_index)

      # Compute frequencies of SOPs
      @views p_hat .= p_mat[index, :]

      # Apply EWMA to p-vectors      
      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      # Compute test statistic
      stat = chart_stat_sop(p_ewma, chart_choice)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end

    rls[r] = rl
  end

  return rls
end

