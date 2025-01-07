#========================================================================

Multiple Dispatch for 'arl_sop()' and 'rl_sop()':
  1. In-control distributions 
  2. Bootstraping using pre-computed p-array
  3. Out-of-control DGPs
  4. BP-statistics for OOC and IC

========================================================================#

"""
    arl_sop(lam, cl, sop_dgp::ICSP, d1::Int, d2::Int, reps=10_000; chart_choice=3)

Compute the average run length (ARL) for a given in-control spatial DGP. 
  
The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `sop_dgp::ICSP`: A struct for the in-control spatial DGP.
- `d1::Int`: An integer value for the first delay (d₁).
- `d2::Int`: An integer value for the second delay (d₂).
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. 
The default value is 3.
"""
function arl_sop(lam, cl, sop_dgp::ICSP, d1::Int, d2::Int, reps=10_000; chart_choice=3)

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
      Threads.@spawn rl_sop(lam, cl, lookup_array_sop, i, dist, chart_choice, m, n, d1, d2)
    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sop(lam, cl, lookup_array_sop, i, dist, chart_choice, m, n, d1, d2)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop(lam, cl, lookup_array_sop, reps_range, dist, chart_choice, m, n, d1::Int, d2::Int)

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
function rl_sop(
  lam, cl, lookup_array_sop, reps_range::UInt, dist, chart_choice, m, n, d1::Int, d2::Int
)

  # Pre-allocate
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)
  data_tmp = zeros(m + d1, n + d2)
  p_ewma = zeros(3)
  p_hat = zeros(3)
  rls = zeros(Int, length(reps_range))
  sop_vec = zeros(4)

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

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
        for j in 1:size(data_tmp, 2)
          for i in 1:size(data_tmp, 1)
            data_tmp[i, j] = data_tmp[i, j] + rand()
          end
        end
      end

      # Compute frequencies of SOPs
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop_vec, win, sop_freq)

      # Fill 'p_hat' with sop-frequencies and compute relative frequencies
      fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

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



"""
    arl_sop(lam, cl, p_mat::Array{Float64,2}, reps=10_000)

Compute the average run length (ARL) using a bootstrap approach  for a particular
delay (d₁-d₂) combination. 

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `p_mat::Array{Float64,2}`: A matrix with the relative frequencies for a particular
delay (d₁-d₂) combination. This matrix will be used for re-sampling. The matrix has to be 
computed by `compute_p_array()`.
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
"""
function arl_sop(lam, cl, p_mat::Array{Float64,2}, reps=10_000)

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sop(lam, cl, i, chart_choice, p_mat)
    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sop(lam, cl, i, chart_choice, p_mat)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop(lam, cl, reps_range, chart_choice, p_mat::Array{Float64,2})

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
function rl_sop(lam, cl, reps_range, chart_choice, p_mat::Array{Float64,2})

  # Pre-allocate  
  p_hat = zeros(3)
  rls = zeros(Int, length(reps_range))
  p_vec_mean = vec(mean(p_mat, dims=1))
  p_ewma = p_vec_mean

  # Set initial value for test statistic
  stat = chart_stat_sop(p_ewma, chart_choice)
  stat0 = stat

  # Compute index to sample from (1 to number of rows ("pictures") in p_mat)
  range_index = axes(p_mat, 1)

  # Loop over repetitions
  for r in 1:length(reps_range)
    p_ewma .= p_vec_mean
    stat = stat0
    rl = 0

    while abs(stat - stat0) < cl
      rl += 1

      # sample from p_vec
      index = rand(range_index)

      # Compute frequencies of SOPs
      p_hat[1] = p_mat[index, 1]
      p_hat[2] = p_mat[index, 2]
      p_hat[3] = p_mat[index, 3]

      # Apply EWMA to p-vectors
      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      # Compute test statistic
      stat = chart_stat_sop(p_ewma, chart_choice)
    end

    rls[r] = rl
  end
  return rls
end


"""
    arl_sop(lam, cl, p_array::Array{Float64,3}, reps=10_000; chart_choice=3)

Compute the average run length for the BP-EWMA-SOP for a given control limit 
  using bootstraping instead of a theoretical in-control distribution.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `p_array::Array{Float64, 3}`: A 3D array with the with the relative frequencies 
of each d1-d2 (delay) combination. The first dimension (rows) is the picture, the 
second dimension refers to the patterns group (s₁, s₂, or s₃) and the third dimension 
denotes each d₁-d₂ combination. This matrix will be used for re-sampling.
- `reps::Int`: An integer value for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""
function arl_sop(lam, cl, p_array::Array{Float64,3}, reps=10_000; chart_choice=3)

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sop(lam, cl, i, chart_choice, p_array)
    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sop(lam, cl, i, chart_choice, p_array)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop(lam, cl, reps_range, chart_choice, p_array::Array{Float64,3})

Compute the EWMA-BP-SOP run length for a given control limit using bootstraping instead 
of a theoretical in-control distribution.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `p_array::Array{Float64,3}`: A 3D array with the with the relative frequencies for 
each d1-d2 (delay) combination. The first dimension (rows) is the picture, the second
dimension refers to the patterns group (s₁, s₂, or s₃) and the third dimension denotes
each d₁-d₂ combination. This array will be used for re-sampling.
"""
function rl_sop(lam, cl, reps_range::UnitRange, chart_choice, p_array::Array{Float64,3})

  # Pre-allocate
  p_hat = zeros(3)
  rls = zeros(Int, length(reps_range))
  p_array_mean = mean(p_array, dims=1)
  range_index = axes(p_array, 1)
  p_ewma = p_array_mean # will be dimension 1 x 3 x 'size(p_array, 3)'
  bp_stat0 = 0.0

  # compute bp-statistic with averages of p_array
  for i in 1:size(p_array, 3)
    @views bp_stat0 += chart_stat_sop(p_ewma[:, :, i], chart_choice)^2
  end

  # Loop over repetitions
  for r in 1:length(reps_range)
    p_ewma .= p_array_mean
    bp_stat = bp_stat0
    rl = 0

    while bp_stat < cl # (bp_stat - bp_stat0) < cl
      rl += 1

      # sample from p_vec
      index = rand(range_index)

      for i in 1:size(p_array, 3)

        p_hat[1] = p_array[index, 1, i]
        p_hat[2] = p_array[index, 2, i]
        p_hat[3] = p_array[index, 3, i]

        # Apply EWMA
        @views p_ewma[1, 1:3, i] .= (1 - lam) .* p_ewma[1, 1:3, i] .+ lam .* p_hat

        # Compute test statistic
        @views stat = chart_stat_sop(p_ewma[1, :, i], chart_choice)
        bp_stat += stat^2

      end

    end

    rls[r] = rl
  end
  return rls
end


"""
     arl_sop(
  lam, cl, spatial_dgp::SpatialDGP, d1::Int, d2::Int, reps=10_000; chart_choice=3
)

Compute the average run length (ARL) for a given out-of-control spatial DGP. 
  
The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `spatial_dgp::SpatialDGP`: A struct for the out-of-control spatial DGP.
- `d1::Int`: An integer value for the first delay (d₁).
- `d2::Int`: An integer value for the second delay (d₂).
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
The default value is 3.
"""
function arl_sop(
  lam, cl, spatial_dgp::SpatialDGP, d1::Int, d2::Int, reps=10_000; chart_choice=3
)

  # Compute m and n
  m_rows = spatial_dgp.M_rows - d1
  n_cols = spatial_dgp.N_cols - d2
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

      Threads.@spawn rl_sop(
        lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice,
        m_rows, n_cols, d1, d2
      )


    end

  elseif nprocs() > 1 # Multi Processing

    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
    par_results = pmap(chunks) do i

      rl_sop(
        lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice,
        m_rows, n_cols, d1, d2
      )

    end

  end
  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end



"""
    rl_sop(
  lam, cl, lookup_array_sop, p_reps, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution, 
  dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice, m, n, d1::Int, d2::Int
  )

Computes the run length for a given out-of-control DGP. The input parameters are:
  
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops,
which will be computed computed using `lookup_array_sop = compute_lookup_array_sop()`.
- `p_reps::UnitRange{Int}`: A range of integers for the number of repetitions.
- `spatial_dgp::SpatialDGP`: A struct for the out-of-control spatial DGP.
- `dist_error::UnivariateDistribution`: A distribution for the error term. Here you can use any
univariate distribution from the `Distributions.jl` package.
- `dist_ao::Union{Nothing,UnivariateDistribution}`: A distribution for the additive outlier.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `m::Int`: An integer value for the number of rows for the final "SOP" matrix.
- `n::Int`: An integer value for the number of columns for the final "SOP" matrix.
- `d1::Int`: An integer value for the first delay (d₁).
- `d2::Int`: An integer value for the second delay (d₂).
"""
function rl_sop(
  lam, cl, lookup_array_sop, p_reps::UnitRange, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution,
  dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice, m, n, d1::Int, d2::Int
)

  # pre-allocate
  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)
  data = zeros(M, N)
  p_ewma = zeros(3)
  p_hat = zeros(3)
  rls = zeros(Int, length(p_reps))
  sop = zeros(4)

  # pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # pre-allocate mat, mat_ao and mat_ma
  # mat:    matrix for the final values of the spatial DGP
  # mat_ao: matrix for additive outlier 
  # mat_ma: matrix for moving averages
  # vec_ar: vector for SAR(1) model
  # vec_ar2: vector for in-place multiplication for SAR(1) model
  if spatial_dgp isa SAR1
    mat::Matrix{Float64} = build_sar1_matrix(spatial_dgp) # will be done only once
    mat_ao = zeros((M + 2 * spatial_dgp.margin), (N + 2 * spatial_dgp.margin))
    vec_ar = zeros((M + 2 * spatial_dgp.margin) * (N + 2 * spatial_dgp.margin))
    vec_ar2 = similar(vec_ar)
    mat2 = similar(mat_ao)
  elseif spatial_dgp isa BSQMA11
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = zeros(M + spatial_dgp.prerun + 1, N + spatial_dgp.prerun + 1) # add one extra row and column for "forward looking" BSQMA11
    mat_ao = similar(mat)
  else
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
  end

  for r in 1:length(p_reps)

    fill!(p_ewma, 1.0 / 3.0)
    stat = chart_stat_sop(p_ewma, chart_choice)

    # Re-initialize matrix 
    if spatial_dgp isa SAR1
      # do nothing, 'mat' will not be overwritten for SAR1
    else
      fill!(mat, 0)
      init_mat!(spatial_dgp, dist_error, spatial_dgp.dgp_params, mat)
    end

    rl = 0

    while abs(stat) < cl
      rl += 1

      # Fill matrix with dgp 
      if spatial_dgp isa SAR1
        data .= fill_mat_dgp_sop!(
          spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2, mat2
        )
      else
        data .= fill_mat_dgp_sop!(
          spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma
        )
      end

      # Check whether to add noise to count data
      if dist_error isa DiscreteUnivariateDistribution
        for j in 1:size(data, 2)
          for i in 1:size(data, 1)
            data[i, j] = data[i, j] + rand()
          end
        end
      end

      # Compute sum of frequencies for each pattern group
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

      # Fill 'p_hat' with sop-frequencies and compute relative frequencies
      fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

      # Apply EWMA to p-vectors
      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      # Compute test statistic
      stat = chart_stat_sop(p_ewma, chart_choice)

      # Reset win, sop_freq and p_hat
      fill!(win, 0)
      fill!(sop_freq, 0)
      fill!(p_hat, 0)
    end

    rls[r] = rl
  end
  return rls
end

# Compute EWMA-BP-SOP for in-control processes
"""
    arl_sop(
  lam, cl, spatial_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000; chart_choice=3
)

Computes the average run length (ARL) for a given in-control spatial DGP, using 
EWMA-BP-SOP statistics.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `spatial_dgp::ICSP`: A struct for the in-control spatial DGP.
- `d1_vec::Vector{Int}`: A vector with integer values for the first delay (d₁).
- `d2_vec::Vector{Int}`: A vector with integer values for the second delay (d₂).
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""
function arl_sop(
  lam, cl, spatial_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000;
  chart_choice=3
)

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

      Threads.@spawn rl_sop(
        lam, cl, lookup_array_sop, spatial_dgp, i, dist_error, chart_choice, d1_vec, d2_vec
      )

    end

  elseif nprocs() > 1 # Multi Processing

    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
    par_results = pmap(chunks) do i

      rl_sop(
        lam, cl, lookup_array_sop, spatial_dgp, i, dist_error, chart_choice, d1_vec, d2_vec
      )

    end

  end
  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop(
  lam, cl, lookup_array_sop, spatial_dgp::ICSP, reps_range, dist, chart_choice, 
  d1_vec::Vector{Int}, d2_vec::Vector{Int}
)

Computes the run length for a given in-control spatial DGP, using the EWMA-BP-SOP statistic.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops,
which will be computed computed using `lookup_array_sop = compute_lookup_array_sop()`.
- `spatial_dgp::ICSP`: A struct for the in-control spatial DGP.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions.
- `dist::Distribution`: A distribution for the error term. Here you can use any
univariate distribution from the `Distributions.jl` package.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `d1_vec::Vector{Int}`: A vector with integer values for the first delay (d₁).
- `d2_vec::Vector{Int}`: A vector with integer values for the second delay (d₂).
"""
function rl_sop(
  lam, cl, lookup_array_sop, spatial_dgp::ICSP, reps_range::UnitRange, dist, chart_choice,
  d1_vec::Vector{Int}, d2_vec::Vector{Int}
)

  # Pre-allocate    
  sop = zeros(4)
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)
  p_hat = zeros(3)
  #p_ewma = zeros(3)
  rls = zeros(Int, length(reps_range))

  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols
  data = zeros(M, N)

  # Compute all possible combinations of d1 and d2
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)
  p_ewma_all = zeros(3, 1, length(d1_d2_combinations))
  p_ewma_all .= 1.0 / 3.0

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  for r in 1:length(reps_range)
    # fill!(p_ewma, 1.0 / 3.0)
    # stat = chart_stat_sop(p_ewma, chart_choice)

    bp_stat = 0.0
    rl = 0

    while bp_stat < cl
      rl += 1

      # Fill data 
      rand!(dist, data)

      # Add noise when using count data
      if dist isa DiscreteUnivariateDistribution
        for j in 1:size(data, 2)
          for i in 1:size(data, 1)
            data[i, j] = data[i, j] + rand()
          end
        end
      end

      # -------------------------------------------------#
      # -----------     Loop for BP-Statistik         ---#
      # -------------------------------------------------#
      for (i, (d1, d2)) in enumerate(d1_d2_combinations)

        m = spatial_dgp.M_rows - d1
        n = spatial_dgp.N_cols - d2

        # Compute sum of frequencies for each pattern group
        sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

        # Fill 'p_hat' with sop-frequencies and compute relative frequencies
        fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

        # Apply EWMA to p-vectors
        @views p_ewma_all[:, :, i] .= (1 - lam) .* p_ewma_all[:, :, i] .+ lam .* p_hat

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
      # -------------------------------------------------#
    end

    rls[r] = rl
  end
  return rls
end

"""

    arl_sop(
  lam, cl, spatial_dgp::SpatialDGP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000; chart_choice=3
)

Compute the average run length (ARL) for a given out-of-control spatial DGP, using 
the EWMA-BP-SOP statistic.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `spatial_dgp::SpatialDGP`: A struct for the out-of-control spatial DGP.
- `d1_vec::Vector{Int}`: A vector with integer values for the first delay (d₁).
- `d2_vec::Vector{Int}`: A vector with integer values for the second delay (d₂).
- `reps::Int`: An integer value for the number of repetitions. The default value is 10,000.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
"""

function arl_sop(
  lam, cl, spatial_dgp::SpatialDGP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000; chart_choice=3
)

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

      Threads.@spawn rl_sop(
        lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao,
        chart_choice, d1_vec, d2_vec
      )

    end

  elseif nprocs() > 1 # Multi Processing

    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
    par_results = pmap(chunks) do i

      rl_sop(
        lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao,
        chart_choice, d1_vec, d2_vec
      )

    end

  end
  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sop(
  lam, cl, lookup_array_sop, p_reps, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution,
  dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice, d1_vec::Vector{Int}, d2_vec::Vector{Int}
)

Computes the run length for a given out-of-control DGP, using the EWMA-BP-SOP statistic.

The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops,
which will be computed computed using `lookup_array_sop = compute_lookup_array_sop()`.
- `p_reps::UnitRange{Int}`: A range of integers for the number of repetitions.
- `spatial_dgp::SpatialDGP`: A struct for the out-of-control spatial DGP.
- `dist_error::UnivariateDistribution`: A distribution for the error term. Here you can use any
univariate distribution from the `Distributions.jl` package.
- `dist_ao::Union{Nothing,UnivariateDistribution}`: A distribution for the additive outlier.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `d1_vec::Vector{Int}`: A vector with integer values for the first delay (d₁).
- `d2_vec::Vector{Int}`: A vector with integer values for the second delay (d₂).
"""
function rl_sop(
  lam, cl, lookup_array_sop, p_reps::UnitRange, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution,
  dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice, d1_vec::Vector{Int}, d2_vec::Vector{Int}
)

  # find maximum values of d1 and d2 for construction of matrices
  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols

  # Compute all possible combinations of d1 and d2
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)

  # pre-allocate
  sop_freq = zeros(Int, 24)
  win = zeros(Int, 4)
  data = zeros(M, N)
  p_hat = zeros(3)
  rls = zeros(Int, length(p_reps))
  sop = zeros(4)
  #p_ewma = zeros(3)
  p_ewma_all = zeros(3, 1, length(d1_d2_combinations))
  p_ewma_all .= 1.0 / 3.0

  # pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # pre-allocate mat, mat_ao and mat_ma
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
  elseif spatial_dgp isa BSQMA11
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = zeros(M + spatial_dgp.prerun + 1, N + spatial_dgp.prerun + 1) # add one extra row and column for "forward looking" BSQMA11
    mat_ao = similar(mat)
  else # SAR11, SAR22, SINAR11, SQMA11
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
  end

  for r in axes(p_reps, 1)

    #fill!(p_ewma, 1.0 / 3.0)
    #stat = chart_stat_sop(p_ewma, chart_choice)

    bp_stat = 0.0

    # Re-initialize matrix 
    if spatial_dgp isa SAR1
      # do nothing, 'mat' will not be overwritten for SAR1
    else
      fill!(mat, 0)
      init_mat!(spatial_dgp, dist_error, spatial_dgp.dgp_params, mat)
    end

    rl = 0

    while bp_stat < cl # BP-statistic can only be positive
      rl += 1

      # Fill matrix with dgp 
      if spatial_dgp isa SAR1
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2, mat2)
      else
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)
      end

      # Check whether to add noise to count data
      if dist_error isa DiscreteUnivariateDistribution
        for j in 1:size(data, 2)
          for i in 1:size(data, 1)
            data[i, j] = data[i, j] + rand()
          end
        end
      end

      # -------------------------------------------------------------------------------#
      # ----------------     Loop for BP-Statistik                     ----------------#
      # -------------------------------------------------------------------------------#
      for (i, (d1, d2)) in enumerate(d1_d2_combinations)

        m = spatial_dgp.M_rows - d1
        n = spatial_dgp.N_cols - d2

        # Compute sum of frequencies for each pattern group
        sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

        # Fill 'p_hat' with sop-frequencies and compute relative frequencies
        fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

        # Apply EWMA
        @views p_ewma_all[:, :, i] .= (1 - lam) .* p_ewma_all[:, :, i] .+ lam .* p_hat

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

