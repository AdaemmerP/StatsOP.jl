"""
    chart_stat_sop(p_ewma, chart_choice)

Compute the the test statistic for spatial ordinal patterns. The first input is a vector with three values, based on SOP counts. The second input is the chart.     

"""
function chart_stat_sop(p_ewma, chart_choice)
  if chart_choice == 1
    chart_val = p_ewma[1] - 1.0 / 3.0
  elseif chart_choice == 2
    chart_val = p_ewma[2] - p_ewma[3]
  elseif chart_choice == 3
    chart_val = p_ewma[3] - 1.0 / 3.0
  elseif chart_choice == 4
    chart_val = p_ewma[1] - p_ewma[2]
  else
    println("Wrong number for test statistic.")
  end

  return chart_val
end

"""
Compute a 4D array to lookup the index of the sops. The original SOPs are based on ranks. Here we use sortperm which computes the order of the elements in the vector.
"""
function compute_lookup_array()

  p_sops = zeros(Int, 24, 4)
  sort_tmp = zeros(Int, 4)

  for (i, j) in enumerate(collect(permutations(1:4)))
    sortperm!(sort_tmp, j)
    @views p_sops[i, :] = sort_tmp
  end

  # Construct multi-dimensional lookup array 
  lookup_array = zeros(Int, 4, 4, 4, 4)

  for i in axes(p_sops, 1)
    @views lookup_array[p_sops[i, :][1], p_sops[i, :][2], p_sops[i, :][3], p_sops[i, :][4]] = i
  end

  return lookup_array

end

# In-place function to sort vector with sops
function order_vec!(x, ix)

  sortperm!(ix, x)

  return ix

end

# Lookup function --> chooses the index of the sop
function lookup_sop(lookup_array_sop, win)

  return @views lookup_array_sop[win[1], win[2], win[3], win[4]]

end


"""
    sop_frequencies(m::Int, n::Int, lookup_array_sop, data, sop)

Compute the frequencies of the spatial ordinal patterns. 
"""
function sop_frequencies(m::Int, n::Int, d1::Int, d2::Int, lookup_array_sop, data, sop)

  # Creat matrices to fill     
  freq_sops = zeros(Int, 24)
  win = zeros(Int, 4)

  # Loop through data to fill sop vector
  for j in 1:n
    for i in 1:m

      sop[1] = data[i, j]
      sop[2] = data[i, j+d2]
      sop[3] = data[i+d1, j]
      sop[4] = data[i+d1, j+d2]

      # Order 'sop_vec' in-place and save results in 'win'
      order_vec!(sop, win)
      # Get index for relevant pattern
      ind2 = lookup_sop(lookup_array_sop, win)
      # Add 1 to relevant pattern
      freq_sops[ind2] += 1
    end
  end

  return freq_sops

end

#--- Function to compute frequencies of sops
function sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop_vec, win, sop_freq)

  # Loop through data to fill sop vector
  for j in 1:n
    for i in 1:m

      sop_vec[1] = data[i, j]
      sop_vec[2] = data[i, j+d2]
      sop_vec[3] = data[i+d1, j]
      sop_vec[4] = data[i+d1, j+d2]

      # Order 'sop_vec' in-place and save results in 'win'
      order_vec!(sop_vec, win)
      # Get index for relevant pattern
      ind2 = lookup_sop(lookup_array_sop, win)
      # Add 1 to relevant pattern
      sop_freq[ind2] += 1
    end
  end

  return sop_freq

end


#===============================================

Multiple Dispatch for 'stat_sop()':
  1. data is only one picture -> data::Matrix{T}
  2. data is a three dimensional array -> data::Array{T, 3}
  3. ?

================================================#

# 1. Method to compute test statistic for one picture
"""
  stat_sop(data::Union{SubArray, Matrix{T}}; chart_choice) where {T<:Real}

Computes the test statistic for a single picture and chosen test statistic. `chart_coice` is an integer value for the chart choice. The options are 1-4.

# Examples
```julia-repl
data = rand(20, 20);

stat_sop(data, 2)
```
"""
function stat_sop(data::Union{SubArray,Matrix{T}}; chart_choice, d1=1, d2=2) where {T<:Real}

  # Compute 4 dimensional cube to lookup sops
  lookup_array_sop = compute_lookup_array()
  p_hat = zeros(3)
  sop = zeros(4)

  # Compute m and n based on data
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  # Compute frequencies of sops
  freq_sop = sop_frequencies(m, n, d1, d2, lookup_array_sop, data, sop)

  # Compute sum of frequencies for each group
  @views p_hat[1] = sum(freq_sop[[1, 3, 8, 11, 14, 17, 22, 24]])
  @views p_hat[2] = sum(freq_sop[[2, 5, 7, 9, 16, 18, 20, 23]])
  @views p_hat[3] = sum(freq_sop[[4, 6, 10, 12, 13, 15, 19, 21]])
  p_hat ./= m * n

  # Compute test statistic
  stat = chart_stat_sop(p_hat, chart_choice)

  return stat
end

# 2. Method to compute test statistic for multiple pictures
"""
   stat_sop(data::Array{Float64, 3}, add_noise::Bool, lam::Float64, chart_choice::Int)

Computes the test statistic for a 3D array of data, a given lambda value, and a given chart choice. The input parameters are:

- `data::Array{Float64,3}`: A 3D array of data.
- `add_noise::Bool`: A boolean value whether to add noise to the data. This is necessary when the matrices consist of count data.
- `lam::Float64`: A scalar value for lambda.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.

# Examples
```julia-repl
data = rand(20, 20, 10);
lam = 0.1;
chart_choice = 2;

stat_sop(data, false, lam, chart_choice)
```
"""
function stat_sop(lam, data::Array{T,3}; chart_choice, add_noise::Bool, d1=1, d2=1) where {T<:Real}

  # Compute 4 dimensional cube to lookup sops
  lookup_array_sop = compute_lookup_array()
  p_hat = zeros(3)
  sop = zeros(4)
  p_ewma = repeat([1.0 / 3.0], 3)
  stats_all = zeros(size(data, 3))
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # Compute m and n based on data
  m = size(data, 1) - 1
  n = size(data, 2) - 1

  for i = axes(data, 3)

    if add_noise
      data_tmp = data[:, :, i] + rand(m + 1, n + 1)
    else
      data_tmp = data[:, :, i]
    end

    # Compute frequencies of sops    
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

    # Compute sum of frequencies for each group
    if chart_choice in (1, 4) # Only need to compute for chart 1 and 4
      for i in s_1
        p_hat[1] += sop_freq[i]
      end
    end

    if chart_choice in (2, 4) # Only need to compute for chart 2 and 4 
      for i in s_2
        p_hat[2] += sop_freq[i]
      end
    end

    if chart_choice in (2, 3) # Only need to compute for chart 2 and 3
      for i in s_3
        p_hat[3] += sop_freq[i]
      end
    end

    # Compute relative frequencies
    p_hat ./= m * n

    # Compute test statistic
    @. p_ewma = (1 - lam) .* p_ewma .+ lam * p_hat

    stat_tmp = chart_stat_sop(p_ewma, chart_choice)

    # Save temporary test statistic
    stats_all[i] = stat_tmp

    # Reset win and sop_freq
    fill!(win, 0)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)

  end

  return stats_all

end

#========================================================================

Multiple Dispatch for 'arl_sop()':
  1. In-control distributions 
  2. Bootstraping using p_mat 
  3. Out-of-control DGPs

========================================================================#

"""
    arl_sop(lam, cl, sop_dgp::ICSP, reps=10_000; chart_choice=3, d=1)

Function to compute the average run length (ARL) for a given control-limit and in-control distribution. The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `sop_dgp::ICSP`: A struct for the in-control spatial DGP.
- `reps::Int`: An integer value for the number of repetitions.  
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. The default value is 3.
- `d::Int` An integer value for the embedding dimension. The default value is 1.
"""
function arl_sop(lam, cl, sop_dgp::ICSP, reps=10_000; chart_choice=3, d1=1, d2=1)

  # Extract values
  m = sop_dgp.M_rows - d1
  n = sop_dgp.N_cols - d2
  dist = sop_dgp.dist

  # Compute lookup array and number of sops
  lookup_array_sop = compute_lookup_array()

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
    arl_sop(lam, cl, reps, chart_choice=3, data::Array{Float64, 3})

Function to compute the average run length (ARL) using a bootstrap approach. The input parameters are:

- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `reps::Int`: An integer value for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. The default value is 3.
- `p_mat::Array{Float64, 3}`: A 3D array with the data. The data has to be in the form of a 3D array.
"""
function arl_sop(lam, cl, p_mat::Matrix{Float64}, reps=10_000; chart_choice=3)

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
    arl_sop(lam, cl, spatial_dgp, reps = 10_000; chart_choice=3, d = 1)

Function to compute the average run length (ARL) for a given out-of-control DGP. The input parameters are:
  
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `spatial_dgp::AbstractSpatialDGP`: A struct for type for the spatial DGP. This can be either `SAR11`, `SINAR11`, `SQMA11`, `BSQMA11` or `SAR1`. Look at their documentation for more information.
- `reps::Int`: An integer value for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. The default value is 3.
- `d::Int` An integer value for the embedding dimension. The default value is 1.
"""
function arl_sop(lam, cl, spatial_dgp::SpatialDGP, reps=10_000; chart_choice=3, d1::Int=1, d2::Int=1)

  # Compute m and n
  m_rows = spatial_dgp.M_rows - d1
  n_cols = spatial_dgp.N_cols - d2
  dist_error = spatial_dgp.dist
  dist_ao = spatial_dgp.dist_ao

  # Compute lookup array to finde SOPs
  lookup_array_sop = compute_lookup_array()

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect
    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_sop(lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice, m_rows, n_cols, d1, d2)

    end

  elseif nprocs() > 1 # Multi Processing

    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
    par_results = pmap(chunks) do i

      rl_sop(lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice, m_rows, n_cols, d1, d2)

    end

  end
  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

#========================================================================

Multiple Dispatch for 'rl_sop()':
  1. Computing run length for in-control distributions
  2. Computing run length for bootstraping 
  3. Computing run length for out-of-control DGPs

========================================================================#

"""
    rl_sop(m, n, lookup_array_sop, lam, cl, reps_range, chart_choice, dist)

A function to compute the run length for a given control limit and in-control distribution. The input parameters are:

- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops. This can be automatically computed using lookup_array_sop = `compute_lookup_array()`. This will be used to find the index of the sops.  
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions. This has to be a range to be compatible with `arl_sop()` which uses threading and multi-processing.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `dist::Distribution`: A distribution for the in-control data. Here you can use any univariate distribution from the `Distributions.jl` package.
"""
function rl_sop(lam, cl, lookup_array_sop, reps_range, dist, chart_choice, m, n, d1::Int, d2::Int)

  # Pre-allocate
  freq_sop = zeros(Int, 24) # factorial(4)
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
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop_vec, win, freq_sop)

      # Compute sum of frequencies for each group
      if chart_choice in (1, 4) # Only need to compute for chart 1 and 4
        for i in s_1
          p_hat[1] += freq_sop[i]
        end
      end

      if chart_choice in (2, 4) # Only need to compute for chart 2 and 4 
        for i in s_2
          p_hat[2] += freq_sop[i]
        end
      end

      if chart_choice in (2, 3) # Only need to compute for chart 2 and 3
        for i in s_3
          p_hat[3] += freq_sop[i]
        end
      end

      # Compute relative frequencies
      p_hat ./= m * n

      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      stat = chart_stat_sop(p_ewma, chart_choice)

      # Reset win and freq_sop
      fill!(win, 0)
      fill!(freq_sop, 0)
      fill!(p_hat, 0)
    end

    rls[r] = rl
  end
  return rls
end


"""
    rl_sop(lam, cl, reps_range, chart_choice, p_mat::Matrix{Float64})

A function to compute the run length for a given control limit using bootstraping instead of a theoretical in-control distribution. The input parameters are:
  
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `cl::Float64`: A scalar value for the control limit.
- `reps_range::UnitRange{Int}`: A range of integers for the number of repetitions. This has to be a range to be compatible with `arl_sop()` which uses threading and multi-processing.  
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `p_mat::Matrix{Float64}`: A matrix with the values of each pattern group obtained by `compute_p_mat()`. This matrix will be used for re-sampling
"""
function rl_sop(lam, cl, reps_range, chart_choice, p_mat::Matrix{Float64})

  # Pre-allocate
  p_hat = zeros(3)
  rls = zeros(Int, length(reps_range))
  p_vec_mean = vec(mean(p_mat, dims=1))
  range_index = axes(p_mat, 1)
  p_ewma = p_vec_mean
  stat = chart_stat_sop(p_ewma, chart_choice)
  stat0 = stat

  for r in 1:length(reps_range)
    p_ewma .= p_vec_mean
    stat = stat0
    rl = 0

    while abs(stat - stat0) < cl
      rl += 1

      # sample from p_vec
      index = rand(range_index)

      # Compute frequencies of SOPs
      @views p_hat[1] = p_mat[index, 1]
      @views p_hat[2] = p_mat[index, 2]
      @views p_hat[3] = p_mat[index, 3]

      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      stat = chart_stat_sop(p_ewma, chart_choice)
    end

    rls[r] = rl
  end
  return rls
end


"""
    rl_sop(m, n, lookup_array_sop, lam, cl, p_reps, chart_choice, spatial_dgp, dist_error, dist_ao)

Computes the run length for a given out-of-control DGP. The input parameters are:
  
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1.
- `lookup_array_sop::Array{Int, 4}`: A 4D array with the lookup array for the sops. This can be automatically computed using `lookup_array_sop = compute_lookup_array()`. This will be used to find the index of the sops.
- `lam`: A scalar value for lambda for the EWMA chart. This has to be between 0 and 1.
- `cl::Float64`: A scalar value for the control limit.
- `p_reps::UInt`: An unsigned integer value for the number of repetitions. This has to be an unsigned integer to be compatible with `arl_sop()` which uses threading and multi-processing.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.
- `spatial_dgp::AbstractSpatialDGP`: A struct for type for the spatial DGP. This can be either `SAR11`, `SINAR11`, `SQMA11`, `BSQMA11` or `SAR1`. Look at their documentation for more information.
- `dist_error::Distribution`: A distribution for the error term. Here you can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Distribution`: A distribution for the out-of-control data. Here you can use any univariate distribution from the `Distributions.jl` package.
"""
function rl_sop(lam, cl, lookup_array_sop, p_reps, spatial_dgp, dist_error::UnivariateDistribution, dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice, m, n, d1::Int, d2::Int)

  # pre-allocate
  freq_sop = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)
  data = zeros(m + d1, n + d2)
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
    mat = build_sar1_matrix(spatial_dgp) # will be done only once
    mat_ao = zeros((m + d1 + 2 * spatial_dgp.margin), (n + d2 + 2 * spatial_dgp.margin))
    vec_ar = zeros((m + d1 + 2 * spatial_dgp.margin) * (n + d2 + 2 * spatial_dgp.margin))
    vec_ar2 = similar(vec_ar)
  elseif spatial_dgp isa BSQMA11
    mat = zeros(m + spatial_dgp.prerun + d1, n + spatial_dgp.prerun + 1)
    mat_ma = zeros(m + spatial_dgp.prerun + d1 + 1, n + spatial_dgp.prerun + d2 + 1) # add one extra row and column for "forward looking" BSQMA11
    mat_ao = similar(mat)
  else
    mat = zeros(m + spatial_dgp.prerun + d1, n + spatial_dgp.prerun + d2)
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
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2)
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

      # Compute sum of frequencies for each pattern group
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, freq_sop)

      # Compute sum of frequencies for each group
      if chart_choice in (1, 4) # Only need to compute for chart 1 and 4
        for i in s_1
          p_hat[1] += freq_sop[i]
        end
      end

      if chart_choice in (2, 4) # Only need to compute for chart 2 and 4 
        for i in s_2
          p_hat[2] += freq_sop[i]
        end
      end

      if chart_choice in (2, 3) # Only need to compute for chart 2 and 3
        for i in s_3
          p_hat[3] += freq_sop[i]
        end
      end

      # Compute relative frequencies
      p_hat ./= m * n

      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      stat = chart_stat_sop(p_ewma, chart_choice)

      # Reset win and freq_sop
      fill!(win, 0)
      fill!(freq_sop, 0)
      fill!(p_hat, 0)
    end

    rls[r] = rl
  end
  return rls
end

#========================================================================

Multiple Dispatch for 'cl_sop()':
  1. Theoretical distribution
  2. Bootstraping

========================================================================#

"""
    cl_sop(lam, L0, sop_dgp::ICSP, cl_init, reps=10_000; chart_choice=3, jmin=4, jmax=6, verbose=false, d=1)

Compute the control limit for a given in-control distribution. The input parameters are:
  
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `L0::Float64`: A scalar value for the desired average run length.
- `sop_dgp::ICSP`: A struct for the in-control spatial DGP.
- `cl_init::Float64`: A scalar value for the initial control limit. This is used to find the control limit.
- `reps::Int`: An integer value for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. The default value is 3.
- `jmin::Int`: An integer value for the minimum value for the control limit.
- `jmax::Int`: An integer value for the maximum value for the control limit.
- `verbose::Bool`: A boolean value whether to print the control limit and the average run length.
- `d::Int` An integer value for the embedding dimension. The default value is 1.


```julia-repl
#-- Example
# Parameters
lam = 0.1
L0 = 370
sop_dgp = ICSP(20, 20, Normal(0, 1))
cl_init = 0.5
reps = 10_000
chart_choice = 2
jmin = 4
jmax = 6
verbose = true
d = 1
cl_sop(lam, L0, sop_dgp, cl_init, reps; chart_choice, jmin, jmax, verbose, d)
```
"""
function cl_sop(lam, L0, sop_dgp::ICSP, cl_init, reps=10_000; chart_choice=3, jmin=4, jmax=6, verbose=false, d1=1, d2=1)
  for j in jmin:jmax
    for dh in 1:40
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(lam, cl_init, sop_dgp, reps; chart_choice, d1=d1, d2=d2)[1]
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1)
      end
      if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
        break
      end
    end
    cl_init = cl_init
  end
  if L1 < L0
    cl_init = cl_init + 1 / 10^jmax
  end
  return cl_init
end


"""
    compute_p_mat(data::Array{Float64,3})

Compute the matrix of p-hat values for a given 3D array of data. These values are used for bootstrapping. 
"""
function compute_p_mat(data::Array{Float64,3}; d1=1, d2=1)

  # pre-allocate
  m = size(data, 1) - d1
  n = size(data, 2) - d2
  lookup_array_sop = compute_lookup_array()
  p_mat = zeros(size(data, 3), 3)
  p_hat = zeros(1, 3)
  sop = zeros(4)
  freq_sop = zeros(Int, 24)
  win = zeros(Int, 4)

  # pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # compute p_hat matrix
  for i in axes(data, 3)

    # Compute frequencies of sops
    @views data_tmp = data[:, :, i]
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, freq_sop)

    # Compute sum of frequencies for each group
    if chart_choice in (1, 4) # Only need to compute for chart 1 and 4
      for i in s_1
        p_hat[1] += freq_sop[i]
      end
    end

    if chart_choice in (2, 4) # Only need to compute for chart 2 and 4 
      for i in s_2
        p_hat[2] += freq_sop[i]
      end
    end

    if chart_choice in (2, 3) # Only need to compute for chart 2 and 3
      for i in s_3
        p_hat[3] += freq_sop[i]
      end
    end

    # Compute relative frequencies
    p_hat ./= m * n

    p_mat[i, :] = p_hat

    # Reset win and freq_sop
    fill!(win, 0)
    fill!(freq_sop, 0)
    fill!(p_hat, 0)
  end

  return p_mat
end

#--- Function to critical run length for SOP based on bootstraping
function cl_sop(lam, L0, p_mat, cl_init, reps=10_000; chart_choice=3,
  jmin=4, jmax=6, verbose=false, d1=1, d2=1)

  for j in jmin:jmax
    for dh in 1:40
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(lam, cl_init, p_mat, reps; chart_choice=chart_choice)[1]
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1)
      end
      if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
        break
      end
    end
    cl_init = cl_init
  end
  if L1 < L0
    cl_init = cl_init + 1 / 10^jmax
  end
  return cl_init
end



# Function to get sensible starting values for the control limit
function init_vals_sop(m, n, lam, chart_choice, dist, runs, p_quantile)

  # Pre-allocate
  lookup_array_sop = compute_lookup_array()
  freq_sop = zeros(24)
  win = zeros(Int, 4)
  data = zeros(m + 1, n + 1)
  p_ewma = zeros(3)
  p_hat = zeros(3)
  vals = zeros(runs)
  sop = zeros(4)

  fill!(p_ewma, 1.0 / 3.0)
  stat = chart_stat_sop(p_ewma, chart_choice)

  for i in 1:runs

    # Fill data with i.i.d data 
    rand!(dist, data)

    # Add noise when using random count data
    if dist isa DiscreteUnivariateDistribution
      for j in 1:size(data, 2)
        for i in 1:size(data, 1)
          data[i, j] = data[i, j] + rand()
        end
      end
    end

    # dist as in SACF functions!
    sop_frequencies!(m, n, lookup_array_sop, data, sop, win, freq_sop)

    @views p_hat[1] = sum(freq_sop[[1, 3, 8, 11, 14, 17, 22, 24]])
    @views p_hat[2] = sum(freq_sop[[2, 5, 7, 9, 16, 18, 20, 23]])
    @views p_hat[3] = sum(freq_sop[[4, 6, 10, 12, 13, 15, 19, 21]])
    p_hat ./= m * n # Divide each element of p_hat by m*n

    @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

    stat = chart_stat_sop(p_ewma, chart_choice)
    vals[i] = abs(stat)

    fill!(win, 0)
    fill!(freq_sop, 0)
  end

  quantile_val = quantile(vals, p_quantile)
  return_vec = (vals, quantile_val)

  return return_vec
end



# --------------------------------------------------#
# Functions for BP-Statistics
# --------------------------------------------------#



function arl_sop(lam, cl, spatial_dgp::SpatialDGP, reps=10_000; chart_choice, d1_vec::Vector{Int}, d2_vec::Vector{Int})

  # Compute m and n
  dist_error = spatial_dgp.dist
  dist_ao = spatial_dgp.dist_ao

  # Compute lookup array to finde SOPs
  lookup_array_sop = compute_lookup_array()

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect
    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_sop(lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice, d1_vec, d2_vec)

    end

  elseif nprocs() > 1 # Multi Processing

    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
    par_results = pmap(chunks) do i

      rl_sop(lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice, d1_vec, d2_vec)

    end

  end
  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


function rl_sop(lam, cl, lookup_array_sop, p_reps, spatial_dgp, dist_error::UnivariateDistribution, dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice, d1_vec::Vector{Int}, d2_vec::Vector{Int})

  # find maximum values of d1 and d2 for construction of matrices
  d1_max = maximum(d1_vec)
  d2_max = maximum(d2_vec)

  # Compute all possible combinations of d1 and d2
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)

  # pre-allocate
  freq_sop = zeros(Int, 24)
  win = zeros(Int, 4)
  data = zeros(m + d1_max, n + d2_max)
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
      mat = build_sar1_matrix(spatial_dgp) # will be done only once
      mat_ao = zeros((m + d1_max + 2 * spatial_dgp.margin), (n + d2_max + 2 * spatial_dgp.margin))
      vec_ar = zeros((m + d1_max + 2 * spatial_dgp.margin) * (n + d2_max + 2 * spatial_dgp.margin))
      vec_ar2 = similar(vec_ar)
  elseif spatial_dgp isa BSQMA11
      mat = zeros(m + spatial_dgp.prerun + d1_max, n + spatial_dgp.prerun + 1)
      mat_ma = zeros(m + spatial_dgp.prerun + d1_max + 1, n + spatial_dgp.prerun + d2_max + 1) # add one extra row and column for "forward looking" BSQMA11
      mat_ao = similar(mat)
  else
      mat = zeros(m + spatial_dgp.prerun + d1_max, n + spatial_dgp.prerun + d2_max)
      mat_ma = similar(mat)
      mat_ao = similar(mat)
  end

  for r in 1:length(p_reps)

      fill!(p_ewma, 1.0 / 3.0)
      stat = chart_stat_sop(p_ewma, chart_choice)
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
              data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2)
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
          for d1d2_tup in d1_d2_combinations

              d1_tmp = d1d2_tup[1]
              d2_tmp = d1d2_tup[2]
              m_tmp = spatial_dgp.M_rows - d1_tmp
              n_tmp = spatial_dgp.N_cols - d2_tmp 

              # Compute sum of frequencies for each pattern group
              sop_frequencies!(m_tmp, n_tmp, d1_tmp, d2_tmp, lookup_array_sop, data, sop, win, freq_sop)

              # Compute sum of frequencies for each group
              if chart_choice in (1, 4) # Only need to compute for chart 1 and 4
                  for i in s_1
                      p_hat[1] += freq_sop[i]
                  end
              end

              if chart_choice in (2, 4) # Only need to compute for chart 2 and 4 
                  for i in s_2
                      p_hat[2] += freq_sop[i]
                  end
              end

              if chart_choice in (2, 3) # Only need to compute for chart 2 and 3
                  for i in s_3
                      p_hat[3] += freq_sop[i]
                  end
              end

              # Compute relative frequencies
              p_hat ./= m * n

              @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

              stat = chart_stat_sop(p_ewma, chart_choice)
              bp_stat += stat^2

              # Reset win and freq_sop
              fill!(win, 0)
              fill!(freq_sop, 0)
              fill!(p_hat, 0)
          end
          # -------------------------------------------------------------------------------#
          # -------------------------------------------------------------------------------#
      end

      rls[r] = rl
  end
  return rls
end


function cl_sop(lam, L0, sop_dgp::ICSP, cl_init, reps=10_000; chart_choice, jmin=4, jmax=6, verbose=false, d1_vec::Vector{Int}, d2_vec::Vector{Int})

  L1 = zeros(2)
  ii = Int
  if cl_init == 0
    for i in 1:50
      L1 = arl_sop(lam, i / 50, sop_dgp, reps; chart_choice, d1_vec=d1_vec, d2_vec=d2_vec)
      if verbose
        println("cl = ", i / 50, "\t", "ARL = ", L1[1])
      end
      if L1[1] > L0
        ii = i
        break
      end
    end
    cl_init = ii / 50
  end

  for j in jmin:jmax
    for dh in 1:40
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(lam, cl_init, sop_dgp, reps; chart_choice, d1_vec=d1_vec, d2_vec=d2_vec)
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1[1])
      end
      if (j % 2 == 1 && L1[1] < L0) || (j % 2 == 0 && L1[1] > L0)
        break
      end
    end
    cl_init = cl_init
  end
  if L1[1] < L0
    cl = cl_init + 1 / 10^jmax
  end
  return cl_init
end

