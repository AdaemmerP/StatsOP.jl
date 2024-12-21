
# ---------------------------------------------------------------------------#
# --------------------------- SACF for lags 1, 1 --------------------------- #
# ---------------------------------------------------------------------------#

# """
#     sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)

# Compute the spatial autocorrelation function (SACF) for a given matrix `data`. The function returns the SACF for the first lag (ρ(1, 1)). The input arguments are:

# - `data`: A matrix of size M x N.
# - `cdata`: A matrix of size M x N to store the demeaned data.
# - `cx_t_cx_t1`: A matrix of size M x N to store the element-wise multiplication of the current and lagged data.

# """
# function sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)

#   # sizes and demeaned data
#   M = size(data, 1)
#   N = size(data, 2)
#   x_bar = mean(data)
#   cdata .= data .- x_bar

#   # slices and multiplications
#   @views cx_t = cdata[2:M, 2:N]
#   @views cx_t1 = cdata[1:(M-1), 1:(N-1)]
#   cx_t_cx_t1 .= cx_t .* cx_t1
#   cdata_sq .= cdata .^ 2

#   # return ρ(1, 1)
#   if allequal(data)
#     return 0.0
#   else
#     return sum(cx_t_cx_t1) / (sum(cdata_sq))
#   end
# end


"""
    sacf(data, cdata, covs, d1::Int=5, d2::Int=5)

- `data`: The picture matrix.
- `cdata`: A matrix store the demeaned data. Has to  be the same size as `data`.
- `covs`: A matrix to store the covariance products. Has to be of size `(d1 + 1) x (d2 + 1)`.
- `d1::Int`: The number of row delays to consider for the SACF. The default is one.
- `d2::Int`: The number of column delays to consider for the SACF. The default is one.    
"""
# Spatial ACF 
function sacf(X_centered, d1::Int, d2::Int)
  
  M = size(X_centered, 1)
  N = size(X_centered, 2)

  # Lag 0x0
  @views cov_00 = dot(X_centered[1:M, 1:N], X_centered[1:M, 1:N]) / (M * N)
  # Lag d1xd2
  @views cov_d1d2 = dot(X_centered[1:(M-d1), 1:(N-d2)], X_centered[(1+d1):M, (1+d2):N]) / (M * N)

  # Return the SACF value
  if allequal(X_centered)
    return 0.0
  else
    return cov_d1d2 / cov_00
  end
end


# # Spatial ACF 
# function sacf(data, cdata, covs, d1::Int, d2::Int)

#   cdata .= data .- mean(data)
#   M = size(cdata, 1)
#   N = size(cdata, 2)

#   # Loop to compute all sums of relevant products
#   for k in (0, d2) # 0:d2 -> we only need 0-0 and one particular combination
#     for l in (0, d1) # 0:d1 -> we only need 0-0 and one particular combination
#       # for (k, l) in zip((0, d2), (0, d1)) 
#       for j in 1:(N-l)
#         for i in 1:(M-k)
#           covs[l+1, k+1] += cdata[i+k, j+l] * cdata[i, j]
#         end
#       end
#     end
#   end

#   # Normalize by the number of elements
#   covs .= covs ./ (M * N)

#   # Compute the SACF value
#   sacf_val = covs[1+d1, 1+d2] / covs[1, 1]

#   # Return the SACF value
#   return sacf_val
# end

# Compute SACF for one picture and for integer delays
function stat_sacf(data::Union{SubArray,Matrix{T}}, d1::Int, d2::Int) where {T<:Real}

  # pre-allocate
  X_centered = data .- mean(data)  

  return sacf(X_centered, d1, d2)

end

# Compute B-Statistik for one picture
function stat_sacf(data::Union{SubArray,Matrix{T}}, d1_vec::Vector{Int}, d2_vec::Vector{Int}) where {T<:Real}

  # Compute all d1-d2 combinations
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)

  # pre-allocate
  X_centered = data .- mean(data)
  bp_stat = 0.0

  for (d1, d2) in d1_d2_combinations
    bp_stat += sacf(X_centered, d1, d2)^2
  end

  return bp_stat

end

# Compute SACF for multiple pictures
function stat_sacf(lam, data::Array{T,3}, d1::Int, d2::Int) where {T<:Real}

  # pre-allocate
  data_tmp = similar(data[:, :, 1])
  X_centered = zeros(size(data_tmp))
  rho_hat = 0.0
  sacf_vals = zeros(size(data, 3))

  # loop over pictures
  for i in axes(data, 3)
    data_tmp .= view(data, :, :, i)
    X_centered .= data_tmp .- mean(data_tmp)
    rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)
    sacf_vals[i] = rho_hat
  end

  return sacf_vals

end

# Compute B-Statistik for multiple pictures
function stat_sacf(lam, data::Array{T,3}, d1_vec::Vector{Int}, d2_vec::Vector{Int}) where {T<:Real}

  # ennsure tha 0 is not included in the d1_vec and d2_vec
  if 0 in d1_vec || 0 in d2_vec
    throw(ArgumentError("0 should not be included in d1_vec or d2_vec"))
  end

  # Compute all d1-d2 combinations
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)

  # pre-allocate
  X_centered = zeros(size(data[:, :, 1]))
  bp_stats = zeros(size(data, 3))
  rho_hat_all = zeros(length(d1_d2_combinations))
  bp_stat = 0.0

  # compute sequential BP-statistic
  for i in axes(data, 3)
    X_centered .= view(data, :, :, i) .- mean(view(data, :, :, i))    
    for (i, (d1, d2)) in enumerate(d1_d2_combinations)      
      rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, d1, d2)
      bp_stat += 2 * rho_hat_all[i]^2
    end
    bp_stats[i] = bp_stat
  end

  return bp_stats

end

"""
    rl_sacf(m::Int, n::Int, lam, cl, p_reps, dist_error)

Compute the in-control run length (RL) using the spatial autocorrelation function (SACF) for a lag length of 1. The function returns the RL for a given control limit `cl` and a given number of repetitions `p_reps`. The input arguments are:   
  
- `m::Int`: The number of rows in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `n::Int`: The number of columns in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `p_reps`: The number of repetitions to compute the RL. This has to be a unit range of integers to allow for parallel processing, since the function is called by `arl_sacf()`.
- `dist_error`: The distribution to use for the error term in the SACF function.

```julia-repl
#--- Example
# Set parameters
m = 10
n = 10
lam = 0.1
cl = .001
p_reps = 1:10
dist_error = Normal(0, 1)

# Compute run length
rls = rl_sacf(m, n, lam, cl, p_reps, dist_error)
```
"""
function rl_sacf(lam, cl, d1::Int, d2::Int,
  p_reps::UnitRange, spatial_dgp::ICSP, dist_error::UnivariateDistribution)

  # pre-allocate  
  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols
  data = zeros(M, N)
  X_centered = similar(data)
  rls = zeros(Int, length(p_reps))

  for r in axes(p_reps, 1)

    rho_hat = 0.0
    rl = 0

    while abs(rho_hat) < cl
      rl += 1

      # fill matrix with iid N(0,1) values
      rand!(dist_error, data)
      X_centered .= data .- mean(data)

      # compute ρ(d1,d2)-EWMA
      rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

    end

    rls[r] = rl

  end

  return rls

end

"""
    rl_sacf(m::Int, n::Int, lam, cl, p_reps::UnitRange, spatial_dgp, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution, Nothing})

Compute the out-of-control run length (RL) using the spatial autocorrelation function (SACF) for a lag length of 1. The function returns the RL for a given control limit `cl` and a given number of repetitions `p_reps`. The input arguments are:  

- `m::Int`: The number of rows in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).  
- `n::Int`: The number of columns in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `p_reps`: The number of repetitions to compute the RL. This has to be a unit range of integers to allow for parallel processing, since the function is called by `arl_sacf()`.
- `spatial_dgp`: The spatial data generating process (DGP) to use for the SACF function. This can be one of the following: `SAR1`, `SAR11`, `SINAR11`, `SQMA11`, `SQINMA11`, or `BSQMA11`. 
- `dist_error`: The distribution to use for the error term in the SACF function. This can be any univariate distribution from the `Distributions.jl` package with a defined mean or a custom distribution. 
- `dist_ao`: The distribution to use for for additive outliers. This can be any univariate distribution from the `Distributions.jl` package or a custom distribution.

```julia-repl
#--- Example
# Set parameters
m = 10
n = 10
lam = 0.1
cl = .001
p_reps = 1:10
spatial_dgp = SAR11((0.1, 0.1, 0.1), 10, 10, 100)
dist_error = Normal(0, 1)
dist_ao = nothing

# Compute run length
rls = rl_sacf(m, n, lam, cl, p_reps, spatial_dgp, dist_error, dist_ao)
```
"""
function rl_sacf(lam, cl, d1::Int, d2::Int, p_reps::UnitRange, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing})

  # pre-allocate  
  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols
  data = zeros(M, N)
  X_centered = similar(data)
  rls = zeros(Int, length(p_reps))

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
  elseif spatial_dgp isa BSQMA11
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = zeros(M + spatial_dgp.prerun + 1, N + spatial_dgp.prerun + 1) # add one extra row and column for "forward looking" BSQMA11
    mat_ao = similar(mat)
  else
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
  end

  for r in axes(p_reps, 1)

    # Re-initialize matrix 
    if spatial_dgp isa SAR1 # Add SQMA11, SQINMA11, BSQMA11 which also do not need re-initialization
    # do nothing, 'mat' will not be overwritten for SAR1
    else
      fill!(mat, 0)
      init_mat!(spatial_dgp, dist_error, spatial_dgp.dgp_params, mat)
    end

    rho_hat = 0
    rl = 0

    while abs(rho_hat) < cl
      rl += 1

      # Fill matrix with dgp 
      if spatial_dgp isa SAR1
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2)
      else
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)
      end

      # Demean data for SACF
      X_centered .= data .- mean(data)

      # Compute ρ(d1,d2)-EWMA
      rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

    end

    rls[r] = rl
  end
  return rls
end


"""
    arl_sacf(lam, cl, sp_dgp::ICSP, reps=10_000)

Compute the in-control average run length (ARL), using the spatial autocorrelation function (SACF) for a lag length of 1. The function returns the ARL for a given control limit `cl` and a given number of repetitions `reps`. The input arguments are:    

- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `sp_dgp`: The in-control spatial data generating process (DGP) to use for the SACF function. 
- `reps`: The number of repetitions to compute the ARL.


```julia
#--- Example
# Set parameters
lam = 0.1
cl = .1
reps = 100
sp_dgp = ICSP(10, 10, Normal(0, 1))

arls = arl_sacf(lam, cl, sp_dgp, reps)
```
"""
function arl_sacf(lam, cl, spatial_dgp::ICSP, d1::Int, d2::Int, reps=10_000)

  # Extract        
  dist_error = spatial_dgp.dist

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(lam, cl, d1, d2, i, spatial_dgp, dist_error)      
    end

  elseif nprocs() > 1 # Multi Processing

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sacf(lam, cl, d1, d2, i, spatial_dgp, dist_error)     
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

"""
    arl_sacf(lam, cl, reps, spatial_dgp, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing})

Compute the out-of-control average run length (ARL), using the spatial autocorrelation function (SACF) for a lag length of 1. The function returns the ARL for a given control limit `cl` and a given number of repetitions `reps`. The input arguments are:    

- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `reps`: The number of repetitions to compute the ARL.
- `spatial_dgp`: The spatial data generating process (DGP) to use for the SACF function. This can be one of the following: `SAR1`, `SAR11`, `SINAR11`, `SQMA11`, `SQINMA11`, or `BSQMA11`.
- `dist_error`: The distribution to use for the error term in the SACF function. This can be any univariate distribution from the `Distributions.jl` package with a defined mean or a custom distribution.
- `dist_ao`: The distribution to use for additive outliers. This can be any univariate distribution from the `Distributions.jl` package or a custom distribution.

```julia
#--- Example
# Set parameters
lam = 0.1
cl = .001
reps = 100
spatial_dgp = SAR11((0.1, 0.1, 0.1), 10, 10, 100)
dist_error = Normal(0, 1)
dist_ao = nothing

# Compute average run length
arl = arl_sacf(lam, cl, reps, spatial_dgp, dist_error, dist_ao)
```
"""
function arl_sacf(lam, cl, sp_dgp::SpatialDGP, d1::Int, d2::Int, reps=10_000)

  # extract m and n from spatial_dgp
  dist_error = sp_dgp.dist
  dist_ao = sp_dgp.dist_ao

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(lam, cl, d1, d2, i, sp_dgp, dist_error, dist_ao)
    end

  elseif nprocs() > 1 # Multi Processing

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sacf(lam, cl, d1, d2, i, sp_dgp, dist_error, dist_ao)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

########################################################################
########################## SACF BP-statistics ##########################
########################################################################
function arl_sacf(lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000)

  # Extract distribution        
  dist = sp_dgp.dist

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads()))

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(lam, cl, sp_dgp, d1_vec, d2_vec, i, dist)
    end

  elseif nprocs() > 1 # Multi Processing

    # Make chunks for separate tasks (based on number of workers)    
    chunks = Iterators.partition(1:reps, div(reps, nworkers()))

    par_results = pmap(chunks) do i
      rl_sacf(lam, cl, sp_dgp, d1_vec, d2_vec, i, dist)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

# Run-length function for in-control 
function rl_sacf(lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, p_reps::UnitRange, dist_error::UnivariateDistribution)

  # pre-allocate
  M = sp_dgp.M_rows
  N = sp_dgp.N_cols
  data = zeros(M, N)
  X_centered = similar(data)
  rls = zeros(Int, length(p_reps))

  # Compute all d1-d2 combinations
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)
  rho_hat_all = zeros(length(d1_d2_combinations))

  for r in axes(p_reps, 1)

    # rho_hat = 0.0
    bp_stat = 0.0

    rl = 0

    while bp_stat < cl
      rl += 1

      # fill data matrix 
      rand!(dist_error, data)

      # Demean data for SACF
      X_centered .= data .- mean(data)

      # compute BP-statistic using all d1-d2 combinations
      for (i, (d1, d2)) in enumerate(d1_d2_combinations)
        # compute ρ(d1,d2)-EWMA
        @views rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, d1, d2)
        bp_stat += 2 * rho_hat_all[i]^2
      end

    end

    rls[r] = rl

  end

  return rls

end

# Run-length function for out-of-control for BP-statistic
function arl_sacf(lam, cl, spatial_dgp::SpatialDGP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000)

  # Extract distribution        
  dist_error = spatial_dgp.dist
  dist_ao = spatial_dgp.dist_ao

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads()))

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(lam, cl, d1_vec, d2_vec, i, spatial_dgp, dist_error, dist_ao)
    end

  elseif nprocs() > 1 # Multi Processing

    # Make chunks for separate tasks (based on number of workers)    
    chunks = Iterators.partition(1:reps, div(reps, nworkers()))

    par_results = pmap(chunks) do i
      rl_sacf(lam, cl, d1_vec, d2_vec, i, spatial_dgp, dist_error, dist_ao)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

# Run-length function for out-of-control for BP-statistic
function rl_sacf(lam, cl, d1_vec::Vector{Int}, d2_vec::Vector{Int}, p_reps::UnitRange, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing})

  # pre-allocate
  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols
  data = zeros(M, N)
  X_centered = similar(data)
  rls = zeros(Int, length(p_reps))

  # Compute all d1-d2 combinations
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)
  rho_hat_all = zeros(length(d1_d2_combinations))

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
  elseif spatial_dgp isa BSQMA11
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = zeros(M + spatial_dgp.prerun + 1, N + spatial_dgp.prerun + 1) # add one extra row and column for "forward looking" BSQMA11
    mat_ao = similar(mat)
  else
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
  end

  # Loop over repetitions
  for r in axes(p_reps, 1)

    # Re-initialize matrix 
    if spatial_dgp isa SAR1 # Add SQMA11, SQINMA11, BSQMA11 which also do not need re-initialization
    # do nothing, 'mat' will not be overwritten for SAR1
    else
      fill!(mat, 0)
      init_mat!(spatial_dgp, dist_error, spatial_dgp.dgp_params, mat)
    end

    bp_stat = 0.0

    rl = 0

    while bp_stat < cl # BP-statistic can only be positive

      rl += 1

      # Fill matrix with dgp 
      if spatial_dgp isa SAR1
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2)
      else
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)
      end

      # Demean data for SACF
      X_centered .= data .- mean(data)

      # Compute BP-statistic using all d1-d2 combinations
      for (i, (d1, d2)) in enumerate(d1_d2_combinations)

        # compute ρ(d1,d2)-EWMA
        @views rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, d1, d2)
        bp_stat += 2 * rho_hat_all[i]^2

      end

    end

    rls[r] = rl
  end
  return rls
end
