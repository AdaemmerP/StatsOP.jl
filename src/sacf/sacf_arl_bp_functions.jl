
"""
    arl_sacf_bp(lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000)

Compute the in-control average run length (ARL) using the spatial autocorrelation function (SACF) for the BP-statistic. The input arguments are:  

- `lam`: The smoothing parameter for the exponentially weighted moving average 
(EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `sp_dgp`: The in-control spatial data generating process (DGP) to use for the 
SACF function.
- `d1_vec::Vector{Int}`: The first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The second (column) delays for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf_bp(lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000)

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

"""
    rl_sacf_bp(
  lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, p_reps::UnitRange, dist_error::UnivariateDistribution
)

Compute the out-of-control run length using the spatial autocorrelation function 
(SACF) for the BP-statistic. The function returns the run length for a given control 
  limit `cl` and a given number of repetitions `reps`. The input arguments are:

- `lam`: The smoothing parameter for the exponentially weighted moving average 
(EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `sp_dgp::ICSP`: The in-control spatial data generating process (DGP) to use for 
the SACF function.
- `d1_vec::Vector{Int}`: The first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The second (column) delays for the spatial process.
- `p_reps::UnitRange`: The number of repetitions to compute the run length. This 
has to be a unit range of integers to allow for parallel processing, since the 
  function is called by `arl_sacf()`.
- `dist_error::UnivariateDistribution`: The distribution to use for the error term 
in the spatial process. This can be any univariate distribution from the `Distributions.jl` package.
"""
function rl_sacf_bp(
  lam, cl, sp_dgp::ICSP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, p_reps::UnitRange, dist_error::UnivariateDistribution
)

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

""" 

    arl_sacf_bp(
  lam, cl, spatial_dgp::SpatialDGP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000
)

Compute the out-of-control average run length (ARL) using the spatial autocorrelation 
function (SACF) for the BP-statistic. The function returns the ARL for a given control 
  limit `cl` and a given number of repetitions `reps`. The input arguments are:

- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) 
control chart.
- `cl`: The control limit for the EWMA control chart.
- `spatial_dgp::SpatialDGP`: The spatial data generating process (DGP) to use for 
the SACF function. This can be one of the following: `SAR1`, `SAR11`, `SAR22`, 
  `SINAR11`, `SQMA11`, `SQINMA11`, or `BSQMA11`.
- `d1_vec::Vector{Int}`: The first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The second (column) delays for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf_bp(
  lam, cl, spatial_dgp::SpatialDGP, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000
)

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

"""
    rl_sacf_bp(
  lam, cl, d1_vec::Vector{Int}, d2_vec::Vector{Int}, p_reps::UnitRange, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing}
)

Compute the out-of-control run length using the spatial autocorrelation function 
(SACF) for the BP-statistic. The function returns the run length for a given control 
  limit `cl` and a given number of repetitions `reps`. The input arguments are:

- `lam`: The smoothing parameter for the exponentially weighted moving average 
(EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `d1_vec::Vector{Int}`: The first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The second (column) delays for the spatial process.
- `p_reps::UnitRange`: The number of repetitions to compute the run length. This 
has to be a unit range of integers to allow for parallel processing, since the function 
  is called by `arl_sacf()`.
- `spatial_dgp::SpatialDGP`: The spatial data generating process (DGP) to use for 
the SACF function. This can be one of the following: `SAR1`, `SAR11`, `SAR22`, 
  `SINAR11`, `SQMA11`, `SQINMA11`, or `BSQMA11`.
- `dist_error::UnivariateDistribution`: The distribution to use for the error 
term in the spatial process. This can be any univariate distribution from the 
`Distributions.jl` package.
- `dist_ao::Union{UnivariateDistribution,Nothing}`: The distribution to use for 
the additive outlier term in the spatial process. This can be any univariate ,
distribution from the `Distributions.jl` package or `Nothing`.
"""
function rl_sacf_bp(
  lam, cl, d1_vec::Vector{Int}, d2_vec::Vector{Int}, p_reps::UnitRange, spatial_dgp::SpatialDGP, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing}
)

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
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2, mat2)
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
