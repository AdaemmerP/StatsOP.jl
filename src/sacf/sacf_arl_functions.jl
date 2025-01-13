
#========================================================================

Multiple Dispatch for 'arl_sacf()' and 'rl_sacf()':
  1. In-control distributions 
  2. Out-of-control DGPs
  3. BP-statistics for IC and OOC 

========================================================================#
"""
    rl_sacf(
  lam, cl, d1::Int, d2::Int, p_reps::UnitRange, spatial_dgp::ICSP, dist_error::UnivariateDistribution
)

Compute the in-control run length using the spatial autocorrelation function (SACF) 
for a delay (d1, d2) combination. The function returns the run length for a given control limit `cl`.

The input arguments are:
  
- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `p_reps::UnitRange`: The number of repetitions to compute the run length. This 
has to be a unit range of integers to allow for parallel processing, since the 
  function is called by `arl_sacf()`.
- `spatial_dgp::ICSP`: The in-control spatial data generating process (DGP) to 
use for the SACF function.
- `dist_error::UnivariateDistribution`: The distribution to use for the error term in the 
spatial process. This can be any univariate distribution from the `Distributions.jl` package.
"""
function rl_sacf(
  spatial_dgp::ICSP, lam, cl, d1::Int, d2::Int, p_reps::UnitRange, dist_error::UnivariateDistribution
)

  # Extract matrix size and pre-allocate matrices
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
   arl_sacf(lam, cl, spatial_dgp::ICSP, d1::Int, d2::Int, reps=10_000)

Compute the in-control average run length (ARL), using the spatial autocorrelation 
function (SACF) for a delay (d1, d2) combination. The function returns the ARL 
for a given control limit `cl` and a given number of repetitions `reps`. 
    
The input arguments are:

- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `spatial_dgp`: The in-control spatial data generating process (DGP) to use for the SACF function.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf(spatial_dgp::ICSP, lam, cl, d1::Int, d2::Int, reps=10_000)

  # Extract        
  dist_error = spatial_dgp.dist

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(spatial_dgp, lam, cl, d1, d2, i,dist_error)
    end

  elseif nprocs() > 1 # Multi Processing

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      Threads.@spawn rl_sacf(spatial_dgp, lam, cl, d1, d2, i,dist_error)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sacf(
 spatial_dgp::SpatialDGP, lam, cl, d1::Int, d2::Int, p_reps::UnitRange, 
  dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing}
)

Compute the out-of-control run length using the spatial autocorrelation function (SACF). 
The function returns the run length for a given control limit `cl` and a given number of 
repetitions `reps`. 

The input arguments are:

- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) 
control chart.
- `cl`: The control limit for the EWMA control chart.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `p_reps::UnitRange`: The number of repetitions to compute the run length. This 
has to be a unit range of integers to allow for parallel processing, since the 
  function is called by `arl_sacf()`.
- `spatial_dgp::SpatialDGP`: The spatial data generating process (DGP) to use for 
the SACF function. This can be one of the following: `SAR1`, `SAR11`, `SAR22`, 
  `SINAR11`, `SQMA11`, `SQINMA11`, or `BSQMA11`.
"""
function rl_sacf(
  spatial_dgp::SpatialDGP, lam, cl, d1::Int, d2::Int, p_reps::UnitRange{Int}, 
  dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing}
)

  # pre-allocate  
  M = spatial_dgp.M_rows
  N = spatial_dgp.N_cols
  data = zeros(M, N)
  X_centered = similar(data)
  rls = zeros(Int, length(p_reps))

  # pre-allocate
  # mat:    matrix for the final values of the spatial DGP
  # mat_ao: matrix for additive outlier 
  # mat_ma: matrix for moving averages
  # vec_ar: vector for SAR(1) model
  # vec_ar2: vector for in-place multiplication for SAR(1) model
  if typeof(spatial_dgp) isa SAR1
    mat = build_sar1_matrix(spatial_dgp) # will be done only once
    mat_ao = zeros((M + 2 * spatial_dgp.margin), (N + 2 * spatial_dgp.margin))
    vec_ar = zeros((M + 2 * spatial_dgp.margin) * (N + 2 * spatial_dgp.margin))
    vec_ar2 = similar(vec_ar)
    mat2 = similar(mat_ao)
  elseif typeof(spatial_dgp) ∈ (SAR11, SINAR11, SAR22)
    mat = zeros(M + spatial_dgp.prerun, N + spatial_dgp.prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
    # Function to initialize matrix only for SAR(1,1) and SINAR(1,1) and SAR(2,2) 
    init_mat!(spatial_dgp, dist_error, mat) 
  elseif typeof(spatial_dgp) ∈ (SQMA11, SQINMA11)
    mat = zeros(M + 1, N + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
  elseif typeof(spatial_dgp) isa SQMA22
    mat = zeros(M + 2, N + 2)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
  elseif typeof(spatial_dgp) isa BSQMA11
    mat = zeros(M + 1, N + 1)
    mat_ma = zeros(M + 2, N + 2) # one extra row and column for "forward looking"
    mat_ao = similar(mat)
  end

  for r in axes(p_reps, 1)

    rho_hat = 0.0
    rl = 0

    while abs(rho_hat) < cl
      rl += 1

      # Fill matrix with dgp
      if spatial_dgp isa SAR1
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2, mat2)
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
    arl_sacf(lam, cl, sp_dgp::SpatialDGP, d1::Int, d2::Int, reps=10_000)

Compute the in-control average run length (ARL) using the spatial autocorrelation 
function (SACF) for a delay (d1, d2) combination and an out-of-control process. 
  
The input arguments are: 

- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `sp_dgp`: The spatial data generating process (DGP) to use for the SACF function. 
This can be one of the following: `SAR1`, `SAR11`, `SAR22`, `SINAR11`, `SQMA11`, 
`SQINMA11`, or `BSQMA11`.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `reps`: The number of repetitions to compute the ARL.
"""
function arl_sacf(sp_dgp::SpatialDGP, lam, cl, d1::Int, d2::Int, reps=10_000)

  # extract m and n from spatial_dgp
  dist_error = sp_dgp.dist
  dist_ao = sp_dgp.dist_ao

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(sp_dgp, lam, cl, d1, d2, i, dist_error, dist_ao)
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
