"""
    sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)

Compute the spatial autocorrelation function (SACF) for a given matrix `data`. The function returns the SACF for the first lag (ρ(1, 1)). The input arguments are:

- `data`: A matrix of size M x N.
- `cdata`: A matrix of size M x N to store the demeaned data.
- `cx_t_cx_t1`: A matrix of size M x N to store the element-wise multiplication of the current and lagged data.

"""
function sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)

  # sizes and demeaned data
  M = size(data, 1)
  N = size(data, 2)
  x_bar = mean(data)
  cdata .= data .- x_bar

  # slices and multiplications
  @views cx_t = cdata[2:M, 2:N]
  @views cx_t1 = cdata[1:(M-1), 1:(N-1)]
  cx_t_cx_t1 .= cx_t .* cx_t1
  cdata_sq .= cdata .^ 2

  # return ρ(1, 1)
  if allequal(data)
    return 0
  else
    return sum(cx_t_cx_t1) / (sum(cdata_sq))
  end
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
function rl_sacf(m::Int, n::Int, lam, cl, p_reps::UnitRange, dist_error::UnivariateDistribution)

  # pre-allocate
  data = zeros(m + 1, n + 1)
  rls = zeros(Int, length(p_reps))

  cdata = zeros(m + 1, n + 1)
  cdata_sq = similar(cdata)
  cx_t_cx_t1 = zeros(m, n)

  for r in 1:length(p_reps)

    p_hat = 0
    rl = 0

    while abs(p_hat) < cl
      rl += 1

      # fill matrix with iid N(0,1) values
      rand!(dist_error, data)

      # compute ρ(1,1)-EWMA
      p_hat = (1 - lam) * p_hat + lam * sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)

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
function rl_sacf(m::Int, n::Int, lam, cl, p_reps::UnitRange, spatial_dgp, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing})

  # pre-allocate
  data = zeros(m + 1, n + 1)
  rls = zeros(Int, length(p_reps))
  cdata = zeros(m + 1, n + 1)
  cdata_sq = similar(cdata)
  cx_t_cx_t1 = zeros(m, n)
  rls = zeros(Int, length(p_reps))

  # pre-allocate mat, mat_ao and mat_ma
  if spatial_dgp isa SAR1
    mat = build_sar1_matrix(spatial_dgp) # will be done only once
    mat_ao = zeros((m + 1 + 2 * spatial_dgp.margin), (n + 1 + 2 * spatial_dgp.margin))
    vec_ar = zeros((m + 1 + 2 * spatial_dgp.margin) * (n + 1 + 2 * spatial_dgp.margin)) # this is a vector but naming is for 
    vec_ar2 = similar(vec_ar)
  elseif spatial_dgp isa BSQMA11
    mat = zeros(m + spatial_dgp.prerun + 1, n + spatial_dgp.prerun + 1)
    mat_ma = zeros(m + spatial_dgp.prerun + 1 + 1, n + spatial_dgp.prerun + 1 + 1) # add one more row and column for "forward looking" BSQMA11
    mat_ao = similar(mat)
  else
    mat = zeros(m + spatial_dgp.prerun + 1, n + spatial_dgp.prerun + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)
  end

  for r in 1:length(p_reps)

    # Re-initialize matrix 
    if spatial_dgp isa SAR1
      # do nothing, 'mat' will not be overwritten for SAR1
    else
      fill!(mat, 0)
      init_mat!(spatial_dgp, dist_error, spatial_dgp.dgp_params, mat)
    end

    p_hat = 0
    rl = 0

    while abs(p_hat) < cl
      rl += 1

      # Fill matrix with dgp 
      if spatial_dgp isa SAR1
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2)
      else
        data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)
      end

      p_hat = (1 - lam) * p_hat + lam * sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)


    end

    rls[r] = rl
  end
  return rls
end


"""
    arl_sacf(m::Int, n::Int, lam, cl, reps::Int, dist_error)

Compute the in-control average run length (ARL), using the spatial autocorrelation function (SACF) for a lag length of 1. The function returns the ARL for a given control limit `cl` and a given number of repetitions `reps`. The input arguments are:    

- `m::Int`: The number of rows in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).  
- `n::Int`: The number of columns in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `lam`: The smoothing parameter for the exponentially weighted moving average (EWMA) control chart.
- `cl`: The control limit for the EWMA control chart.
- `reps`: The number of repetitions to compute the ARL.
- `dist_error`: The distribution to use for the error term in the SACF function. This can be any univariate distribution from the `Distributions.jl` package or a custom distribution with a defined method for `rand()` and `rand!()`.

```julia
#--- Example
# Set parameters
m = 10
n = 10
lam = 0.1
cl = .001
reps = 100
dist_error = Normal(0, 1)

# Compute average run length
arl = arl_sacf(m, n, lam, cl, reps, dist_error)
```
"""
function arl_sacf(m::Int, n::Int, lam, cl, reps::Int, dist_error::UnivariateDistribution)

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(m, n, lam, cl, i, dist_error)
    end

  elseif nprocs() > 1 # Multi Processing

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sacf(m, n, lam, cl, i, dist_error)
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
function arl_sacf(lam, cl, reps, spatial_dgp, dist_error::UnivariateDistribution, dist_ao::Union{UnivariateDistribution,Nothing})

  # extract m and n from spatial_dgp
  m = spatial_dgp.m
  n = spatial_dgp.n

  # Check whether to use threading or multi processing
  if nprocs() == 1 # Threading

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_sacf(m, n, lam, cl, i, spatial_dgp, dist_error, dist_ao)
    end

  elseif nprocs() > 1 # Multi Processing

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_sacf(m, n, lam, cl, i, spatial_dgp, dist_error, dist_ao)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    cl_sacf(m, n, lam, L0, reps, clinit, jmin, jmax, verbose, dist_error)

Compute the control limit for the exponentially weighted moving average (EWMA) control chart using the spatial autocorrelation function (SACF) for a lag length of 1. The function returns the control limit for a given average run length (ARL) `L0` and a given number of repetitions `reps`. The input arguments are:
 
- `m::Int`: The number of rows in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `n::Int`: The number of columns in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `lam`: The smoothing parameter for the EWMA control chart.
- `L0`: The average run length (ARL) to use for the control limit.
- `reps`: The number of repetitions to compute the ARL.
- `clinit`: The initial control limit to use for the EWMA control chart. If set to 0, the function will search for the control limit that gives an ARL greater than `L0`.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose`: A boolean to indicate whether to print the control limit and ARL for each iteration.
- `dist_error`: The distribution to use for the error term in the SACF function. This can be any univariate distribution from the `Distributions.jl` package or a custom distribution with a defined method for `rand()` and `rand!()`.

```julia
#--- Example
# Set parameters
m = 10
n = 10
lam = 0.1
L0 = 370
reps = 1000
clinit = 0.05
jmin = 4
jmax = 7
verbose = true
dist_error = Normal(0, 1)

# Compute control limit
cl = cl_sacf(m, n, lam, L0, reps, clinit, jmin, jmax, verbose, dist_error)
```
"""
function cl_sacf(m::Int, n::Int, lam, L0, reps::Int, clinit::Float64, jmin, jmax, verbose, dist_error)
  L1 = zeros(2)
  ii = Int       # set inital value depending on λbda
  if clinit == 0
    for i in 1:50
      L1 = arl_sacf(m, n, lam, i / 10, reps, dist_error)
      if verbose
        println("cl = ", i / 10, "\t", "ARL = ", L1[1])
      end
      if L1[1] > L0
        ii = i
        break
      end
    end
    clinit = ii / 50
  end

  for j in jmin:jmax
    for dh in 1:80
      clinit = clinit + (-1)^j * dh / 10^j
      L1 = arl_sacf(m, n, lam, clinit, reps, dist_error)
      if verbose
        println("cl = ", clinit, "\t", "ARL = ", L1[1])
      end
      if (j % 2 == 1 && L1[1] < L0) || (j % 2 == 0 && L1[1] > L0)
        break
      end
    end
    clinit = clinit
  end
  if L1[1] < L0
    cl = clinit + 1 / 10^jmax
  end
  return clinit
end





