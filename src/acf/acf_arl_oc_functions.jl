export arl_acf_oc

"""
    arl_acf(lam, cl, acf_dgp, reps=10000)

Function to compute the average run length (ARL) for a specified DGP using the ACF statistic by XXX.

- `lam::Float64`: Smoothing parameter for the EWMA statistic.
- `cl::Float64`: Control limit for the ACF statistic.
- `acf_dgp::Union{IC, AR1, TEAR1}`: DGP.
- `reps::Int64`: Number of replications.  

```julia
arl_acf(0.1, 3.0, IC(Normal(0, 1)), 10000)
```
"""
function arl_acf_oc(lam, cl, acf_dgp, dist_null, reps, acf_version)

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_acf_oc(lam, cl, i, acf_dgp, acf_dgp.dist, dist_null, acf_version)

    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_acf(lam, cl, i, acf_dgp, acf_dgp.dist, dist_null, acf_version)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_acf(lam, cl, p_reps, acf_dgp)

Function to compute the run length (RL) for a specified DGP using the ACF statistic by XXX.
  
- `lam::Float64`: Smoothing parameter for the EWMA statistic.
- `cl::Float64`: Control limit for the ACF statistic.
- `p_reps::Vector{Int64}`: Unit range for number of replications.
- `acf_dgp::Union{IC, AR1, TEAR1}`: DGP.

```julia
rl_acf(0.1, 3.0, 10_000, IC(Normal(0, 1)))
```
"""
function rl_acf_oc(lam, cl, p_reps, acf_dgp, acf_dgp_dist, dist_null, acf_version)

  # Pre-allocate 
  rls = Vector{Int64}(undef, length(p_reps))
  x_vec = Vector{Float64}(undef, 2)

  for r in 1:length(p_reps)

    # initialize values
    # Convert all values to ensure type stability
    if acf_version == 1
      rl = 0
      c_0 = 0.0
      s_0 = var(dist_null)
      m_0 = mean(dist_null)
      acf_stat = 0.0
      μ0 = mean(dist_null)
      σ0 = std(dist_null)

    elseif acf_version == 2
      rl = 0
      c_0 = mean(dist_null)^2
      s_0 = var(dist_null) + mean(dist_null)^2
      m_0 = mean(acf_dgp_dist)
      acf_stat = 0.0
      μ0 = mean(dist_null)
      σ0 = std(dist_null)

    elseif acf_version == 3
      rl = 0
      c_0 = 0.0
      # --- not necessary but still ensure type stability
      s_0 = 0.0
      m_0 = 0.0
      acf_stat = 0.0
      μ0 = mean(dist_null)
      σ0 = std(dist_null)

    end

    # initialize sequence depending on DGP
    init_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)

    # set ACF statistic to zero
    acf_stat = 0

    while abs(acf_stat) < cl

      # increase run length
      rl += 1

      # compute EWMA ACF
      if acf_version == 1

        # Equation (3), page 3 in the paper
        c_0 = lam * (x_vec[2] - μ0) * (x_vec[1] - μ0) + (1.0 - lam) * c_0
        s_0 = lam * (x_vec[2] - μ0)^2 + (1.0 - lam) * s_0
        acf_stat = c_0 / s_0

      elseif acf_version == 2
        # Equation (4), page 3 in the paper
        c_0 = lam * x_vec[2] * x_vec[1] + (1.0 - lam) * c_0
        s_0 = lam * x_vec[2]^2 + (1.0 - lam) * s_0
        m_0 = lam * x_vec[2] + (1.0 - lam) * m_0
        acf_stat = (c_0 - m_0^2) / (s_0 - m_0^2)

      elseif acf_version == 3
        # Equation (5), page 3 in the paper
        c_0 = lam * (x_vec[2] - μ0) * (x_vec[1] - μ0) + (1 - lam) * c_0
        acf_stat = c_0 / σ0^2

      end


      # update x_vec depending on DGP
      update_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)
    end

    rls[r] = rl
  end
  return rls
end

