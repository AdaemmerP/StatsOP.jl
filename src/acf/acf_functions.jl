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
function rl_acf(lam, cl, p_reps, acf_dgp, acf_version)

  # Pre-allocate 
  rls = Vector{Int64}(undef, length(p_reps))
  x_vec = Vector{Float64}(undef, 2)

  for r in 1:length(p_reps)

    # initialize values
    # Convert all values to ensure type stability
    if acf_version == 1
      rl = 0
      c_0 = 0.0
      s_0 = convert(Float64, acf_dgp.sig^2)
      m_0 = convert(Float64, acf_dgp.mu)
      acf_stat = 0.0

    elseif acf_version == 2
      rl = 0
      c_0 = convert(Float64, acf_dgp.mu^2)
      s_0 = convert(Float64, acf_dgp.sig^2) + convert(Float64, acf_dgp.mu^2)
      m_0 = convert(Float64, acf_dgp.mu)
      acf_stat = 0.0

    elseif acf_version == 3
      rl = 0
      c_0 = 0.0
      # --- not necessary but still ensure type stability
      s_0 = 0.0
      m_0 = 0.0
      acf_stat = 0.0

    elseif acf_version == 4
      rl = 0
      c_0 = 0.0
      s_0 = 1.0
      acf_stat = 0.0

    end

    # initialize sequence depending on DGP
    init_dgp_op!(acf_dgp, x_vec, acf_dgp.dist, 1)

    # set ACF statistic to zero
    acf_stat = 0

    while abs(acf_stat) < cl

      # increase run length
      rl += 1

      # compute EWMA ACF
      if acf_version == 1

        # Equation (3), page 3 in the paper
        c_0 = lam * (x_vec[1] - acf_dgp.mu) * (x_vec[2] - acf_dgp.mu) + (1.0 - lam) * c_0
        s_0 = lam * (x_vec[1] - acf_dgp.mu)^2 + (1.0 - lam) * s_0
        acf_stat = c_0 / s_0

      elseif acf_version == 2
        # Equation (4), page 3 in the paper
        c_0 = lam * x_vec[1] * x_vec[2] + (1.0 - lam) * c_0
        s_0 = lam * x_vec[1]^2 + (1.0 - lam) * s_0
        m_0 = lam * x_vec[1] + (1.0 - lam) * m_0
        acf_stat = (c_0 - m_0 * m_0) / (s_0 - m_0 * m_0)

      elseif acf_version == 3
        # Equation (5), page 3 in the paper
        c_0 = lam * (x_vec[1] - acf_dgp.mu) * (x_vec[2] - acf_dgp.mu) + (1 - lam) * c_0
        acf_stat = c_0 / acf_dgp.sig^2

      elseif acf_version == 4

        # Equation (7), page 346 in Weiss and Testik (2023)
        c_0 = lam * x_vec[1] * x_vec[2] + (1 - lam) * c_0
        s_0 = lam * x_vec[1]^2 + (1 - lam) * s_0
        acf_stat = c_0 / s_0

      end

      # update x_vec depending on DGP
      update_dgp_op!(acf_dgp, x_vec, acf_dgp.dist, 1)
    end

    rls[r] = rl
  end
  return rls
end

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
function arl_acf(lam, cl, acf_dgp, reps=10_000)

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i
      Threads.@spawn rl_acf(lam, cl, i, acf_dgp)

    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_acf(lam, cl, i, acf_dgp)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


# # -------------------------------------------------#
# # --------------- In-control methods---------------#
# # -------------------------------------------------#

# # Method to initialize in-control
# function init_dgp_acf!(dgp::ICTS, x_vec, dist_error)
#   rand!(dist_error, x_vec)
#   return nothing
# end

# # Method to update in-control
# function update_dgp_acf!(dgp::ICTS, x_vec, dist_error)
#   x_vec[1] = x_vec[2]
#   x_vec[2] = rand(dist_error)
#   return nothing
# end

# # -------------------------------------------------#
# # ---------------  AR(1) methods    ---------------#
# # -------------------------------------------------#

# # Method to initialize AR(1)
# function init_dgp_acf!(dgp::AR1, x_vec, dist_error)
#   x_vec[1] = rand(dist_error)
#   x_vec[2] = dgp.α * x_vec[1] + rand(Normal(0, sqrt(1 - (dgp.α^2))))
#   return nothing
# end

# # Method to update AR(1)
# function update_dgp_acf!(dgp::AR1, x_vec, dist_error)
#   x_vec[1] = x_vec[2]
#   x_vec[2] = dgp.α * x_vec[1] + rand(Normal(0, sqrt(1 - (dgp.α^2))))
#   return nothing
# end

# # -------------------------------------------------#
# # ---------------  TEAR(1) methods   --------------#
# # -------------------------------------------------#

# # Method to initialize TEAR(1) 
# function init_dgp_acf!(dgp::TEAR1, x_vec, dist_error)
#   y = rand(dist_error)

#   x_vec[1] = quantile(Normal(0, 1), cdf(dgp.dist, y))
#   y = (1 - dgp.α) * rand(dgp.dist) + rand(Bernoulli(dgp.α)) * y
#   x_vec[2] = quantile(Normal(0, 1), cdf(dgp.dist, y))

#   return nothing
# end

# # Method to update TEAR(1) for ACF
# function update_dgp_acf!(dgp::TEAR1, x_vec, dist_error)
#   x_vec[1] = x_vec[2]

#   # compute old "y" value to compute new "x" value
#   y_old = quantile(Exponential(1), cdf(Normal(0, 1), x_vec[2]))
#   y = (1 - dgp.α) * rand(dist_error) + rand(Bernoulli(dgp.α)) * y_old
#   x_vec[2] = quantile(Normal(0, 1), cdf(dgp.dist, y))
#   return nothing
# end


