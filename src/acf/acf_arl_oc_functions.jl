
# function arl_acf_oc(lam, cl, acf_dgp, dist_null, reps)

#   # Check whether to use threading or multi processing --> only one process threading, else distributed
#   if nprocs() == 1

#     # Make chunks for separate tasks (based on number of threads)        
#     chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

#     # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
#     par_results = map(chunks) do i
#       Threads.@spawn rl_acf_oc(lam, cl, i, acf_dgp, acf_dgp.dist, dist_null)

#     end

#   elseif nprocs() > 1

#     # Make chunks for separate tasks (based on number of workers)
#     chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

#     par_results = pmap(chunks) do i
#       rl_acf(lam, cl, i, acf_dgp, acf_dgp.dist, dist_null)
#     end

#   end

#   # Collect results from tasks
#   rls = fetch.(par_results)
#   rlvec = Iterators.flatten(rls) |> collect
#   return (mean(rlvec), std(rlvec) / sqrt(reps))
# end

# function rl_acf_ic(lam, cl, p_reps, acf_dgp, dgp_dist_ic; ced=false, ad=100)

#   # Pre-allocate 
#   rls = Vector{Int64}(undef, length(p_reps))
#   x_vec = Vector{Float64}(undef, 2)

#   # Reference null distribution parameters
#   μ₀ = mean(dgp_dist_ic)
#   σ₀² = var(dgp_dist_ic)

#   # Create pool vector for CED runs to sample from the stationary distribution
#   if ced
#     pool_vector = Vector{Float64}(undef, 10_000)
#     # Corrected variable names to match function arguments
#     init_dgp_op!(acf_dgp, pool_vector, dgp_dist_ic, 1)
#   else
#     pool_vector = Float64[]
#   end

#   for r in 1:length(p_reps)

#     # -------------------------------------------------------------------------#
#     # 1. Initialization / CED Phase (Version 1 Logic)
#     # -------------------------------------------------------------------------#
#     if ced
#       icrun = true
#       while icrun
#         # Initialize x_vec using the stationary pool
#         init_dgp_op_ced!(acf_dgp, x_vec, pool_vector, 1)
#         cₜ = 0.0
#         sₜ = σ₀²
#         falarm = false

#         for _ in 1:ad
#           # Compute statistic
#           cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
#           sₜ = lam * (x_vec[2] - μ₀)^2 + (1.0 - lam) * sₜ
#           acf_stat = cₜ / sₜ

#           # Prepare next observation from stationary pool
#           update_dgp_op_ced!(acf_dgp, x_vec, pool_vector, 1)

#           if abs(acf_stat) > cl
#             falarm = true
#             break # Exit directly when a false alarm occurs
#           end
#         end

#         # Step out of while loop if no false alarm occurred
#         if !falarm
#           icrun = false
#         end
#       end
#     else
#       # Standard initialization: no delay phase
#       init_dgp_op!(acf_dgp, x_vec, dgp_dist_ic, 1)
#       cₜ = 0.0
#       sₜ = σ₀²
#       acf_stat = 0.0
#     end

#     # -------------------------------------------------------------------------#
#     # 2. Run Length (RL) Phase
#     # -------------------------------------------------------------------------#
#     rl = 0

#     # Start monitoring. If ced=true, acf_stat is the state at t=ad.
#     while abs(acf_stat) < cl
#       rl += 1

#       # Update statistics with actual process data
#       cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
#       sₜ = lam * (x_vec[2] - μ₀)^2 + (1.0 - lam) * sₜ
#       acf_stat = cₜ / sₜ

#       # Generate next data point from the actual process DGP
#       update_dgp_op!(acf_dgp, x_vec, dgp_dist_ic, 1)
#     end

#     rls[r] = rl
#   end
#   return rls
# end







"""
    arl_acf_oc(lam, cl, acf_dgp, dist_null, reps, acf_version; rl_max=typemax(Int))

Compute the out-of-control average run length (ARL) of the EWMA lag-1 autocorrelation
(ACF) chart via simulation. The computation is multithreaded.

- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the ACF chart.
- `acf_dgp`: out-of-control DGP (e.g. `AR1`, `TEAR1`).
- `dist_null`: in-control (null) distribution used to center and scale the statistic.
- `reps::Int`: number of replications.
- `acf_version::Int`: version of the ACF statistic (see [`stat_acf`](@ref)).
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_acf_oc(lam, cl, acf_dgp, dist_null, reps, acf_version; rl_max::Int=typemax(Int))

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_acf_oc(lam, cl, i, acf_dgp, acf_dgp.dist, dist_null, acf_version, rl_max)
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_acf_oc(lam, cl, p_reps, acf_dgp, acf_dgp_dist, dist_null, acf_version,
      rl_max=typemax(Int))

Compute out-of-control run lengths of the EWMA lag-1 autocorrelation (ACF) chart for a
chunk of replications. This is the single-threaded worker used by [`arl_acf_oc`](@ref).

- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the ACF chart.
- `p_reps`: range of replication indices to process.
- `acf_dgp`: out-of-control DGP (e.g. `AR1`, `TEAR1`).
- `acf_dgp_dist`: innovation distribution of `acf_dgp`.
- `dist_null`: in-control (null) distribution used to center and scale the statistic.
- `acf_version::Int`: version of the ACF statistic (see [`stat_acf`](@ref)).
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_acf_oc(lam, cl, p_reps, acf_dgp, acf_dgp_dist, dist_null, acf_version, rl_max::Int=typemax(Int))

  # Pre-allocate 
  rls = Vector{Int64}(undef, length(p_reps))
  x_vec = Vector{Float64}(undef, 2)

  # Reference null distribution parameters (Target/In-Control)
  μ₀ = mean(dist_null)
  σ₀² = var(dist_null)

  for r in 1:length(p_reps)

    # 1. Initialize data vector with the first OC observations
    init_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)

    # 2. Set starting values based on the In-Control (dist_null) expectation
    if acf_version == 1
      cₜ = 0.0
      sₜ = σ₀²
    elseif acf_version == 2
      # Raw moment expectations for the IC process
      cₜ = μ₀^2
      sₜ = σ₀² + μ₀^2
      mₜ = μ₀
    elseif acf_version == 3
      cₜ = 0.0
      sₜ = σ₀²
    end

    # Set neutral start to ensure the while loop captures the first observation
    rl = 0
    acf_stat = 0.0

    # 3. Out-of-Control Run Length Phase
    while abs(acf_stat) < cl
      rl += 1

      if acf_version == 1
        # Equation (3)
        cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
        sₜ = lam * (x_vec[2] - μ₀)^2 + (1.0 - lam) * sₜ
        acf_stat = cₜ / sₜ

      elseif acf_version == 2
        # Equation (4)
        cₜ = lam * x_vec[2] * x_vec[1] + (1.0 - lam) * cₜ
        sₜ = lam * x_vec[2]^2 + (1.0 - lam) * sₜ
        mₜ = lam * x_vec[2] + (1.0 - lam) * mₜ
        acf_stat = (cₜ - mₜ^2) / (sₜ - mₜ^2)

      elseif acf_version == 3
        # Equation (5)
        cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
        acf_stat = cₜ / σ₀²
      end

      # Update observations from the OC distribution
      update_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end

    rls[r] = rl
  end
  return rls
end
