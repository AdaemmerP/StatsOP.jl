
"""
    arl_acf_ic(lam, cl, acf_dgp, reps; ced=false, ad=100, rl_max=typemax(Int))
    arl_acf_ic(lam, cl, acf_dgp, reps, acf_version; ced=false, ad=100, rl_max=typemax(Int))

Compute the in-control average run length (ARL) of the EWMA lag-1 autocorrelation (ACF)
chart via simulation. The computation is multithreaded.

- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the ACF chart.
- `acf_dgp`: in-control DGP (e.g. `ContinuousDGPIC`).
- `reps::Int`: number of replications.
- `acf_version::Int`: version of the ACF statistic (see [`stat_acf`](@ref)); the method
  without this argument uses version 1.
- `ced::Bool=false`: use conditional expected delay initialization.
- `ad::Int=100`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_acf_ic(lam, cl, acf_dgp, reps; ced=false, ad=100, rl_max::Int=typemax(Int))

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_acf_ic(lam, cl, i, acf_dgp, acf_dgp.dist; ced=ced, ad=ad, rl_max=rl_max)
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

# -----------------------------------------------------------------------------#
# --------            Only for testing different versions ---------------------#
# -----------------------------------------------------------------------------#
function arl_acf_ic(lam, cl, acf_dgp, reps, acf_version; ced=false, ad=100, rl_max::Int=typemax(Int))

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_acf_ic(lam, cl, i, acf_dgp, acf_dgp.dist, acf_version; ced=ced, ad=ad, rl_max=rl_max)
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_acf_ic(lam, cl, p_reps, acf_dgp, acf_dgp_dist, acf_version; ced=false, ad=100,
      rl_max=typemax(Int))

Compute in-control run lengths of the EWMA lag-1 autocorrelation (ACF) chart for a chunk
of replications. This is the single-threaded worker used by [`arl_acf_ic`](@ref).

- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the ACF chart.
- `p_reps`: range of replication indices to process.
- `acf_dgp`: in-control DGP.
- `acf_dgp_dist`: innovation distribution of `acf_dgp`.
- `acf_version::Int`: version of the ACF statistic (see [`stat_acf`](@ref)).
- `ced::Bool=false`: use conditional expected delay initialization.
- `ad::Int=100`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_acf_ic(lam, cl, p_reps, acf_dgp, acf_dgp_dist, acf_version; ced=false, ad=100, rl_max::Int=typemax(Int))

  # Pre-allocate
  rls = Vector{Int64}(undef, length(p_reps))
  x_vec = Vector{Float64}(undef, 2)

  # Pre-calculate process parameters
  μ₀ = mean(acf_dgp_dist)
  σ₀² = var(acf_dgp_dist)

  # Declare mₜ in outer scope (used only by version 2)
  mₜ = μ₀

  for r in 1:length(p_reps)

    # -------------------------------------------------------------------------#
    # 1. Initialization / CED Phase
    # -------------------------------------------------------------------------#
    if ced
      icrun = true
      while icrun
        init_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)

        # Set starting values based on version
        if acf_version == 1
          cₜ = 0.0
          sₜ = σ₀²
        elseif acf_version == 2
          cₜ = μ₀^2
          sₜ = σ₀² + μ₀^2
          mₜ = μ₀
        elseif acf_version == 3
          cₜ = 0.0
          sₜ = σ₀²
        end

        falarm = false
        acf_stat = 0.0

        for _ in 1:ad
          if acf_version == 1
            cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
            sₜ = lam * (x_vec[2] - μ₀)^2 + (1.0 - lam) * sₜ
            acf_stat = cₜ / sₜ
          elseif acf_version == 2
            cₜ = lam * x_vec[2] * x_vec[1] + (1.0 - lam) * cₜ
            sₜ = lam * x_vec[2]^2 + (1.0 - lam) * sₜ
            mₜ = lam * x_vec[2] + (1.0 - lam) * mₜ
            acf_stat = (cₜ - mₜ^2) / (sₜ - mₜ^2)
          elseif acf_version == 3
            cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
            acf_stat = cₜ / σ₀²
          end
          update_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)
          if abs(acf_stat) > cl
            falarm = true
            break
          end
        end

        if falarm == false
          icrun = false
        end
      end
    else
      # Standard initialization without CED
      init_dgp_op!(acf_dgp, x_vec, acf_dgp_dist, 1)
      if acf_version == 1
        cₜ = 0.0
        sₜ = σ₀²
      elseif acf_version == 2
        cₜ = μ₀^2
        sₜ = σ₀² + μ₀^2
        mₜ = μ₀
      elseif acf_version == 3
        cₜ = 0.0
        sₜ = σ₀²
      end
      acf_stat = 0.0
    end

    # -------------------------------------------------------------------------#
    # 2. Run Length (RL) Phase
    # -------------------------------------------------------------------------#
    rl = 0
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

      # Update x_vec for the next iteration
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
