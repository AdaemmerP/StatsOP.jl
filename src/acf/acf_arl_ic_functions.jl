
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

function rl_acf_ic(lam, cl, p_reps, acf_dgp, dgp_dist_ic; ced=false, ad=100, rl_max::Int=typemax(Int))

  # Pre-allocate 
  rls = Vector{Int64}(undef, length(p_reps))
  x_vec = Vector{Float64}(undef, 2)

  # Pre-calculate process parameters
  μ₀ = mean(dgp_dist_ic)
  σ₀² = var(dgp_dist_ic)

  for r in 1:length(p_reps)

    # -------------------------------------------------------------------------#
    # 1. Initialization / CED Phase
    # -------------------------------------------------------------------------#
    if ced
      icrun = true
      while icrun
        # Initialize DGP and statistics for each IC attempt
        init_dgp_op!(acf_dgp, x_vec, dgp_dist_ic, 1)
        cₜ = 0.0
        sₜ = σ₀²
        falarm = false

        for _ in 1:ad
          # Compute statistic 
          cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
          sₜ = lam * (x_vec[2] - μ₀)^2 + (1.0 - lam) * sₜ
          acf_stat = cₜ / sₜ

          # Prepare next observation
          update_dgp_op!(acf_dgp, x_vec, dgp_dist_ic, 1)

          # Check for false alarm
          if abs(acf_stat) > cl
            falarm = true
            break # Exit for loop early if false alarm occurs
          end
        end

        # If no false alarm occurred during 'ad' steps, accept the state
        if falarm == false
          icrun = false
        end
      end
    else
      # Standard initialization without CED
      init_dgp_op!(acf_dgp, x_vec, dgp_dist_ic, 1)
      cₜ = 0.0
      sₜ = σ₀²
      # Set to 0.0 so the RL while-loop condition is met initially
      acf_stat = 0.0
    end

    # -------------------------------------------------------------------------#
    # 2. Run Length (RL) Phase
    # -------------------------------------------------------------------------#
    rl = 0
    # Note: If ced=true, acf_stat starts with the value from the last ad step.
    # Since falarm was false, abs(acf_stat) < cl is guaranteed here.
    while abs(acf_stat) < cl
      rl += 1

      # Update statistics with current x_vec
      # (t=1 if ced=false; t=ad+1 if ced=true)
      cₜ = lam * (x_vec[2] - μ₀) * (x_vec[1] - μ₀) + (1.0 - lam) * cₜ
      sₜ = lam * (x_vec[2] - μ₀)^2 + (1.0 - lam) * sₜ
      acf_stat = cₜ / sₜ

      # Update x_vec for the next iteration
      update_dgp_op!(acf_dgp, x_vec, dgp_dist_ic, 1)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end

    rls[r] = rl
  end
  return rls
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




