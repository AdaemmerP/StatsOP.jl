export arl_kappa_ic,
  rl_kappa_ic


# Function to compute average run length for ordinal patterns
function arl_kappa_ic(
  qual_dgp, lam, cl, reps; chart_choice
)

  # No threading or multiprocessing
  if nprocs() == 1 && reps <= Threads.nthreads()
    results = rl_kappa_ic(
      lam, cl, 1:reps, qual_dgp, qual_dgp.dist, chart_choice
    )

    return (mean(results), std(results) / sqrt(reps))

    # Threading
  elseif nprocs() == 1 && reps > Threads.nthreads()

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_kappa_ic(
        lam, cl, i, qual_dgp, qual_dgp.dist, chart_choice
      )

    end

    # Multiprocessing    
  elseif nprocs() > 1 && reps >= nworkers()

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_kappa_ic(lam, cl, i, qual_dgp, qual_dgp.dist, chart_choice)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


# function rl_kappa_ic(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN;
#   ced=false, ad=100
# )

#   # Pre-allocate variables
#   rls = zeros(Int64, length(p_reps))
#   p_low, p_high = 1e-12, 1 - 1e-12

#   sup_lb = isfinite(minimum(qual_dgp_dist)) ?
#            minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
#   sup_ub = isfinite(maximum(qual_dgp_dist)) ?
#            maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)

#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Global initial states (targets)
#   q₀ = pdf(qual_dgp_dist, sup)
#   Q₀ = sum(q₀ .^ 2)

#   x_vec = zeros(2)

#   for r in axes(p_reps, 1)

#     # -------------------------------------------------------------------------#
#     # 1. Initialization / CED Phase
#     # -------------------------------------------------------------------------#
#     if ced
#       icrun = true
#       while icrun
#         # Reset to target IC state for each attempt
#         qₜ = copy(q₀)
#         Qₜ = Q₀
#         seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
#         falarm = false

#         for _ in 1:ad
#           # Update match counts
#           @. Bₜ = (sup == seq[2])
#           @. Bₜ₋₁ = (sup == seq[1])
#           dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#           # EWMA update
#           @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#           Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#           stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#           # Prepare next observation
#           seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#           if abs(stat) > cl
#             falarm = true
#             break
#           end
#         end

#         if !falarm
#           icrun = false
#         end
#       end
#     else
#       # Standard initialization
#       qₜ = copy(q₀)
#       Qₜ = Q₀
#       seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
#       # Neutral start to enter loop and process t=1 correctly
#       stat = 0.0
#     end

#     # -------------------------------------------------------------------------#
#     # 2. Run Length (RL) Phase
#     # -------------------------------------------------------------------------#
#     rl = 0

#     while abs(stat) < cl
#       rl += 1

#       # Update match counts for current seq
#       @. Bₜ = (sup == seq[2])
#       @. Bₜ₋₁ = (sup == seq[1])
#       dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#       # EWMA update
#       @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#       # Update sequence for the next iteration
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
#     end

#     rls[r] = rl
#   end
#   return rls
# end







# ---------------------------------------------------------------------#
# Comparison of all charts
# ---------------------------------------------------------------------#
function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN1
)
  rls = zeros(Int64, length(p_reps))
  p_low, p_high = 1e-12, 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ? minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ? maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  q₀ = pdf(qual_dgp_dist, sup)
  Q₀ = sum(q₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    # Initialize observations and states
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    qₜ = copy(q₀)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1

      # Update match counts for current x_vec
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])

      # EWMA update for both components
      @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

      # Update sequence for next iteration
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    end
    rls[r] = rl
  end
  return rls
end


function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN2
)
  rls = zeros(Int64, length(p_reps))
  p_low, p_high = 1e-12, 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ? minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ? maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  p₀ = pdf(qual_dgp_dist, sup)
  Q₀ = sum(p₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])

      # EWMA update only for joint probability component
      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(p₀, Qₜ, chart_choice)

      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    end
    rls[r] = rl
  end
  return rls
end


function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO1
)
  rls = zeros(Int64, length(p_reps))
  p_low, p_high = 1e-12, 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ? minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ? maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  # Initialize with CDF for Ordinal version
  q₀ = cdf(qual_dgp_dist, sup)
  Q₀ = sum(q₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    qₜ = copy(q₀)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
      # Note: For ordinal, B is usually the indicator for the cumulative state
      @. Bₜ = (sup >= seq[2])
      @. Bₜ₋₁ = (sup >= seq[1])

      @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    end
    rls[r] = rl
  end
  return rls
end


function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO2
)
  rls = zeros(Int64, length(p_reps))
  p_low, p_high = 1e-12, 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ? minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ? maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  f₀ = cdf(qual_dgp_dist, sup)
  Q₀ = sum(f₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
      @. Bₜ = (sup >= seq[2])
      @. Bₜ₋₁ = (sup >= seq[1])

      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(f₀, Qₜ, chart_choice)

      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    end
    rls[r] = rl
  end
  return rls
end





# #--- Run-length method for KNominal
# function rl_kappa_ic(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN1
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_dgp_dist)) ?
#            minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
#   sup_ub = isfinite(maximum(qual_dgp_dist)) ?
#            maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   qₜ = pdf(qual_dgp_dist, sup)
#   Qₜ = sum(qₜ .^ 2)

#   # compute length of 'x_vec', containing the time series observations
#   x_vec = zeros(2)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0

#     # Initialize observations
#     # d=1 -> use dgp from ops to reduce redundancy
#     seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#     # Set match counts
#     @. Bₜ = (sup == seq[2])
#     @. Bₜ₋₁ = (sup == seq[1])
#     dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#     # EWMA statistic
#     @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#     Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#     stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#     while abs(stat) < cl

#       # increase run length
#       rl += 1

#       # reset match counts
#       fill!(Bₜ, 0)
#       fill!(Bₜ₋₁, 0)

#       # update sequence depending on DGP
#       # d=1 -> use dgp from ops to reduce redundancy
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#       # update match counts
#       @. Bₜ = (sup == seq[2])
#       @. Bₜ₋₁ = (sup == seq[1])
#       dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#       # EWMA update
#       @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#     end

#     rls[r] = rl
#   end
#   return rls
# end

# function rl_kappa_ic(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN2
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_dgp_dist)) ?
#            minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
#   sup_ub = isfinite(maximum(qual_dgp_dist)) ?
#            maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   p₀ = pdf(qual_dgp_dist, sup)
#   Qₜ = sum(p₀ .^ 2)

#   # compute length of 'x_vec', containing the time series observations
#   x_vec = zeros(2)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0

#     # Initialize observations
#     # d=1 -> use dgp from ops to reduce redundancy
#     seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#     # Set match counts
#     @. Bₜ = (sup == seq[2])
#     @. Bₜ₋₁ = (sup == seq[1])
#     dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#     # EWMA statistic
#     Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#     stat = chart_stat_qual(p₀, Qₜ, chart_choice)

#     while abs(stat) < cl
#       # increase run length
#       rl += 1
#       fill!(Bₜ, 0)
#       fill!(Bₜ₋₁, 0)

#       # update sequence depending on DGP
#       # d=1 -> use dgp from ops to reduce redundancy
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#       # update match counts
#       @. Bₜ = (sup == seq[2])
#       @. Bₜ₋₁ = (sup == seq[1])
#       dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#       # EWMA statistic
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(p₀, Qₜ, chart_choice)

#     end

#     rls[r] = rl

#   end
#   return rls
# end

# #--- Run-length method for KOrdinal
# function rl_kappa_ic(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO1
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_dgp_dist)) ?
#            minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
#   sup_ub = isfinite(maximum(qual_dgp_dist)) ?
#            maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   qₜ = cdf(qual_dgp_dist, sup)
#   Qₜ = sum(qₜ .^ 2)

#   # compute length of 'x_vec', containing the time series observations
#   x_vec = zeros(2)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0

#     # Initialize observations
#     # d=1 -> use dgp from ops to reduce redundancy
#     seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#     # Set match counts
#     @. Bₜ = (sup == seq[2])
#     @. Bₜ₋₁ = (sup == seq[1])
#     dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#     # EWMA statistic
#     @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#     Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#     stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#     while abs(stat) < cl
#       # increase run length
#       rl += 1
#       fill!(Bₜ, 0)
#       fill!(Bₜ₋₁, 0)

#       # update sequence depending on DGP
#       # d=1 -> use dgp from ops to reduce redundancy
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#       # update match counts
#       @. Bₜ = (sup == seq[2])
#       @. Bₜ₋₁ = (sup == seq[1])
#       dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#       # EWMA statistic
#       @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#     end

#     rls[r] = rl
#   end
#   return rls
# end


# #--- Run-length method for KOrdinal
# function rl_kappa_ic(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO2
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_dgp_dist)) ?
#            minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
#   sup_ub = isfinite(maximum(qual_dgp_dist)) ?
#            maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   f₀ = cdf(qual_dgp_dist, sup)
#   Qₜ = sum(f₀ .^ 2)

#   # compute length of 'x_vec', containing the time series observations
#   x_vec = zeros(2)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0

#     # Initialize observations
#     # d=1 -> use dgp from ops to reduce redundancy
#     seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#     # Set match counts
#     @. Bₜ = (sup == seq[2])
#     @. Bₜ₋₁ = (sup == seq[1])
#     dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#     # EWMA statistic
#     Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#     stat = chart_stat_qual(f₀, Qₜ, chart_choice)

#     while abs(stat) < cl
#       # increase run length
#       rl += 1
#       fill!(Bₜ, 0)
#       fill!(Bₜ₋₁, 0)

#       # update sequence depending on DGP
#       # d=1 -> use dgp from ops to reduce redundancy
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

#       # update match counts
#       @. Bₜ = (sup == seq[2])
#       @. Bₜ₋₁ = (sup == seq[1])
#       dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#       # EWMA statistic
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(f₀, Qₜ, chart_choice)

#     end

#     rls[r] = rl
#   end
#   return rls
# end
