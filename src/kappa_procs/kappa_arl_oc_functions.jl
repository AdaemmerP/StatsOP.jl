
"""
    arl_kappa_oc(qual_dgp, qual_null_dist, lam, cl, reps; chart_choice,
      rl_max=typemax(Int))

Compute the out-of-control average run length (ARL) of the EWMA κ-chart for qualitative
processes via simulation. The computation is multithreaded.

- `qual_dgp`: out-of-control DGP.
- `qual_null_dist`: in-control (null) distribution used to initialize the recursion.
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `reps::Int`: number of replications.
- `chart_choice`: one of [`KappaN1`](@ref)`()`, [`KappaN2`](@ref)`()`,
  [`KappaO1`](@ref)`()`, [`KappaO2`](@ref)`()`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_kappa_oc(
  qual_dgp, qual_null_dist, lam, cl, reps; chart_choice, rl_max::Int=typemax(Int)
)

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_kappa_oc(lam, cl, i, qual_dgp, qual_dgp.dist, qual_null_dist, chart_choice, rl_max)
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


# function rl_kappa_oc(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist,
#   chart_choice::KappaN; ced=false, ad=100
# )
#   # Pre-allocate variables
#   rls = zeros(Int64, length(p_reps))
#   p_low, p_high = 1e-12, 1 - 1e-12

#   # Define support based on null distribution
#   sup_lb = isfinite(minimum(qual_null_dist)) ? minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
#   sup_ub = isfinite(maximum(qual_null_dist)) ? maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
#   sup = collect(sup_lb:sup_ub)

#   Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))

#   # Global target states from null distribution
#   q₀ = pdf(qual_null_dist, sup)
#   Q₀ = sum(q₀ .^ 2)
#   x_vec = zeros(2)

#   # Pre-sample stationary pool if CED is used
#   if ced
#     pool_vector = Vector{Float64}(undef, 10_000)
#     # Assuming stationary IC state matches null_dist
#     init_dgp_op!(qual_dgp, pool_vector, qual_null_dist, 1)
#   else
#     pool_vector = Float64[]
#   end

#   for r in axes(p_reps, 1)

#     # ----------------------------------------------------------------------#
#     # 1. Initialization / CED Phase (In-Control)
#     # ----------------------------------------------------------------------#
#     if ced
#       icrun = true
#       while icrun
#         # Reset to fresh IC target states
#         qₜ = copy(q₀)
#         Qₜ = Q₀
#         # Sample initial sequence from stationary pool
#         seq = init_dgp_op_ced!(qual_dgp, x_vec, pool_vector, 1)
#         falarm = false

#         for _ in 1:ad
#           # Update match counts
#           @. Bₜ = (sup == seq[2])
#           @. Bₜ₋₁ = (sup == seq[1])

#           # EWMA update
#           @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
#           Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
#           stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#           # Prepare next IC observation
#           seq = update_dgp_op_ced!(qual_dgp, x_vec, pool_vector, 1)

#           if abs(stat) > cl
#             falarm = true
#             break
#           end
#         end

#         if !falarm
#           icrun = false
#         end
#       end
#       # seq is now ready for the first OOC step (time ad + 1)
#     else
#       # Standard initialization: Start directly with OOC distribution
#       qₜ = copy(q₀)
#       Qₜ = Q₀
#       seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
#       stat = 0.0 # Neutral start to enter while-loop
#     end

#     # ----------------------------------------------------------------------#
#     # 2. Run Length Phase (Out-of-Control)
#     # ----------------------------------------------------------------------#
#     rl = 0

#     # Loop continues until chart triggers alarm
#     while abs(stat) < cl
#       rl += 1

#       # Compute statistic for current observations
#       @. Bₜ = (sup == seq[2])
#       @. Bₜ₋₁ = (sup == seq[1])

#       @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
#       Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
#       stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#       # Update observations from the OOC distribution
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
#     end

#     rls[r] = rl
#   end
#   return rls
# end


# function rl_kappa_oc(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist,
#   chart_choice::KappaO; ced=false, ad=100
# )
#   # Pre-allocate variables
#   rls = zeros(Int64, length(p_reps))
#   p_low, p_high = 1e-12, 1 - 1e-12

#   # Define support based on null distribution
#   sup_lb = isfinite(minimum(qual_null_dist)) ? minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
#   sup_ub = isfinite(maximum(qual_null_dist)) ? maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
#   sup = collect(sup_lb:sup_ub)

#   Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))

#   # Target IC CDF states from null distribution
#   q₀ = cdf(qual_null_dist, sup)
#   Q₀ = sum(q₀ .^ 2)
#   x_vec = zeros(2)

#   # Pre-sample stationary pool if CED is used
#   if ced
#     pool_vector = Vector{Float64}(undef, 10_000)
#     init_dgp_op!(qual_dgp, pool_vector, qual_null_dist, 1)
#   else
#     pool_vector = Float64[]
#   end

#   for r in axes(p_reps, 1)

#     # -------------------------------------------------------------------------#
#     # 1. Initialization / CED Phase (In-Control stationary phase)
#     # -------------------------------------------------------------------------#
#     if ced
#       icrun = true
#       while icrun
#         # Reset to target IC CDF state
#         qₜ = copy(q₀)
#         Qₜ = Q₀
#         # Initialize x_vec from the stationary pool
#         seq = init_dgp_op_ced!(qual_dgp, x_vec, pool_vector, 1)
#         falarm = false

#         for _ in 1:ad
#           # Update cumulative match counts
#           @. Bₜ = (sup >= seq[2])
#           @. Bₜ₋₁ = (sup >= seq[1])

#           # EWMA update (Ordinal version using CDFs)
#           @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
#           Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
#           stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#           # Update from stationary pool
#           seq = update_dgp_op_ced!(qual_dgp, x_vec, pool_vector, 1)

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
#       # Standard initialization: Start directly with OOC distribution
#       qₜ = copy(q₀)
#       Qₜ = Q₀
#       seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
#       stat = 0.0 # Neutral start to enter loop
#     end

#     # -------------------------------------------------------------------------#
#     # 2. Run Length Phase (Out-of-Control monitoring)
#     # -------------------------------------------------------------------------#
#     rl = 0

#     while abs(stat) < cl
#       rl += 1

#       # Update cumulative match counts for current OOC observations
#       @. Bₜ = (sup >= seq[2])
#       @. Bₜ₋₁ = (sup >= seq[1])

#       # EWMA update using OOC data against IC targets
#       @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
#       Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
#       stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#       # Update sequence from the OOC DGP
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
#     end

#     rls[r] = rl
#   end
#   return rls
# end


# ---------------------------------------------------------------------#
# Comparison of all charts
# ---------------------------------------------------------------------#
"""
    rl_kappa_oc(lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice,
      rl_max=typemax(Int))

Compute out-of-control run lengths of the EWMA κ-chart for a chunk of replications. This
is the single-threaded worker used by [`arl_kappa_oc`](@ref). Methods exist for the chart
choices [`KappaN1`](@ref)`()`, [`KappaN2`](@ref)`()`, [`KappaO1`](@ref)`()`, and
[`KappaO2`](@ref)`()`.

- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `p_reps`: range of replication indices to process.
- `qual_dgp`: out-of-control DGP.
- `qual_dgp_dist`: marginal distribution of `qual_dgp`.
- `qual_null_dist`: in-control (null) distribution used to initialize the recursion.
- `chart_choice`: chart choice (see above).
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaN1, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_null_dist)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  # Target IC states from null distribution
  q₀ = pdf(qual_null_dist, sup)
  Q₀ = sum(q₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    # Start OOC observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    qₜ = copy(q₀)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])

      # EWMA update using OOC data against IC targets
      @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end

function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaN2, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_null_dist)
  sup = collect(sup_lb:sup_ub)
  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))

  # Fixed null distribution target
  p₀ = pdf(qual_null_dist, sup)
  Q₀ = sum(p₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])

      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(p₀, Qₜ, chart_choice)

      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end


function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaO1, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_null_dist)
  sup = collect(sup_lb:sup_ub)
  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))

  # Target IC CDF
  q₀ = cdf(qual_null_dist, sup)
  Q₀ = sum(q₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    qₜ = copy(q₀)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)
      @. Bₜ = (sup >= seq[2])
      @. Bₜ₋₁ = (sup >= seq[1])

      @. qₜ = lam * Bₜ + (1.0 - lam) * qₜ
      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end


function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaO2, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_null_dist)
  sup = collect(sup_lb:sup_ub)
  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))

  # Fixed IC target CDF
  f₀ = cdf(qual_null_dist, sup)
  Q₀ = sum(f₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)
      @. Bₜ = (sup >= seq[2])
      @. Bₜ₋₁ = (sup >= seq[1])

      Qₜ = lam * dot(Bₜ, Bₜ₋₁) + (1.0 - lam) * Qₜ
      stat = chart_stat_qual(f₀, Qₜ, chart_choice)

      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end


# #--- Run-length method for KNominal
# function rl_kappa_oc(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaN1
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_null_dist)) ?
#            minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
#   sup_ub = isfinite(maximum(qual_null_dist)) ?
#            maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   # Compute support of null distribution
#   qₜ = pdf(qual_null_dist, sup)
#   Qₜ = sum(qₜ .^ 2)

#   # compute length of 'x_vec', containing the time series observations
#   x_vec = zeros(2)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0

#     # Initialize observations
#     seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

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
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

#       # update match counts
#       @. Bₜ = (sup == seq[2])
#       @. Bₜ₋₁ = (sup == seq[1])
#       dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#       # Compute EWMA statistic
#       @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#     end

#     rls[r] = rl
#   end
#   return rls
# end


# #--- Run-length method for KNominal
# function rl_kappa_oc(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaN2
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_null_dist)) ?
#            minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
#   sup_ub = isfinite(maximum(qual_null_dist)) ?
#            maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   # Compute support of null distribution
#   p₀ = pdf(qual_null_dist, sup)
#   Qₜ = sum(p₀ .^ 2)

#   # compute length of 'x_vec', containing the time series observations
#   x_vec = zeros(2)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0

#     # Initialize observations
#     seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

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

#       # reset match counts
#       fill!(Bₜ, 0)
#       fill!(Bₜ₋₁, 0)

#       # update sequence depending on DGP
#       seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

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


# function rl_kappa_oc(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaO1
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_null_dist)) ?
#            minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
#   sup_ub = isfinite(maximum(qual_null_dist)) ?
#            maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   # Compute support of null distribution
#   qₜ = cdf(qual_null_dist, sup)
#   Qₜ = sum(qₜ .^ 2)

#   # compute length of 'x_vec', containing the time series observations
#   x_vec = zeros(2)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0

#     # Initialize observations
#     seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

#     # Set match counts
#     @. Bₜ = (sup == seq[2])
#     @. Bₜ₋₁ = (sup == seq[1])
#     dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

#     # Compute EWMA statistic
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

#       # EWMA statistic
#       @. qₜ = lam * Bₜ + (1 - lam) * qₜ
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

#     end

#     rls[r] = rl
#   end
#   return rls
# end


# function rl_kappa_oc(
#   lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaO2
# )

#   # Pre-allocate variables
#   # Compute support
#   rls = zeros(Int64, length(p_reps))
#   p_low = 1e-12
#   p_high = 1 - 1e-12
#   sup_lb = isfinite(minimum(qual_null_dist)) ?
#            minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
#   sup_ub = isfinite(maximum(qual_null_dist)) ?
#            maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
#   sup = collect(sup_lb:sup_ub)
#   Bₜ = zeros(Int, length(sup))
#   Bₜ₋₁ = similar(Bₜ)

#   # Initialize at t = 0
#   # Compute support of null distribution
#   f₀ = cdf(qual_null_dist, sup)
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

#       # Compute EWMA statistic
#       Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
#       stat = chart_stat_qual(f₀, Qₜ, chart_choice)

#     end

#     rls[r] = rl
#   end
#   return rls
# end