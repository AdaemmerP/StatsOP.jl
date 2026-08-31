

"""
    arl_kappa_ic(qual_dgp, lam, cl, reps; chart_choice, rl_max=typemax(Int))

Compute the in-control average run length (ARL) of the EWMA κ-chart for qualitative
processes via simulation. The computation is multithreaded.

- `qual_dgp`: in-control DGP (e.g. `DiscreteDGPIC`).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `reps::Int`: number of replications.
- `chart_choice`: one of [`KappaN1`](@ref)`()`, [`KappaN2`](@ref)`()`,
  [`KappaO1`](@ref)`()`, [`KappaO2`](@ref)`()`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_kappa_ic(
  qual_dgp, lam, cl, reps; chart_choice, rl_max::Int=typemax(Int)
)

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_kappa_ic(
      lam, cl, i, qual_dgp, qual_dgp.dist, chart_choice, rl_max
    )
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end
# ---------------------------------------------------------------------#
# Comparison of all charts
# ---------------------------------------------------------------------#
"""
    rl_kappa_ic(lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice,
      rl_max=typemax(Int))

Compute in-control run lengths of the EWMA κ-chart for a chunk of replications. This is
the single-threaded worker used by [`arl_kappa_ic`](@ref). Methods exist for the chart
choices [`KappaN1`](@ref)`()`, [`KappaN2`](@ref)`()`, [`KappaO1`](@ref)`()`, and
[`KappaO2`](@ref)`()`.

- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `p_reps`: range of replication indices to process.
- `qual_dgp`: in-control DGP.
- `qual_dgp_dist`: marginal distribution of `qual_dgp`.
- `chart_choice`: chart choice (see above).
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN1, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_dgp_dist)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  q₀ = pdf.(qual_dgp_dist, sup)
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

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end


function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN2, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_dgp_dist)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  p₀ = pdf.(qual_dgp_dist, sup)
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

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end


function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO1, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_dgp_dist)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  # Initialize with CDF for Ordinal version
  q₀ = cdf.(qual_dgp_dist, sup)
  Q₀ = sum(q₀ .^ 2)
  x_vec = zeros(2)

  for r in axes(p_reps, 1)
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)
    qₜ = copy(q₀)
    Qₜ = Q₀
    rl, stat = 0, 0.0

    while abs(stat) < cl
      rl += 1
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


function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO2, rl_max::Int=typemax(Int)
)
  rls = zeros(Int64, length(p_reps))
  sup_lb, sup_ub = get_bounds(qual_dgp_dist)
  sup = collect(sup_lb:sup_ub)

  Bₜ, Bₜ₋₁ = zeros(Int, length(sup)), zeros(Int, length(sup))
  f₀ = cdf.(qual_dgp_dist, sup)
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

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end
