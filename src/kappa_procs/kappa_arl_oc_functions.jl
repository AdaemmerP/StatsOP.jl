export arl_kappa_oc,
  rl_kappa_oc

# Function to compute average run length for ordinal patterns
function arl_kappa_oc(
  qual_dgp, qual_null_dist, lam, cl, reps; chart_choice
)

  # No threading or multiprocessing
  if nprocs() == 1 && reps <= Threads.nthreads()
    results = rl_kappa_oc(
      lam, cl, 1:reps, qual_dgp, qual_dgp.dist, chart_choice
    )

    return (mean(results), std(results) / sqrt(reps))

    # Threading
  elseif nprocs() == 1 && reps > Threads.nthreads()

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_kappa_oc(lam, cl, i, qual_dgp, qual_dgp.dist, qual_null_dist, chart_choice)

    end

    # Multiprocessing    
  elseif nprocs() > 1 && reps >= nworkers()

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_kappa_oc(lam, cl, i, qual_dgp, qual_dgp.dist, qual_null_dist, chart_choice)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


#--- Run-length method for KNominal
function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaN1
)

  # Pre-allocate variables
  # Compute support
  rls = zeros(Int64, length(p_reps))
  p_low = 1e-12
  p_high = 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ?
           minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ?
           maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  # Compute support of null distribution
  sup_null_lb = isfinite(minimum(qual_null_dist)) ?
                minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
  sup_null_ub = isfinite(maximum(qual_null_dist)) ?
                maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
  qₜ = pdf(qual_null_dist, sup_null_lb:sup_null_ub)
  Qₜ = sum(qₜ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    @. Bₜ = (sup == seq[2])
    @. Bₜ₋₁ = (sup == seq[1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # EWMA statistic
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    while abs(stat) < cl

      # increase run length
      rl += 1

      # reset match counts
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

      # update match counts
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # Compute EWMA statistic
      @. qₜ = lam * Bₜ + (1 - lam) * qₜ
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end


#--- Run-length method for KNominal
function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaN2
)

  # Pre-allocate variables
  # Compute support
  rls = zeros(Int64, length(p_reps))
  p_low = 1e-12
  p_high = 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ?
           minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ?
           maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  # Compute support of null distribution
  sup_null_lb = isfinite(minimum(qual_null_dist)) ?
                minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
  sup_null_ub = isfinite(maximum(qual_null_dist)) ?
                maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
  p₀ = pdf(qual_null_dist, sup_null_lb:sup_null_ub)
  Qₜ = sum(p₀ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    @. Bₜ = (sup == seq[2])
    @. Bₜ₋₁ = (sup == seq[1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(p₀, Qₜ, chart_choice)

    while abs(stat) < cl

      # increase run length
      rl += 1

      # reset match counts
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

      # update match counts
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # EWMA statistic
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(p₀, Qₜ, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end


function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaO1
)

  # Pre-allocate variables
  # Compute support
  rls = zeros(Int64, length(p_reps))
  p_low = 1e-12
  p_high = 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ?
           minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ?
           maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  # Compute support of null distribution
  sup_null_lb = isfinite(minimum(qual_null_dist)) ?
                minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
  sup_null_ub = isfinite(maximum(qual_null_dist)) ?
                maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
  qₜ = cdf(qual_null_dist, sup_null_lb:sup_null_ub)
  Qₜ = sum(qₜ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    @. Bₜ = (sup == seq[2])
    @. Bₜ₋₁ = (sup == seq[1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    while abs(stat) < cl

      # increase run length
      rl += 1

      # reset match counts
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      # d=1 -> use dgp from ops to reduce redundancy
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

      # update match counts
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # EWMA statistic
      @. qₜ = lam * Bₜ + (1 - lam) * qₜ
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end


function rl_kappa_oc(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, qual_null_dist, chart_choice::KappaO2
)

  # Pre-allocate variables
  # Compute support
  rls = zeros(Int64, length(p_reps))
  p_low = 1e-12
  p_high = 1 - 1e-12
  sup_lb = isfinite(minimum(qual_dgp_dist)) ?
           minimum(qual_dgp_dist) : quantile(qual_dgp_dist, p_low)
  sup_ub = isfinite(maximum(qual_dgp_dist)) ?
           maximum(qual_dgp_dist) : quantile(qual_dgp_dist, p_high)
  sup = collect(sup_lb:sup_ub)
  Bₜ = zeros(Int, length(sup))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  # Compute support of null distribution
  sup_null_lb = isfinite(minimum(qual_null_dist)) ?
                minimum(qual_null_dist) : quantile(qual_null_dist, p_low)
  sup_null_ub = isfinite(maximum(qual_null_dist)) ?
                maximum(qual_null_dist) : quantile(qual_null_dist, p_high)
  f₀ = cdf(qual_null_dist, sup_null_lb:sup_null_ub)
  Qₜ = sum(f₀ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    # d=1 -> use dgp from ops to reduce redundancy
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

    # Set match counts
    @. Bₜ = (sup == seq[2])
    @. Bₜ₋₁ = (sup == seq[1])
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(f₀, Qₜ, chart_choice)

    while abs(stat) < cl

      # increase run length
      rl += 1

      # reset match counts
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      # d=1 -> use dgp from ops to reduce redundancy
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1)

      # update match counts
      @. Bₜ = (sup == seq[2])
      @. Bₜ₋₁ = (sup == seq[1])
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # Compute EWMA statistic
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(f₀, Qₜ, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end