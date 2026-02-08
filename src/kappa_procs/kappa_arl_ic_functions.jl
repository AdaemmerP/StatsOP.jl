export arl_kappa_ic,
  rl_kappa_ic

# Function to compute average run length for ordinal patterns
function arl_kappa_ic(
  qual_dgp, lam, cl, reps; chart_choice, d=1
)

  # No threading or multiprocessing
  if nprocs() == 1 && reps <= Threads.nthreads()
    results = rl_gop_ic(
      lam, cl, 1:reps, qual_dgp, qual_dgp.dist, chart_choice, d
    )

    return (mean(results), std(results) / sqrt(reps))

    # Threading
  elseif nprocs() == 1 && reps > Threads.nthreads()

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_gop_ic(lam, cl, i, qual_dgp, qual_dgp.dist, chart_choice, d)

    end

    # Multiprocessing    
  elseif nprocs() > 1 && reps >= nworkers()

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_gop_ic(lam, cl, i, qual_dgp, qual_dgp.dist, chart_choice, d)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

#--- Run-length method for KNominal
function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN1
)

  # Pre-allocate variables
  rls = zeros(Int64, length(p_reps))
  Bₜ = zeros(Int, length(pdf(qual_dgp_dist)))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  qₜ = pdf(qual_dgp_dist)
  Qₜ = sum(qₜ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    Bₜ[seq[2]] += 1
    Bₜ₋₁[seq[1]] += 1
    # Update
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    while abs(stat) < cl
      # increase run length
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

      # update match counts
      Bₜ[seq[2]] += 1
      Bₜ₋₁[seq[1]] += 1

      @. qₜ = lam * Bₜ + (1 - lam) * qₜ
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # Compute EWMA statistic
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end

function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaN2
)

  # Pre-allocate variables
  rls = zeros(Int64, length(p_reps))
  Bₜ = zeros(Int, length(pdf(qual_dgp_dist)))
  Bₜ₋₁ = similar(Bt)

  # Initialize at t = 0
  p₀ = pdf(qual_dgp_dist)
  Qₜ = sum(p₀ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    Bₜ[seq[2]] += 1
    Bₜ₋₁[seq[1]] += 1
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(p₀, Qₜ, chart_choice)

    while abs(stat) < cl
      # increase run length
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

      # update match counts
      Bₜ[seq[2]] += 1
      Bₜ₋₁[seq[1]] += 1
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # Compute EWMA statistic
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    end

    rls[r] = rl

  end
  return rls
end

#--- Run-length method for KNominal
function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO1
)

  # Pre-allocate variables
  rls = zeros(Int64, length(p_reps))
  Bₜ = zeros(Int, length(pdf(qual_dgp_dist)))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  qₜ = cdf(qual_dgp, support(qual_dgp_dist))
  Qₜ = sum(qₜ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    Bₜ[seq[2]] += 1
    Bₜ₋₁[seq[1]] += 1
    # Update
    @. qₜ = lam * Bₜ + (1 - lam) * qₜ
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    while abs(stat) < cl
      # increase run length
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

      # update match counts
      Bₜ[seq[2]] += 1
      Bₜ₋₁[seq[1]] += 1

      @. qₜ = lam * Bₜ + (1 - lam) * qₜ
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # Compute EWMA statistic
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(qₜ, Qₜ, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end


#--- Run-length method for KNominal
function rl_kappa_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KappaO2
)

  # Pre-allocate variables
  rls = zeros(Int64, length(p_reps))
  Bₜ = zeros(Int, length(pdf(qual_dgp_dist)))
  Bₜ₋₁ = similar(Bₜ)

  # Initialize at t = 0
  f₀ = cdf(qual_dgp, support(qual_dgp_dist))
  Qₜ = sum(f₀ .^ 2)

  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    Bₜ[seq[2]] += 1
    Bₜ₋₁[seq[1]] += 1
    dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

    # Compute EWMA statistic
    Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
    stat = chart_stat_qual(f₀, Qₜ, chart_choice)

    while abs(stat) < cl
      # increase run length
      rl += 1
      fill!(Bₜ, 0)
      fill!(Bₜ₋₁, 0)

      # update sequence depending on DGP
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

      # update match counts
      Bₜ[seq[2]] += 1
      Bₜ₋₁[seq[1]] += 1
      dot_Bₜ_Bₜ₋₁ = dot(Bₜ, Bₜ₋₁)

      # Compute EWMA statistic
      Qₜ = lam * dot_Bₜ_Bₜ₋₁ + (1 - lam) * Qₜ
      stat = chart_stat_qual(f₀, Qₜ, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end
