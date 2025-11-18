export arl_qual_ic,
  rl_qual_ic



# Function to compute average run length for ordinal patterns
function arl_qual_ic(qual_dgp, lam, cl, reps; chart_choice, d=1)

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

#--- Run-length method for D-Chart
function rl_qual_ic(
  lam, cl, p_reps, qual_dgp, qual_dgp_dist, chart_choice::KNominal
)

  # value of patterns (can become variable in future versions)
  m = 3

  # Pre-allocate variables
  rls = zeros(Int64, length(p_reps))
  bt = zeros(Int, length(pdf(qual_dgp_dist)))
  bt1 = similar(bt)
  q0 = pdf(qual_dgp_dist)
  Q0 = sum(q0 .^ 2)
  # compute length of 'x_vec', containing the time series observations
  x_vec = zeros(2)

  for r in axes(p_reps, 1) # p_reps is a range

    # initialize run length at zero
    rl = 0

    # Initialize observations
    seq = init_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

    # Set match counts
    bt[seq[2]] += 1
    bt1[seq[1]] += 1
    @. q0 = lam * bt + (1 - lam) * q0
    dot_bt_bt1 = dot(bt, bt1)

    # Compute EWMA statistic
    Q0 = lam * dot_bt_bt1 + (1 - lam) * Q0
    stat = chart_stat_qual(q0, Q0, chart_choice)

    while stat < cl
      # increase run length
      rl += 1
      fill!(bt, 0.0)
      fill!(bt1, 0.0)

      # update sequence depending on DGP
      seq = update_dgp_op!(qual_dgp, x_vec, qual_dgp_dist, 1) # d=1 -> use dgp from ops to reduce redundancy

      # update match counts
      bt[seq[2]] += 1
      bt1[seq[1]] += 1

      @. q0 = lam * bt + (1 - lam) * q0
      dot_bt_bt1 = dot(bt, bt1)

      # Compute EWMA statistic 
      Q0 = lam * dot_bt_bt1 + (1 - lam) * Q0
      stat = chart_stat_qual(q0, Q0, chart_choice)

    end

    rls[r] = rl
  end
  return rls
end

