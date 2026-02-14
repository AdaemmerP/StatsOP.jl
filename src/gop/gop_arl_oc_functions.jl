export arl_gop_oc,
  rl_gop_oc



# Function to compute average run length for ordinal patterns
function arl_gop_oc(
  gop_dgp, null_dist, lam, cl, reps; chart_choice, d=1, ced=false, ad=100
)

  # Compute lookup array and number of ops
  lookup_array_gop = compute_lookup_array_gop()

  # No threading or multiprocessing
  if nprocs() == 1 && reps <= Threads.nthreads()
    results = rl_gop_oc(
      lam, cl, lookup_array_gop, 1:reps, gop_dgp, gop_dgp.dist, null_dist, chart_choice, d, ced, ad
    )

    return (mean(results), std(results) / sqrt(reps))

    # Threading
  elseif nprocs() == 1 && reps > Threads.nthreads()

    # Make chunks for separate tasks (based on number of threads)        
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_gop_oc(
        lam, cl, lookup_array_gop, i, gop_dgp, gop_dgp.dist, null_dist, chart_choice, d, ced, ad
      )

    end

    # Multiprocessing    
  elseif nprocs() > 1 && reps >= nworkers()

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_gop_oc(
        lam, cl, lookup_array_gop, i, gop_dgp, gop_dgp.dist, null_dist, chart_choice, d, ced, ad
      )
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps), median(rlvec))
end

#--- Run-length method for D-Chart
function rl_gop_oc(
  lam, cl, lookup_array_gop, p_reps, gop_dgp, gop_dgp_dist, null_dist, chart_choice::Union{D_Chart,Persistence}, d::Int, ced::Bool, ad::Int
)

  # value of patterns (can become variable in future versions)
  m = 3

  # Pre-allocate variables
  rls = zeros(Int, length(p_reps))
  bin = zeros(Int, 13)
  win = zeros(Int, m)
  ix = similar(win)
  pₜ = zeros(13)
  p₀ = zeros(13)
  pₜ_p₀ = zeros(13) # for "pₜ - p₀"
  fill_p0!(p₀, null_dist)

  # compute length of 'x_seq' vector based on d
  x_seq = zeros(1 + (m - 1) * d)

  # Create pool vector for CED runs (if "ced=true")
  # If true, create and fill vector with initial values
  pool_vector = if ced
    Vector{Float64}(undef, 10_000)
    init_dgp_op!(dgp, pool_vector, dist, 1)
  else
    Float64[]
  end

  for r in axes(p_reps, 1) # p_reps is a range

    #----------------------------------------------------------------------#
    #---------   Use Conditional Expected Delay (CED)?  -------------------#
    #----------------------------------------------------------------------#
    if ced

      icrun = true

      while icrun

        # initialze EWMA statistic, Equation (17), in the paper

        # Initialize observations
        pₜ .= p₀
        seq = init_dgp_op_ced!(gop_dgp, x_seq, pool_vector, d)

        falarm = false
        # Loop for "in-control" run
        for _ in 1:ad

          bin .= 0

          # compute ordinal pattern based on permutations    
          competerank!(win, seq, ix)

          # Binarization of ordinal pattern
          j, k, l = win
          bin[lookup_array_gop[j, k, l]] = 1

          # Compute EWMA statistic, Equation (17), in the paper
          @. pₜ = lam * bin + (1 - lam) * pₜ
          @. pₜ_p₀ = pₜ - p₀
          stat = chart_stat_gop(pₜ_p₀, chart_choice)

          # update sequence depending on DGP
          seq = update_dgp_op_ced!(gop_dgp, x_seq, pool_vector, d)
          fill!(win, 0)

          if stat > cl
            falarm = true
          end

        end # for ad run

        if falarm == false
          icrun = false
        end
      end

    end
    #----------------------------------------------------------------------#

    if ced
      seq = update_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
    else
      # initialze EWMA statistic, Equation (17), in the paper
      pₜ .= p₀
      # Initialize observations
      seq = init_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)

      # initial statistic
      @. pₜ_p₀ = pₜ - p₀
      stat = chart_stat_gop(pₜ_p₀, chart_choice)
    end

    # initialize run length at zero
    rl = 0

    while !abort_criterium_gop(stat, cl, chart_choice)
      # increase run length
      rl += 1
      bin .= 0

      # compute ordinal pattern based on permutations    
      competerank!(win, seq, ix)

      # Binarization of ordinal pattern
      j, k, l = win
      bin[lookup_array_gop[j, k, l]] = 1

      # Compute EWMA statistic, Equation (17), in the paper
      @. pₜ = lam * bin + (1 - lam) * pₜ
      # statistic based on smoothed pₜ-estimate
      @. pₜ_p₀ = pₜ - p₀
      stat = chart_stat_gop(pₜ_p₀, chart_choice)

      # update sequence depending on DGP
      seq = update_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
      fill!(win, 0)
    end

    rls[r] = rl
  end
  return rls
end


# #--- Run-length method for G-Chart
# function rl_gop_oc(
#   lam, cl, lookup_array_gop, p_reps, gop_dgp, gop_dgp_dist, null_dist, chart_choice::G_Chart, d::Int
# )

#   # value of patterns (can become variable in future versions)
#   m = 3

#   # Pre-allocate variables
#   rls = zeros(Int64, length(p_reps))
#   bin = zeros(Int, 13)
#   win = zeros(Int, m)
#   ix = similar(win)
#   pₜ = zeros(13)
#   p₀ = similar(pₜ)
#   pₜ_p₀ = similar(pₜ) # for "pₜ - p₀"
#   G = [
#     1 0 0 0 0 0 0 1 0 1 0 0 0;
#     0 0 0 0 0 1 0 0 0 0 1 0 1;
#     0 1 1 1 1 0 1 0 1 0 0 1 0
#   ]
#   G1G = G' * G
#   fill_p0!(p₀, null_dist)

#   # compute length of 'x_seq' vector based on d
#   x_seq = zeros(1 + (m - 1) * d)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # initialize run length at zero
#     rl = 0
#     # initialze EWMA statistic, Equation (17), in the paper
#     pₜ .= p₀
#     # Initialize observations
#     seq = init_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
#     # initial statistic
#     @. pₜ_p₀ = pₜ - p₀
#     stat = chart_stat_gop(pₜ_p₀, G1G, chart_choice)

#     while !abort_criterium_gop(stat, cl, chart_choice)
#       # increase run length
#       rl += 1
#       bin .= 0

#       # compute ordinal pattern based on permutations    
#       competerank!(win, seq, ix)

#       @assert isfinite(stat)

#       # Binarization of ordinal pattern
#       j, k, l = win
#       bin[lookup_array_gop[j, k, l]] = 1

#       # Compute EWMA statistic, Equation (17), in the paper
#       @. pₜ = lam * bin + (1 - lam) * pₜ
#       # statistic based on smoothed pₜ-estimate
#       @. pₜ_p₀ = pₜ - p₀
#       stat = chart_stat_gop(pₜ_p₀, G1G, chart_choice)

#       # update sequence depending on DGP
#       seq = update_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
#       fill!(win, 0)

#     end

#     rls[r] = rl
#   end
#   return rls
# end

