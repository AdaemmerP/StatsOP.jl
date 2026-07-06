
"""
    arl_op_oc(op_dgp, lam, cl, reps=10_000; chart_choice, d=1, m=3, ced=false, ad=100,
      rl_max=typemax(Int))

Compute the out-of-control average run length (ARL) of the EWMA chart based on ordinal
patterns via simulation, following Weiß and Testik (2023). The computation is
multithreaded.

- `op_dgp`: out-of-control DGP (e.g. `AR1`, `MA1`, `MA2`, `TEAR1`, `QAR1`).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `reps::Int=10_000`: number of replications.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`
  (see [`chart_stat_op`](@ref)).
- `d::Int=1`: delay between observations of a pattern.
- `m::Int=3`: length of the ordinal patterns.
- `ced::Bool=false`: use conditional expected delay initialization.
- `ad::Int=100`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_op_oc(
  op_dgp, lam, cl, reps=10_000; chart_choice, d::Int=1, m::Int=3, ced=false, ad=100, rl_max::Int=typemax(Int)
)

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_op_oc(
      op_dgp, lam, cl, i, op_dgp.dist, chart_choice,
      d, m, ced, ad, rl_max
    )
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end



"""
    rl_op_oc(op_dgp, lam, cl, p_reps, op_dgp_dist, chart_choice, d, m, ced, ad,
      rl_max=typemax(Int))

Compute out-of-control run lengths of the EWMA chart based on ordinal patterns for a
chunk of replications. This is the single-threaded worker used by [`arl_op_oc`](@ref).

- `op_dgp`: out-of-control DGP (e.g. `AR1`, `MA1`, `MA2`, `TEAR1`, `QAR1`).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `p_reps`: range of replication indices to process.
- `op_dgp_dist::UnivariateDistribution`: distribution of the DGP.
- `chart_choice`: chart choice (see [`chart_stat_op`](@ref)).
- `d::Int`: delay between observations of a pattern.
- `m::Int`: length of the ordinal patterns.
- `ced::Bool`: use conditional expected delay initialization.
- `ad::Int`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_op_oc(
  op_dgp, lam, cl, p_reps, op_dgp_dist, chart_choice, d, m, ced, ad, rl_max::Int=typemax(Int)
)
  m_fact = factorial(m)
  rls = zeros(Int, length(p_reps))
  p = zeros(Float64, m_fact)
  bin = zeros(Int, m_fact)
  win = zeros(Int, m)
  idx_used = similar(win)

  if ced
    pool_vector = Vector{Float64}(undef, 10_000)
    init_dgp_op!(op_dgp, pool_vector, op_dgp_dist, 1)
  else
    pool_vector = Float64[]
  end

  x_seq = Vector{Float64}(undef, m + (d > 1 ? d : 0))
  # xbiv not needed for generic OP DGPs

  for r in axes(p_reps, 1)
    if ced
      icrun = true
      while icrun
        fill!(p, 1 / m_fact)
        seq = init_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)
        falarm = false
        for _ in 1:ad
          bin .= 0
          sortperm!(win, seq)
          index = perm_to_lehm_idx!(win, idx_used)
          bin[index] = 1
          fill!(idx_used, 0)

          @. p = lam * bin + (1.0 - lam) * p
          stat = chart_stat_op(p, chart_choice)
          seq = update_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)

          if abort_criterium_op(stat, cl, chart_choice)
            falarm = true
            break
          end
        end
        !falarm && (icrun = false)
      end
    else
      fill!(p, 1 / m_fact)
      # Standard init without eps_long
      seq = init_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
      stat = 0.0
    end

    rl = 0
    while !abort_criterium_op(stat, cl, chart_choice)
      rl += 1
      bin .= 0
      sortperm!(win, seq)
      index = perm_to_lehm_idx!(win, idx_used)
      bin[index] = 1
      fill!(idx_used, 0)

      @. p = lam * bin + (1.0 - lam) * p
      stat = chart_stat_op(p, chart_choice)
      # Standard update
      seq = update_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end


# Methods specifically for MA1 and MA2, which need an extra vector for the epsilons
function rl_op_oc(
  op_dgp::Union{MA1,MA2}, lam, cl, p_reps, op_dgp_dist, chart_choice, d, m, ced, ad, rl_max::Int=typemax(Int)
)
  m_fact = factorial(m)
  rls = zeros(Int, length(p_reps))
  p = zeros(Float64, m_fact)
  bin = zeros(Int, m_fact)
  win = zeros(Int, m)
  idx_used = similar(win)

  # Pre-allocate pool if CED is used
  if ced
    pool_vector = Vector{Float64}(undef, 10_000)
    init_dgp_op!(op_dgp, pool_vector, op_dgp_dist, 1)
  else
    pool_vector = Float64[]
  end

  # Determine sequence lengths for MA states
  offset = (op_dgp isa MA1) ? 1 : 2
  x_seq = Vector{Float64}(undef, m + (d > 1 ? d : 0) + offset)
  eps_long = similar(x_seq)
  # xbiv not needed for generic OP DGPs

  for r in axes(p_reps, 1)
    if ced
      icrun = true
      while icrun
        fill!(p, 1 / m_fact)
        seq = init_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)
        falarm = false
        for _ in 1:ad
          bin .= 0
          sortperm!(win, seq)
          index = perm_to_lehm_idx!(win, idx_used)
          bin[index] = 1
          fill!(idx_used, 0)

          @. p = lam * bin + (1.0 - lam) * p
          stat = chart_stat_op(p, chart_choice)
          seq = update_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)

          if abort_criterium_op(stat, cl, chart_choice)
            falarm = true
            break
          end
        end
        !falarm && (icrun = false)
      end
    else
      fill!(p, 1 / m_fact)
      seq = init_dgp_op!(op_dgp, x_seq, eps_long, op_dgp_dist, d, xbiv)
      stat = 0.0
    end

    rl = 0
    while !abort_criterium_op(stat, cl, chart_choice)
      rl += 1
      bin .= 0
      sortperm!(win, seq)
      index = perm_to_lehm_idx!(win, idx_used)
      bin[index] = 1
      fill!(idx_used, 0)

      @. p = lam * bin + (1.0 - lam) * p
      stat = chart_stat_op(p, chart_choice)
      # Specific update for MA processes using eps_long
      seq = update_dgp_op!(op_dgp, x_seq, eps_long, op_dgp_dist, d)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end






################################################################################
#     OLD VERSIONS OF rl_op_oc() - KEPT FOR REFERENCE, NOT USED IN FINAL CODE
################################################################################
# function rl_op_oc(
#   op_dgp, lam, cl, p_reps, op_dgp_dist::Union{ContinuousUnivariateDistribution,Nothing}, chart_choice, d, m, ced, ad
# )

#   # Pre-allocate variables
#   m_fact = factorial(m)
#   rls = zeros(Int, length(p_reps))
#   p = zeros(Float64, m_fact)
#   bin = zeros(Int, m_fact)
#   win = zeros(Int, m)
#   idx_used = similar(win)

#   # Create pool vector for CED runs (if "ced=true")
#   # If true, create and fill vector with initial values
#   if ced
#     pool_vector = Vector{Float64}(undef, 10_000)
#     init_dgp_op!(op_dgp, pool_vector, op_dgp_dist, 1)
#   else
#     pool_vector = Float64[]
#   end

#   # Check for MA1 and MA2 and compute length of the vectors accordingly
#   if op_dgp isa MA1

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m + 1)
#       eps_long = similar(x_seq)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d + 1)
#       eps_long = similar(x_seq)
#     end

#   elseif op_dgp isa MA2

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m + 2)
#       eps_long = similar(x_seq)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d + 2)
#       eps_long = similar(x_seq)
#     end

#     # Anything other than MA1 or MA2
#   else

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m)
#       eps_long = similar(x_seq)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d)
#       eps_long = similar(x_seq)
#     end

#   end

#   xbiv = Vector{Float64}(undef, ad) # burn-in vector for AAR(1) and QAR(1) DGPs

#   for r in axes(p_reps, 1) # p_reps is a range

#     # ------------------------------------------------------------------------------
#     # ---------------------      check whether to use ced     ----------------------
#     # ------------------------------------------------------------------------------
#     if ced

#       icrun = true

#       while icrun

#         fill!(p, 1 / 6)
#         seq = init_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)

#         falarm = false

#         for _ in 1:ad

#           bin .= 0
#           # compute ordinal pattern based on permutations
#           sortperm!(win, seq)

#           # binarization of ordinal pattern
#           index = perm_to_lehm_idx!(win, idx_used)
#           bin[index] = 1
#           fill!(idx_used, 0)

#           # compute EWMA statistic
#           @. p = lam * bin .+ (1 - lam) * p
#           # test statistic
#           stat = chart_stat_op(p, chart_choice)
#           # update sequence depending on DGP
#           seq = update_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)
#           # check whether false alarm 
#           if abort_criterium_op(stat, cl, chart_choice)
#             falarm = true
#           end

#         end # for ad run
#         # in case of no false alarm, set icrun to false and step out of while loop
#         if falarm == false
#           icrun = false
#         end
#       end

#     end
#     # ------------------------------------------------------------------------------

#     # initialize run length at zero
#     rl = 0

#     # check whether to use ced. If ced is used, update observations. Otherwise, initialize observations
#     if ced
#       seq = update_dgp_op!(op_dgp, x_seq, eps_long, op_dgp_dist, d)
#     else
#       seq = init_dgp_op!(op_dgp, x_seq, eps_long, op_dgp_dist, d, xbiv)
#       fill!(p, 1 / m_fact)
#       stat = chart_stat_op(p, chart_choice)
#     end

#     while !abort_criterium_op(stat, cl, chart_choice)
#       # increase run length
#       rl += 1
#       bin .= 0

#       # binarization of ordinal pattern
#       sortperm!(win, seq)
#       index = perm_to_lehm_idx!(win, idx_used)
#       bin[index] = 1
#       fill!(idx_used, 0)

#       # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
#       @. p = lam * bin .+ (1 - lam) * p
#       # statistic based on smoothed p-estimate
#       stat = chart_stat_op(p, chart_choice)
#       # update sequence depending on DGP
#       seq = update_dgp_op!(op_dgp, x_seq, eps_long, op_dgp_dist, d)
#     end

#     rls[r] = rl
#   end
#   return rls
# end


# function rl_op_oc(
#   op_dgp, lam, cl, p_reps, op_dgp_dist::DiscreteUnivariateDistribution, chart_choice, d, m, ced, ad
# )

#   # Pre-allocate variables
#   m_fact = factorial(m)
#   rls = zeros(Int, length(p_reps))
#   p = zeros(Float64, m_fact)
#   bin = zeros(Int, m_fact)
#   win = zeros(Int, m)
#   idx_used = similar(win)

#   # Check for MA1 and MA2 and compute length of the vectors accordingly
#   if op_dgp isa MA1

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m + 1)
#       eps_long = similar(x_seq)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d + 1)
#       eps_long = similar(x_seq)
#     end

#   elseif op_dgp isa MA2

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m + 2)
#       eps_long = similar(x_seq)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d + 2)
#       eps_long = similar(x_seq)
#     end

#     # Anything other than MA1 or MA2
#   else

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m)
#       eps_long = similar(x_seq)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d)
#       eps_long = similar(x_seq)
#     end
#   end

#   for r in axes(p_reps, 1) # p_reps is a range

#     # ------------------------------------------------------------------------------
#     # ---------------------      check whether to use ced     ----------------------
#     # ------------------------------------------------------------------------------
#     if ced

#       icrun = true

#       while icrun

#         fill!(p, 1 / 6)
#         seq = init_dgp_op_ced!(op_dgp, x_seq, d)

#         falarm = false

#         for _ in 1:ad

#           bin .= 0
#           # compute ordinal pattern based on permutations
#           sortperm!(win, seq)

#           # binarization of ordinal pattern
#           index = perm_to_lehm_idx!(win, idx_used)
#           bin[index] = 1
#           fill!(idx_used, 0)

#           # compute EWMA statistic
#           @. p = lam * bin .+ (1 - lam) * p
#           # test statistic
#           stat = chart_stat_op(p, chart_choice)
#           # update sequence depending on DGP
#           seq = update_dgp_op_ced!(op_dgp, x_seq, d)
#           # check whether false alarm 
#           if abort_criterium_op(stat, cl, chart_choice)
#             falarm = true
#           end

#         end # for ad run
#         # false alarm -> set icrun to false and step out of while loop
#         if falarm == false
#           icrun = false
#         end
#       end

#     end
#     # ------------------------------------------------------------------------------

#     # initialize run length at zero
#     rl = 0

#     # check whether to use ced. If ced is used, update observations. Otherwise, initialize observations
#     if ced
#       seq = update_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
#     else
#       seq = init_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
#       fill!(p, 1 / m_fact)
#       stat = chart_stat_op(p, chart_choice)
#     end

#     while !abort_criterium_op(stat, cl, chart_choice)
#       # increase run length
#       rl += 1
#       bin .= 0

#       # binarization of ordinal pattern
#       sortperm!(win, seq)
#       index = perm_to_lehm_idx!(win, idx_used)
#       bin[index] = 1
#       fill!(idx_used, 0)

#       # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
#       @. p = lam * bin .+ (1 - lam) * p
#       # statistic based on smoothed p-estimate
#       stat = chart_stat_op(p, chart_choice)
#       # update sequence depending on DGP
#       seq = update_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
#     end

#     rls[r] = rl
#   end
#   return rls
# end


# function rl_op_oc(
#   op_dgp, lam, cl, p_reps, op_dgp_dist::Nothing, chart_choice, d, m, ced, ad
# )

#   # Pre-allocate variables
#   m_fact = factorial(m)
#   rls = zeros(Int, length(p_reps))
#   p = zeros(Float64, m_fact)
#   bin = zeros(Int, m_fact)
#   win = zeros(Int, m)
#   idx_used = similar(win)

#   # Check for MA1 and MA2 and compute length of the vectors accordingly
#   if op_dgp isa MA1

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m + 1)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d + 1)
#     end

#   elseif op_dgp isa MA2

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m + 2)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d + 2)
#     end

#     # Anything other than MA1 or MA2
#   else

#     if d isa Int && d == 1
#       x_seq = Vector{Float64}(undef, m)
#     elseif d isa Int && d > 1
#       x_seq = Vector{Float64}(undef, m + d)
#     end
#   end

#   for r in axes(p_reps, 1) # p_reps is a range

#     # ------------------------------------------------------------------------------
#     # ---------------------      check whether to use ced     ----------------------
#     # ------------------------------------------------------------------------------
#     if ced

#       icrun = true

#       while icrun

#         fill!(p, 1 / 6)
#         seq = init_dgp_op_ced!(op_dgp, x_seq, d)

#         falarm = false

#         for _ in 1:ad

#           bin .= 0
#           # compute ordinal pattern based on permutations
#           sortperm!(win, seq)

#           # binarization of ordinal pattern
#           index = perm_to_lehm_idx!(win, idx_used)
#           bin[index] = 1
#           fill!(idx_used, 0)

#           # compute EWMA statistic
#           @. p = lam * bin .+ (1 - lam) * p
#           # test statistic
#           stat = chart_stat_op(p, chart_choice)
#           # update sequence depending on DGP
#           seq = update_dgp_op_ced!(op_dgp, x_seq, d)
#           # check whether false alarm 
#           if abort_criterium_op(stat, cl, chart_choice)
#             falarm = true
#           end

#         end # for ad run
#         # in case of no false alarm, set icrun to false and step out of while loop
#         if falarm == false
#           icrun = false
#         end
#       end

#     end
#     # ------------------------------------------------------------------------------

#     # initialize run length at zero
#     rl = 0

#     # check whether to use ced. If ced is used, update observations. Otherwise, initialize observations
#     if ced
#       seq = update_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
#     else
#       seq = init_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
#       fill!(p, 1 / m_fact)
#       stat = chart_stat_op(p, chart_choice)
#     end

#     while !abort_criterium_op(stat, cl, chart_choice)
#       # increase run length
#       rl += 1
#       bin .= 0

#       # binarization of ordinal pattern
#       sortperm!(win, seq)
#       index = perm_to_lehm_idx!(win, idx_used)
#       bin[index] = 1
#       fill!(idx_used, 0)

#       # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
#       @. p = lam * bin .+ (1 - lam) * p
#       # statistic based on smoothed p-estimate
#       stat = chart_stat_op(p, chart_choice)
#       # update sequence depending on DGP
#       seq = update_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
#     end

#     rls[r] = rl
#   end
#   return rls
# end
