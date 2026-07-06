
"""
    arl_op_ic(op_dgp, lam, cl, reps=10_000; chart_choice, d=1, m=3, ced=false, ad=100,
      rl_max=typemax(Int))

Compute the in-control average run length (ARL) of the EWMA chart based on ordinal
patterns via simulation, following Weiß and Testik (2023). The computation is
multithreaded.

- `op_dgp::Union{ContinuousDGPIC,DiscreteDGPIC}`: in-control DGP.
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

```julia
arl_op_ic(
  ContinuousDGPIC(Normal(0, 1)), 0.1, 0.3, 10_000; chart_choice=Shannon(), d=1
)
```
"""
function arl_op_ic(
  op_dgp::Union{ContinuousDGPIC,DiscreteDGPIC}, lam, cl, reps=10_000; chart_choice, d::Int=1, m::Int=3, ced=false, ad=100, rl_max::Int=typemax(Int)
)

  # Compute lookup array and number of ops
  #lookup_array_op = compute_lookup_array_op()

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_op_ic(
      op_dgp, lam, cl, i, op_dgp.dist, chart_choice; d=d, m=m, ced=ced, ad=ad, rl_max=rl_max
    )
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_op_ic(op_dgp, lam, cl, p_reps, op_dgp_dist, chart_choice; d=1, m, ced=false,
      ad=100, rl_max=typemax(Int))

Compute in-control run lengths of the EWMA chart based on ordinal patterns for a chunk
of replications. This is the single-threaded worker used by [`arl_op_ic`](@ref).

- `op_dgp::Union{ContinuousDGPIC,DiscreteDGPIC}`: in-control DGP.
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `p_reps`: range of replication indices to process.
- `op_dgp_dist`: distribution of the DGP.
- `chart_choice`: chart choice (see [`chart_stat_op`](@ref)).
- `d::Int=1`: delay between observations of a pattern.
- `m::Int`: length of the ordinal patterns.
- `ced::Bool=false`: use conditional expected delay initialization.
- `ad::Int=100`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_op_ic(
  op_dgp::Union{ContinuousDGPIC,DiscreteDGPIC}, lam, cl, p_reps,
  op_dgp_dist, chart_choice; d::Int=1, m::Int, ced=false, ad=100, rl_max::Int=typemax(Int)
)

  # Pre-allocate variables
  m_fact = factorial(m)
  rls = zeros(Int, length(p_reps))
  p = zeros(Float64, m_fact)
  bin = zeros(Int, m_fact)
  win = zeros(Int, m)
  idx_used = similar(win)

  # Compute vector length based on delay d
  if d isa Int && d == 1
    x_vec = Vector{Float64}(undef, m)
  elseif d isa Int && d > 1
    x_vec = Vector{Float64}(undef, m + d)
  end

  for r in axes(p_reps, 1)

    # -------------------------------------------------------------------------#
    # 1. Initialization / CED Phase (In-Control)
    # -------------------------------------------------------------------------#
    if ced
      icrun = true
      while icrun

        # Initialize probabilities to uniform distribution
        fill!(p, 1 / m_fact)
        seq = init_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)
        falarm = false

        for _ in 1:ad
          bin .= 0
          sortperm!(win, seq)
          index = perm_to_lehm_idx!(win, idx_used)
          bin[index] = 1
          fill!(idx_used, 0)

          # Compute EWMA statistic
          @. p = lam * bin + (1.0 - lam) * p
          stat = chart_stat_op(p, chart_choice)

          # Update prepares the sequence for the next step
          seq = update_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)

          if abort_criterium_op(stat, cl, chart_choice)
            falarm = true
            break
          end
        end

        if !falarm
          icrun = false
        end
      end
      # After CED: seq is ready for step ad+1. No extra update here.
    else
      # Standard initialization: no warm-up delay
      fill!(p, 1 / m_fact)
      seq = init_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)
      stat = chart_stat_op(p, chart_choice)
    end

    # -------------------------------------------------------------------------#
    # 2. Run Length (RL) Phase
    # -------------------------------------------------------------------------#
    rl = 0

    # Loop enters for the first monitor step (t=1 or t=ad+1)
    while !abort_criterium_op(stat, cl, chart_choice)
      rl += 1
      bin .= 0

      # compute ordinal pattern
      sortperm!(win, seq)
      index = perm_to_lehm_idx!(win, idx_used)
      bin[index] = 1
      fill!(idx_used, 0)

      # Update EWMA
      @. p = lam * bin + (1.0 - lam) * p
      stat = chart_stat_op(p, chart_choice)

      # Prepare sequence for next iteration
      seq = update_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end

    rls[r] = rl
  end
  return rls
end


# function rl_op_ic(
#   op_dgp::Union{ContinuousDGPIC,DiscreteDGPIC}, lam, cl, p_reps,
#   op_dgp_dist, chart_choice; d::Int=1, m::Int, ced=false, ad=100
# )

#   # Pre-allocate variables
#   m_fact = factorial(m)
#   rls = zeros(Int, length(p_reps))
#   p = zeros(Float64, m_fact)
#   bin = zeros(Int, m_fact)
#   win = zeros(Int, m)
#   idx_used = similar(win)

#   # Compute vectors accordingly
#   if d isa Int && d == 1
#     x_vec = Vector{Float64}(undef, m)
#   elseif d isa Int && d > 1
#     x_vec = Vector{Float64}(undef, m + d)
#   end

#   # burn-in vector for AAR(1) and QAR(1) DGPs
#   # xbiv = zeros(Float64, ad)

#   for r in axes(p_reps, 1) # p_reps is a range

#     # ------------------------------------------------------------------------------
#     # ---------------------      check whether to use ced     ----------------------
#     # ------------------------------------------------------------------------------
#     if ced

#       icrun = true

#       while icrun

#         fill!(p, 1 / m_fact)
#         seq = init_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)

#         # Add noise when using discrete distribution
#         add_noise!(vec, op_dgp_dist)

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
#           seq = update_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)
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
#       seq = update_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)
#     else
#       seq = init_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)
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
#       seq = update_dgp_op!(op_dgp, x_vec, op_dgp_dist, d)
#     end

#     rls[r] = rl
#   end
#   return rls
# end
