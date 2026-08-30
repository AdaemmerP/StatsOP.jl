

"""
    arl_gop_oc(gop_dgp, null_dist, lam, cl, reps; chart_choice, d=1, ced=false, ad=100,
      rl_max=typemax(Int))

Compute the out-of-control average run length (ARL) of the EWMA chart based on
generalized ordinal patterns (GOPs) via simulation. The computation is multithreaded.

- `gop_dgp`: out-of-control DGP.
- `null_dist`: in-control (null) distribution used to compute the in-control GOP
  distribution.
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `reps::Int`: number of replications.
- `chart_choice`: [`D_Chart`](@ref)`()` or `Persistence()`.
- `d=1`: delay between observations of a pattern.
- `ced::Bool=false`: use conditional expected delay initialization.
- `ad::Int=100`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_gop_oc(
  gop_dgp, null_dist, lam, cl, reps; chart_choice, d=1, ced=false, ad=100, rl_max::Int=typemax(Int)
)

  # Compute lookup array and number of ops
  lookup_array_gop = _LOOKUP_GOP

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_gop_oc(
      lam, cl, lookup_array_gop, i, gop_dgp, gop_dgp.dist, null_dist, chart_choice, d, ced, ad, rl_max
    )
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

"""
    rl_gop_oc(lam, cl, lookup_array_gop, p_reps, gop_dgp, gop_dgp_dist, null_dist,
      chart_choice, d, ced, ad, rl_max=typemax(Int))

Compute out-of-control run lengths of the EWMA chart based on generalized ordinal
patterns (GOPs) for a chunk of replications. This is the single-threaded worker used by
[`arl_gop_oc`](@ref).

- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `lookup_array_gop`: lookup array from [`compute_lookup_array_gop`](@ref).
- `p_reps`: range of replication indices to process.
- `gop_dgp`: out-of-control DGP.
- `gop_dgp_dist`: marginal distribution of `gop_dgp`.
- `null_dist`: in-control (null) distribution used to compute the in-control GOP
  distribution.
- `chart_choice`: [`D_Chart`](@ref)`()` or `Persistence()`.
- `d::Int`: delay between observations of a pattern.
- `ced::Bool`: use conditional expected delay initialization.
- `ad::Int`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_gop_oc(
  lam, cl, lookup_array_gop, p_reps, gop_dgp, gop_dgp_dist, null_dist,
  chart_choice::Union{D_Chart,Persistence}, d::Int, ced::Bool, ad::Int, rl_max::Int=typemax(Int)
)

  # Pattern size
  m = 3

  # Pre-allocate variables
  rls = zeros(Int, length(p_reps))
  bin = zeros(Int, 13)
  win = zeros(Int, m)
  ix = similar(win)
  pₜ = zeros(13)
  p₀ = zeros(13)
  pₜ_p₀ = zeros(13)
  fill_p0!(p₀, null_dist)

  # Compute length of 'x_seq' vector based on delay d
  x_seq = zeros(1 + (m - 1) * d)

  # Create pool vector for CED runs (stationary distribution of null process)
  if ced
    pool_vector = Vector{Float64}(undef, 10_000)
    # Using null_dist to represent the in-control stationary state
    init_dgp_op!(gop_dgp, pool_vector, null_dist, 1)
  else
    pool_vector = Float64[]
  end

  for r in axes(p_reps, 1)

    #----------------------------------------------------------------------#
    # 1. Initialization / CED Phase (In-Control stationary phase)
    #----------------------------------------------------------------------#
    if ced
      icrun = true
      while icrun
        pₜ .= p₀
        # Initialize x_seq from the stationary pool
        seq = init_dgp_op_ced!(gop_dgp, x_seq, pool_vector, d)
        falarm = false

        for _ in 1:ad
          bin .= 0
          competerank!(win, seq, ix)
          j, k, l = win
          bin[lookup_array_gop[j, k, l]] = 1

          # Update EWMA using null mean p₀
          @. pₜ = lam * bin + (1 - lam) * pₜ
          @. pₜ_p₀ = pₜ - p₀
          stat = chart_stat_gop(pₜ_p₀, chart_choice)

          # Update from stationary pool
          seq = update_dgp_op_ced!(gop_dgp, x_seq, pool_vector, d)
          fill!(win, 0)

          if abort_criterium_gop(stat, cl, chart_choice)
            falarm = true
            break
          end
        end

        if !falarm
          icrun = false
        end
      end
      # Transition: No extra update here. 
      # seq is ready for the first OOC observation from gop_dgp_dist.
    else
      # Standard initialization for immediate OOC phase
      pₜ .= p₀
      seq = init_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
      # Neutral start to ensure loop enters and processes first point at rl=1
      @. pₜ_p₀ = 0.0
      stat = chart_stat_gop(pₜ_p₀, chart_choice)
    end

    #----------------------------------------------------------------------#
    # 2. Run Length (RL) Phase (Out-of-Control)
    #----------------------------------------------------------------------#
    rl = 0

    while !abort_criterium_gop(stat, cl, chart_choice)
      rl += 1
      bin .= 0

      # compute ordinal pattern on current sequence
      competerank!(win, seq, ix)
      j, k, l = win
      bin[lookup_array_gop[j, k, l]] = 1

      # Update EWMA
      @. pₜ = lam * bin + (1 - lam) * pₜ
      @. pₜ_p₀ = pₜ - p₀
      stat = chart_stat_gop(pₜ_p₀, chart_choice)

      # update sequence depending on OOC DGP
      seq = update_dgp_op!(gop_dgp, x_seq, gop_dgp_dist, d)
      fill!(win, 0)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end

    rls[r] = rl
  end
  return rls
end
