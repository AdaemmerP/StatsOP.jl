"""
    rl_op(lam, cl, lookup_array_op, p_reps, op_dgp,
      op_dgp_dist, chart_choice; d::Union{Int,Vector{Int}}=1, ced=false, ad=100)

Function to compute run length for ordinal patterns. 
  
- `lam::Float64`: Smoothing parameter for EWMA chart.
- `cl::Float64`: Control limit for the EWMA chart.
- `lookup_array_op: Array to lookup ordinal patterns. Can be created with function `compute_lookup_array_op()`.
- `p_reps::Vector{Int}`: Unit range of repetitions.
- `op_dgp::Union{IC, AR1, MA1, MA2, TEAR1, AAR1, QAR1}`: DGP.
- `op_dgp_dist::UnivariateDistribution`: Distribution of the DGP.
- `chart_choice::Int`: Chart choice (1: XXX, 2: XXX, 3: XXX, 4: XXX, 5: XXX, 6: XXX).
- `d::Union{Int,Vector{Int}}=1`: Delay vector. Default is 1.
- `ced::Bool=false`: Use conditional expected delay? Default is false.
- `ad::Int=100`: Number of iterations for ced. 

```julia
rl_op(0.1, 3.0, lookup_array_op, 1:10_000, IC(Normal(0, 1)), Normal(0, 1), 1; d=1, ced=false, ad=100)
```
"""
function rl_op(lam, cl, lookup_array_op, p_reps, op_dgp,
  op_dgp_dist, chart_choice; d::Union{Int,Vector{Int}}=1, ced=false, ad=100)

  # value of patterns (can become variable in future versions)
  op_length = 3

  # if d is a vector of integer, check whether its length equals "m" (3 by default)
  if d isa Vector{Int}
    if length(d) != op_length
      throw(ArgumentError("Length of delay vector must be equal toop_length (3 by default)."))
    end
  end

  # Pre-allocate variables
  m! = factorial(op_length)
  rls = Vector{Int64}(undef, length(p_reps)) # Vector{Int64}(undef, length(p_reps)) # 
  p = Vector{Float64}(undef, m!) # MVector{6,Float64}(undef) # 
  bin = Vector{Int64}(undef, m!) # MVector{6,Int}(undef)  # 
  win = Vector{Int64}(undef, op_length) # MVector{3,Int}(undef)  # 

  # Check for MA1 and MA2 and compute length of the vectors accordingly
  if op_dgp isa MA1

    if d isa Int && d == 1
      x_long = Vector{Float64}(undef, op_length + 1)
      eps_long = similar(x_long)
    elseif d isa Int && d > 1
      x_long = Vector{Float64}(undef, op_length + d + 1)
      eps_long = similar(x_long)
    elseif d isa Vector{Int}
      x_long = Vector{Float64}(undef, last(d) + 1)
      eps_long = similar(x_long)
    end

  elseif op_dgp isa MA2

    if d isa Int && d == 1
      x_long = Vector{Float64}(undef, op_length + 2)
      eps_long = similar(x_long)
    elseif d isa Int && d > 1
      x_long = Vector{Float64}(undef, op_length + d + 2)
      eps_long = similar(x_long)
    elseif d isa Vector{Int}
      x_long = Vector{Float64}(undef, last(d) + 2)
      eps_long = similar(x_long)
    end

    # Anything other than MA1 or MA2
  else

    if d isa Int && d == 1
      x_long = Vector{Float64}(undef, op_length)
      eps_long = similar(x_long)
    elseif d isa Int && d > 1
      x_long = Vector{Float64}(undef, op_length + d)
      eps_long = similar(x_long)
    elseif d isa Vector{Int}
      x_long = Vector{Float64}(undef, last(d))
      eps_long = similar(x_long)
    end

  end

  xbiv = Vector{Float64}(undef, ad) # burn-in vector for AAR(1) and QAR(1) DGPs

  for r in axes(p_reps, 1) # p_reps is a range

    # ------------------------------------------------------------------------------
    # ---------------------      check whether to use ced     ----------------------
    # ------------------------------------------------------------------------------
    if ced

      icrun = true

      while icrun

        fill!(p, 1 / 6)
        seq = init_dgp_op_ced!(op_dgp, x_long, d)

        falarm = false

        for t in 1:ad

          bin .= 0 # zeros(6)
          # compute ordinal pattern based on permutations
          order_vec!(seq, win)
          # binarization of ordinal pattern
          bin[lookup_array_op[win[1], win[2], win[3]]] = 1
          # compute EWMA statistic
          @. p = lam * bin .+ (1 - lam) * p
          # test statistic
          stat = chart_stat_op(p, chart_choice)
          # update sequence depending on DGP
          seq = update_dgp_op_ced!(op_dgp, x_long, d)
          # check whether false alarm 
          if !abort_criterium_op(stat, cl, chart_choice)
            falarm = true
          end

        end # for ad run
        # in case of no false alarm, set icrun to false and step out of while loop
        if falarm == false
          icrun = false
        end
      end

    end
    # ------------------------------------------------------------------------------

    # initialize run length at zero
    rl = 0

    # check whether to use ced. If ced is used, update observations. Otherwise, initialize observations
    if ced
      #seq = update_dgp_op!(op_dgp, x_long, op_dgp_dist, d)
      seq = update_dgp_op!(op_dgp, x_long, eps_long, op_dgp_dist, d)
    else
      #seq = init_dgp_op!(op_dgp, x_long, op_dgp_dist, d, xbiv)
      seq = init_dgp_op!(op_dgp, x_long, eps_long, op_dgp_dist, d, xbiv)
      # in-control OP-distribution            
      fill!(p, 1 / 6)
      stat = chart_stat_op(p, chart_choice)
    end

    while abort_criterium_op(stat, cl, chart_choice)
      # increase run length
      rl += 1
      bin .= 0
      # compute ordinal pattern based on permutations        
      order_vec!(seq, win)
      # Binarization of ordinal pattern
      bin[lookup_array_op[win[1], win[2], win[3]]] = 1
      # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
      @. p = lam * bin .+ (1 - lam) * p
      # statistic based on smoothed p-estimate
      stat = chart_stat_op(p, chart_choice)
      # update sequence depending on DGP
      #seq = update_dgp_op!(op_dgp, x_long, op_dgp_dist, d)
      seq = update_dgp_op!(op_dgp, x_long, eps_long, op_dgp_dist, d)
    end

    rls[r] = rl
  end
  return rls
end

#--- Function to compute average run length for ordinal patterns
function arl_op(lam, cl, op_dgp, reps=10_000; chart_choice, d=1, ced=false, ad=100) 

  # Compute lookup array and number of ops
  lookup_array_op = compute_lookup_array_op()

  # Check whether to use threading or multi processing --> only one process threading, else distributed
  if nprocs() == 1

    # Make chunks for separate tasks (based on number of threads)   
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_op(lam, cl, lookup_array_op, i, op_dgp, op_dgp.dist, chart_choice; d=d, ced=ced, ad=ad)

    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_op(lam, cl, lookup_array_op, i, op_dgp, op_dgp.dist, chart_choice; d=d, ced=ced, ad=ad)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end

# --- Function to compute control limit for OPs --- #
function cl_op(lam, L0, op_dgp, cl_init, reps=10000; chart_choice, jmin=4, jmax=6, verbose=false, d=1, ced=false, ad=100)
         
  L1 = zeros(2)
  ii = Int
  if cl_init == 0
    for i in 1:20
      L1 = arl_op(lam, cl_init, op_dgp, reps; chart_choice=chart_choice, d=d, ced=ced, ad=ad) 

      if verbose
        println("cl = ", i / 20, "\t", "ARL = ", L1[1])
      end
      if L1[1] > L0
        ii = i
        break
      end
    end
    cl_init = ii / 20
  end

  for j in jmin:jmax
    for dh in 1:80

      if (chart_choice == 1 || chart_choice == 2)
        cl_init = cl_init - (-1)^j * dh / 10^j
      else
        cl_init = cl_init + (-1)^j * dh / 10^j
      end
      L1 = arl_op(lam, cl_init, op_dgp, reps; chart_choice=chart_choice, d=d, ced=ced, ad=ad) 

      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1[1])
      end
      if (j % 2 == 1 && L1[1] < L0) || (j % 2 == 0 && L1[1] > L0)
        break
      end
    end
    cl_init = cl_init
  end
  if L1[1] < L0
    cl = cl_init + 1 / 10^jmax
  end
  return cl_init
end

