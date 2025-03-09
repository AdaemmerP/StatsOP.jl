
"""
    arl_op_ic( op_dgp, lam, cl, reps=10_000; chart_choice, d=1, ced=false, ad=100)

Function to compute the average run length (ARL) for ordinal patterns using the EWMA statistic. The function implements the test statistics by Weiss and Testik (2023), who use a pattern length of 3. 

* `op_dgp::Union{AR1, MA1, MA2, TEAR1, AAR1, QAR1}` DGP.
* `lam::Float64` Smoothing parameter for the EWMA statistic.
* `cl::Float64` Control limit for the EWMA statistic.
* `reps::Int64` Number of replications.
* `chart_choice::Int` 
  1. ``\\widehat{H}^{(d)}=-\\sum_{k=1}^{m!} \\hat{p}_k{ }^{(d)} \\ln \\hat{p}_k{ }^{(d)}``
  2. ``\\widehat{H}_{\\mathrm{ex}}^{(d)}=-\\sum_{k=1}^{m!}\\left(1-\\hat{p}_k{ }^{(d)}\\right) \\ln \\left(1-\\hat{p}_k{ }^{(d)}\\right)``
  3. ``\\widehat{\\Delta}^{(d)}=\\sum_{k=1}^{m!}\\left(\\hat{p}_k^{(d)}-1 / m!\\right)^2``
  4. ``\\hat{\\beta}^{(d)}=\\hat{p}_6^{(d)}-\\hat{p}_1^{(d)}``
  5. ``\\hat{\\tau}^{(d)}=\\hat{p}_6^{(d)}+\\hat{p}_1^{(d)}-\\frac{1}{3}``
  6. ``\\hat{\\delta}^{(d)}=\\hat{p}_4^{(d)}+\\hat{p}_5^{(d)}-\\hat{p}_3^{(d)}-\\hat{p}_2^{(d)}``

  The patterns are categorized as follows:

  ``
  \\qquad p_1 = (3,2,1);  \\quad p_2=(3,1,2);  \\quad p_3 = (2,3,1); 
  ``

  ``
  \\qquad p_4 = (1,3,2);  \\quad p_5 = (2,1,3);  \\quad p_ 6 = (1,2,3)
  ``

* `d::Union{Int,Vector{Int}}=1`: Delay vector. Default is 1. A vector would denote the indices of the observations to use. For example, 
`d = [1, 3, 4]` would denote the first, third, and fourth observations.
* `ced::Bool=false`: Use conditional expected delay? Default is false.
* `ad::Int=100`: Number of iterations for ced.

```julia
# Compute initial values via function cl_op()
 if j == 1 || j == 2
      cl_init = quantile(stat_op(data, lam[i], j)[1], 0.01)                
  else
      cl_init = quantile(stat_op(data, lam[i], j)[1], 0.99)
end 

# Run function
arl_op(0.1, cl_init, IC(Normal(0, 1)), 10_000; chart_choice=1, d=1, ced=false, ad=100)
```
"""
function arl_op_ic(op_dgp::ICTS, lam, cl, reps=10_000; chart_choice, d::Union{Int,Vector{Int}}=1, ced=false, ad=100) 

  # Compute lookup array and number of ops
  lookup_array_op = compute_lookup_array_op()

  # Check whether to use threading or multi processing -
  if nprocs() == 1

    # Make chunks for separate tasks (based on number of threads)   
    chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect

    # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
    par_results = map(chunks) do i

      Threads.@spawn rl_op_ic(op_dgp, lam, cl, lookup_array_op, i, op_dgp.dist, chart_choice; d=d, ced=ced, ad=ad)

    end

  elseif nprocs() > 1

    # Make chunks for separate tasks (based on number of workers)
    chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect

    par_results = pmap(chunks) do i
      rl_op_ic(op_dgp, lam, cl, lookup_array_op, i,op_dgp.dist, chart_choice; d=d, ced=ced, ad=ad)
    end

  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_op_ic(lam, cl, lookup_array_op, p_reps, op_dgp,
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
function rl_op_ic(
  op_dgp::ICTS, lam, cl, lookup_array_op, p_reps, 
  op_dgp_dist, chart_choice; d::Union{Int,Vector{Int}}=1, ced=false, ad=100
  )

  # value of patterns (can become variable in future versions)
  op_length = 3

  # if d is a vector of integer, check whether its length equals "m" (3 by default)
  if d isa Vector{Int}
    if length(d) != op_length
      throw(ArgumentError("Length of delay vector must be equal to op_length (3 by default)."))
    end
  end

  # Pre-allocate variables
  m! = factorial(op_length)
  rls = Vector{Int64}(undef, length(p_reps)) 
  p = Vector{Float64}(undef, m!)
  bin = Vector{Int64}(undef, m!) 
  win = Vector{Int64}(undef, op_length) 

  # Compute vectors accordingly
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
