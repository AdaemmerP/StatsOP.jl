
# TODO: Write one function for sequential charts, using λ, 
# TODO: Write one function for one chart, without λ,    


# Function to chart statistic and relative frequencies of ordinal patterns
function stat_op(data; chart_choice, m::Int=3, d::Int=1)

  # Compute lookup array and number of ops
  lookup_array_op = compute_lookup_array_op(m=m)
  m! = factorial(m)

  p_vec = Vector{Float64}(undef, m!)
  p_count = zeros(Int, m!)
  fill!(p_vec, 1 / m!)
  bin = Vector{Int64}(undef, m!)
  win = Vector{Int64}(undef, m)

  for range_start in 1:(length(data)-(m-1)*d) #for (i, j) in enumerate(dindex_ranges)

    # Reset binarization vector
    fill!(bin, 0)

    # create unit range for indexing data
    unit_range = range(range_start; step=d, length=m)

    x_long = view(data, unit_range)

    # compute ordinal pattern based on permutations    
    sortperm!(win, x_long)

    # Binarization of ordinal pattern
    if m == 2
      bin[lookup_array_op[win[1], win[2]]] = 1
    elseif m == 3
      bin[lookup_array_op[win[1], win[2], win[3]]] = 1
    end

    @. p_count += bin

  end

  p_rel = p_count ./ sum(p_count) #length(dindex_ranges)
  stat = chart_stat_op(p_rel, chart_choice)
  return [stat, p_rel]

end



# Function to compute EWMA chart statistic
function stat_op(data, lam; chart_choice, m::Int=3, d::Int=1)
  #stat_op(data, lam, chart_choice; m::Int=3, d=1)

  # create vector with unit range for indexing 
  dindex_ranges = compute_dindex_op(data; m=m, d=d)

  # Compute lookup array and number of ops
  lookup_array_op = compute_lookup_array_op(m=m)
  m! = factorial(m)

  p_vec = Vector{Float64}(undef, m!)
  p_count = zeros(Int, m!)
  fill!(p_vec, 1 / m!)
  bin = Vector{Int64}(undef, m!)
  win = Vector{Int64}(undef, m)
  stats_all = Vector{Float64}(undef, length(dindex_ranges))

  for (i, j) in enumerate(dindex_ranges)

    # Reset binarization vector
    fill!(bin, 0)

    x_long = view(data, j)

    # compute ordinal pattern based on permutations
    order_vec!(x_long, win)
    # Binarization of ordinal pattern
    if m == 2
      bin[lookup_array_op[win[1], win[2]]] = 1
    elseif m == 3
      bin[lookup_array_op[win[1], win[2], win[3]]] = 1
    end
    # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
    @. p_vec = lam * bin + (1 - lam) * p_vec
    @. p_count += bin
    # statistic based on smoothed p-estimate
    stat = chart_stat_op(p_vec, chart_choice)
    # Save temporary test statistic
    stats_all[i] = stat
  end

  p_rel = p_count ./ sum(p_count) #length(dindex_ranges)
  return [stats_all, p_rel]

end

