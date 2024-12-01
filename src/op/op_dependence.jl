
# Function to only count the number of ordinal patterns in bins for one time series
function count_uv_op(ts; op_length::Int=3, d=1)

  # Assert that 2 <= op_length <= 4
  @assert 2 <= op_length <= 4 "This function is only implemented for pattern lengths of 2, 3 and 4"

  # create vector with unit range for indexing 
  dindex_ranges = compute_dindex_op(ts; op_length=op_length, d=d)

  # Compute lookup array and number of ops
  lookup_array_op = compute_lookup_array_op(op_length=op_length)
  op_length! = factorial(op_length)
  p_count = zeros(Int, op_length!)     
  bin = Vector{Int}(undef, op_length!) 
  win = Vector{Int}(undef, op_length) 
  seq = Vector{Float64}(undef, op_length) 

  for (i, j) in enumerate(dindex_ranges)

    seq = view(ts, j)
    fill!(bin, 0)
    order_vec!(seq, win)

    if op_length == 2
      bin[lookup_array_op[win[1], win[2]]] = 1
    elseif op_length == 3
      bin[lookup_array_op[win[1], win[2], win[3]]] = 1
    elseif op_length == 4
      bin[lookup_array_op[win[1], win[2], win[3], win[4]]] = 1
    end

    @. p_count += bin
  end

  # Return tuple with relative frequencies and counts
  return ([p_count ./ length(dindex_ranges)], [p_count])

end

# Function to only count the number of ordinal patterns in bins for two time series  
function count_mv_op(tsx, tsy; op_length::Int=3, d=1)

  # Assert that time series have the same length
  @assert length(tsx) == length(tsy) "The time series must have the same length"

  # Assert that 2 <= op_length <= 4
  @assert 2 <= op_length <= 4 "This function is only implemented for pattern lengths of 2, 3 and 4"

  # Create vector with unit range for indexing 
  dindex_ranges = compute_dindex_op(tsx, op_length=op_length, d=d)

  # Compute lookup array and number of ops
  lookup_array_op = compute_lookup_array_op(op_length=op_length)

  # Vectors to store counts for op of x, y and reversed y
  op_length! = factorial(op_length)
  count_x = zeros(Int, op_length!)
  count_y = zeros(Int, op_length!)
  count_yrev = zeros(Int, op_length!)

  # Vectors to store counts for equal and non-equal op
  count_eq = zeros(Int, op_length!)
  count_neq = zeros(Int, op_length!)

  # Vectors for storing op match 
  bin_x = Vector{Int}(undef, op_length!)
  bin_y = Vector{Int}(undef, op_length!)

  # Vectors for storing ordered sequence
  win_x = Vector{Int}(undef, op_length)
  win_y = Vector{Int}(undef, op_length)

  pattern_seq_tsx = Vector{Int}(undef, length(dindex_ranges))
  pattern_seq_tsy = Vector{Int}(undef, length(dindex_ranges))

  for (i, j) in enumerate(dindex_ranges)
    seq_x = view(tsx, j)
    seq_y = view(tsy, j)
    fill!(bin_x, 0)
    fill!(bin_y, 0)

    order_vec!(seq_x, win_x)
    order_vec!(seq_y, win_y)

    if op_length == 2
      index_x = lookup_array_op[win_x[1], win_x[2]]
      index_y = lookup_array_op[win_y[1], win_y[2]]
    elseif op_length == 3
      index_x = lookup_array_op[win_x[1], win_x[2], win_x[3]]
      index_y = lookup_array_op[win_y[1], win_y[2], win_y[3]]
    elseif op_length == 4
      index_x = lookup_array_op[win_x[1], win_x[2], win_x[3], win_x[4]]
      index_y = lookup_array_op[win_y[1], win_y[2], win_y[3], win_y[4]]
    end

    pattern_seq_tsx[i] = index_x
    pattern_seq_tsy[i] = index_y

    bin_x[index_x] = 1
    bin_y[index_y] = 1

    @. count_x += bin_x
    @. count_y += bin_y
    @. count_eq = count_eq + bin_x * bin_y

    # Reverse tsy to account for negative dependence
    reverse!(win_y)
    fill!(bin_y, 0)

    if op_length == 2
      bin_y[lookup_array_op[win_y[1], win_y[2]]] = 1
    elseif op_length == 3
      bin_y[lookup_array_op[win_y[1], win_y[2], win_y[3]]] = 1
    elseif op_length == 4
      bin_y[lookup_array_op[win_y[1], win_y[2], win_y[3], win_y[4]]] = 1
    end

    @. count_yrev += bin_y
    @. count_neq = count_neq + bin_x * bin_y

  end

  #end

  # Create return array
  return_array = (count_x, count_y, count_yrev, count_eq, count_neq, pattern_seq_tsx, pattern_seq_tsy)

  return return_array

end

function dependence_op(tsx, tsy; op_length::Int=3, d=1)

  @assert length(tsx) == length(tsy) "The time series must have the same length"

  results_count = count_mv_op(tsx, tsy; op_length=op_length, d=d)

  count_x = results_count[1] # all pattern counts for x
  count_y = results_count[2] # all pattern counts for y
  count_yrev = results_count[3] # all pattern counts for y reversed
  count_eq = results_count[4] # all pattern counts for equal patterns
  count_neq = results_count[5] # all pattern counts for non-equal patterns
  pattern_seq_tsx = results_count[6]
  pattern_seq_tsy = results_count[7]

  # Convert count matrices to relative frequencies
  n = length(pattern_seq_tsx)
  p_x = count_x ./ n
  p_y = count_y ./ n
  p_yrev = count_yrev ./ n

  n_same = sum(count_eq)
  n_neq = sum(count_neq)

  # Same notation as Schnurr & Dehling (2017), p. 707 
  p = n_same / n
  q = sum(p_x .* p_y)
  r = n_neq / n
  s = sum(p_x .* p_yrev)

  α = p - q
  β = r - s

  cor_pos = α / (1 - q)
  cor_neg = β / (1 - s)
  cor_standard = maximum([cor_pos, 0]) - maximum([cor_neg, 0])

  return (cor_standard, pattern_seq_tsx, pattern_seq_tsy)

end

# Kernel function that is used for the changepoint detection
function kernel_change(x)
  return maximum([0, 1 - abs(x)])
end

# Weight function that is used for the changepoint detection
# Based on https://github.com/cran/ordinalpattern/blob/17b24cfe203893c3ceb41e867de8021760fea1e4/R/Pattern.R#L175 
function weightfun(maxdif, x)
  return return ((maxdif - x) / maxdif)
end

function changepoint_op(tsx, tsy; conf_level=0.95, weight=true, bn=log(length(tsx)), op_length::Int=3, d=1)

  # Check whether op_length is 2, 3, or 4
  @assert 2 <= op_length <= 4 "This function is only implemented for pattern lengths of 2, 3 and 4"

  # Get possible ranks
  rank_pattern = get_ranks_op(; op_length=op_length)

  # Defining standard weight function
  # Based on https://github.com/cran/ordinalpattern/blob/17b24cfe203893c3ceb41e867de8021760fea1e4/R/Pattern.R#L173
  if (weight == true)
    maxdif = floor(op_length / 2) * (floor(op_length / 2) + 1) + floor((op_length - 1) / 2) * (floor((op_length - 1) / 2) + 1)
  end

  results_count = count_mv_op(tsx, tsy; op_length=op_length, d=d)
  pattern_x_index::Vector{Int64} = results_count[6]
  pattern_index_y::Vector{Int64} = results_count[7]

  # Pre-allocate vector for L1 norm
  L1_vec = Vector{Int}(undef, length(pattern_x_index))
  x_minus_y = Vector{Int}(undef, op_length) # Vector to save in-place subtraction  

  # Loop to compute L1 norm for each pattern
  for i in axes(pattern_x_index, 1)

    # Get index for pattern
    ind_x = pattern_x_index[i]
    ind_y = pattern_index_y[i]

    # Extract rank pattern
    @views pattern_x = rank_pattern[ind_x, :]
    @views pattern_y = rank_pattern[ind_y, :]

    # Compute absolute vector differences 
    for j in 1:op_length
      x_minus_y[j] = abs(pattern_x[j] - pattern_y[j])
    end

    # Compute L1 norm
    L1_vec[i] = sum(x_minus_y)

  end

  # Check whether to use weight function
  if weight == true
    obs = broadcast(weightfun, maxdif, L1_vec)
  else
    obs = L1_vec .== 0
  end

  # Calculation of long-run-variance
  n = length(obs)
  weightv = kernel_change.(collect(0:floor(bn)) ./ bn)
  acfv = StatsBase.autocov(obs, 0:floor(Int, bn), demean=true)
  sigma = acfv[1] + 2 * sum(acfv[2:floor(Int, bn)])

  # Calculation of Cusum statistic
  Tn = 1 / sqrt(n) * abs.(cumsum(obs .- mean(obs))) ./ sqrt(sigma)
  Tn_max = findmax(Tn)
  changepoint = Tn_max[2]
  Tnmax = Tn_max[1]
  p_value = 1 - cdf(Kolmogorov(), Tnmax) # pKS2(Tnmax)
  conf_iv = (-1, 1) .* quantile(Kolmogorov(), conf_level)

  return (Tnmax, changepoint, p_value, conf_iv)

end
