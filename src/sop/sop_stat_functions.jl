"""
    chart_stat_sop(p_ewma, chart_choice)

Compute the the test statistic for spatial ordinal patterns. The first input is 
a vector with three values, based on SOP counts. The second input is the chart.     

"""
function chart_stat_sop(p_ewma, chart_choice)
  if chart_choice == 1
    chart_val = p_ewma[1] - 1.0 / 3.0
  elseif chart_choice == 2
    chart_val = p_ewma[2] - p_ewma[3]
  elseif chart_choice == 3
    chart_val = p_ewma[3] - 1.0 / 3.0
  elseif chart_choice == 4
    chart_val = p_ewma[1] - p_ewma[2]
  else
    println("Wrong number for test statistic.")
  end

  return chart_val
end

#===============================================

Multiple Dispatch for 'stat_sop()':
  1. data is only one picture -> data::Matrix{T}
  2. data is a three dimensional array -> data::Array{T, 3}

================================================#

# 1. Method to compute test statistic for one picture
"""
  function stat_sop(
    data::Union{SubArray,Array{T,2}}, d1::Int, d2::Int;
    chart_choice=3, add_noise::Bool=false
) where {T<:Real}

Computes the test statistic for a single picture and chosen test statistic. 
`chart_coice` is an integer value for the chart choice. The options are 1-4.

# Examples
```julia-repl
data = rand(20, 20);

stat_sop(data, 2)
```
"""
function stat_sop(
  data::Union{SubArray,Array{T,2}}, d1::Int=1, d2::Int=1;
  chart_choice=3, add_noise::Bool=false
) where {T<:Real}

  # Compute 4 dimensional cube to lookup sops
  lookup_array_sop = compute_lookup_array_sop()
  p_hat = zeros(3)
  sop = zeros(4)
  win = zeros(Int, 4)
  sop_freq = zeros(Int, 24) # factorial(4)  

  # Compute m and n based on data
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # Add noise?
  if add_noise
    data .= data + rand(size(data, 1), size(data, 2))
  end

  # Compute frequencies of sops    
  sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

  # Fill 'p_hat' with sop-frequencies and compute relative frequencies
  fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

  # Compute test statistic
  stat = chart_stat_sop(p_hat, chart_choice)

  return stat
end

# 2. Method to compute test statistic for multiple pictures
"""
   stat_sop(data::Array{Float64, 3}, add_noise::Bool, lam::Float64, chart_choice::Int)

Computes the test statistic for a 3D array of data, a given lambda value, and a given chart choice. 
The input parameters are:

- `data::Array{Float64,3}`: A 3D array of data.
- `add_noise::Bool`: A boolean value whether to add noise to the data. This is 
necessary when the matrices consist of count data.
- `lam::Float64`: A scalar value for lambda.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4.

# Examples
```julia-repl
data = rand(20, 20, 10);
lam = 0.1;
chart_choice = 2;

stat_sop(data, false, lam, chart_choice)
```
"""
function stat_sop(
  lam, data::Array{T,3}, d1::Int=1, d2::Int=1; chart_choice=3, add_noise::Bool=false
) where {T<:Real}

  # Compute lookup cube
  lookup_array_sop = compute_lookup_array_sop()

  # Pre-allocate
  p_hat = zeros(3)
  sop = zeros(4)
  p_ewma = repeat([1.0 / 3.0], 3)
  stats_all = zeros(size(data, 3))
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # Compute m and n based on data
  data_tmp = similar(data[:, :, 1])
  rand_tmp = similar(data_tmp)
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  for i = axes(data, 3)

    # add noise?
    if add_noise
      data_tmp .= view(data, :, :, i) .+ rand!(rand_tmp)
    else
      data_tmp .= view(data, :, :, i)
    end

    # Compute frequencies of sops    
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

    # Fill 'p_hat' with sop-frequencies and compute relative frequencies
    fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

    # Compute test statistic
    @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

    stat_tmp = chart_stat_sop(p_ewma, chart_choice)

    # Save temporary test statistic
    stats_all[i] = stat_tmp

    # Reset win and sop_freq
    fill!(win, 0)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)
  end

  return stats_all

end

# Compute test statistics for one picture and when deleays are vectors
function stat_sop(
  data::Union{SubArray,Array{T, 2}}, d1_vec::Vector{Int}, d2_vec::Vector{Int};
  chart_choice=3, add_noise::Bool=false
) where {T<:Real}

  # Pre-allocate
  lookup_array_sop = compute_lookup_array_sop()
  p_hat = zeros(3)
  sop = zeros(4)
  sop_freq = zeros(Int, 24) # factorial(4) = 24
  win = zeros(Int, 4)
  bp_stat = 0.0

  # Compute all combinations of d1 and d2
  M_rows = size(data, 1)
  N_cols = size(data, 2)
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # Add noise?
  if add_noise
    data .= data + rand(size(data, 1), size(data, 2))
  end

  for (d1, d2) in d1_d2_combinations

    m = M_rows - d1
    n = N_cols - d2

    # Compute sum of frequencies for each pattern group
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

    # Fill 'p_hat' with sop-frequencies and compute relative frequencies
    fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

    # Compute test statistic
    stat = chart_stat_sop(p_hat, chart_choice)
    bp_stat += stat^2
  end
  return bp_stat
end

# Compute test statistics for multiple pictures and when delays are vectors
function stat_sop(
  lam, data::Array{T,3}, d1_vec::Vector{Int}, d2_vec::Vector{Int};
  chart_choice=3, add_noise=false
) where {T<:Real}

  # Compute 4 dimensional cube to lookup sops
  lookup_array_sop = compute_lookup_array_sop()
  p_hat = zeros(3)
  sop = zeros(4)
  p_ewma = repeat([1.0 / 3.0], 3)

  # Pre-allocate for BP-computations
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)
  p_ewma_all = zeros(3, 1, length(d1_d2_combinations))
  p_ewma_all .= 1.0 / 3.0
  bp_stat = 0.0
  bp_stats_all = zeros(size(data, 3))

  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)
  M_rows = size(data, 1)
  N_cols = size(data, 2)
  data_tmp = similar(data[:, :, 1])
  rand_tmp = rand(M_rows, N_cols)

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  for i = axes(data, 3)

    # Add noise?
    if add_noise
      data_tmp .= view(data, :, :, i) .+ rand!(rand_tmp)
    else
      data_tmp .= view(data, :, :, i)
    end

    # -------------------------------------------------------------------------#
    # ----------------     Loop for BP-Statistik                     ----------#
    # -------------------------------------------------------------------------#
    for (i, (d1, d2)) in enumerate(d1_d2_combinations)

      m = M_rows - d1
      n = N_cols - d2

      # Compute frequencies of sops    
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

      # Fill 'p_hat' with sop-frequencies and compute relative frequencies
      fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

      # Apply EWMA to p-vectors
      @views p_ewma_all[:, :, i] .= (1 - lam) .* p_ewma_all[:, :, i] .+ lam .* p_hat

      # Compute test statistic            
      @views stat = chart_stat_sop(p_ewma_all[:, :, i], chart_choice)

      # Save temporary test statistic
      bp_stat += stat^2

      # Reset win, sop_freq and p_hat
      fill!(win, 0)
      fill!(sop_freq, 0)
      fill!(p_hat, 0)
    end
    
    bp_stats_all[i] = bp_stat
    
  end

  return bp_stats_all

end

"""
  crit_val_sop(m, n, alpha, chart_choice, approximate::Bool)

Computes the critical value for the SOP test. Also allows the approximation of 
  the critical value. The input parameters are:

- `m::Int64`: The number of rows in the sop-matrix. Note that the data matrix has 
dimensions `M = m + d₁`, where `d₁` denotes the row delay.
- `n::Int64`: The number of columns in the sop-matrix. Note that the data matrix 
has dimensions `N = n + d₂`, where `d₂` denotes the column delay.
- `alpha::Float64`: The significance level.
- `chart_choice::Int64`: The choice of chart. 
- `approximate::Bool`: If `true`, the approximate critical value is computed. 
If `false`, the exact critical value is computed.

# Examples
```julia-repl
# compute approximate critical value for chart 1 
crit_val_sop(10, 10, 0.05, 1, true)
```
"""
function crit_val_sop(m, n, alpha; chart_choice, approximate::Bool=false)

  if approximate
    if chart_choice == 1
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 9 + 1 / 45
      )
    elseif chart_choice == 2
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 3 + 1 / 9
      )
    elseif chart_choice == 3
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 9 + 2 / 45
      )
    elseif chart_choice == 4
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 3 + 2 / 45
      )
    end

  else

    if chart_choice == 1
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 9 + 1 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n))
      )
    elseif chart_choice == 2
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 3 + 1 / 9 * (1 - 1 / (2 * m) - 1 / (2 * n))
      )
    elseif chart_choice == 3
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 9 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n))
      )
    elseif chart_choice == 4
      quantile(
        Normal(0, 1),
        1 - alpha / 2) * sqrt(2 / 3 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n))
      )
    end
  end
end

"""
  crit_val_sacf(M, N, alpha)

Computes the critical value for the SACF of lag 1. The input parameters are:

- `M::Int64`: The number of rows in the data matrix.
- `N::Int64`: The number of columns in the data matrix.
- `alpha::Float64`: The significance level.

# Examples
```julia-repl
# compute critical value
crit_val_sacf(11, 11, 0.05)
```
"""
function crit_val_sacf(M, N, alpha)
  quantile(Normal(0, 1), 1 - alpha / 2) / sqrt(M * N)
end



# """
#   crit_val_sop(m, n, alpha, chart_choice, approximate::Bool)

# Computes the exact critical value for the SOP test. The input parameters are:

# - `m::Int64`: The number of rows in the sop-matrix. Note that the data matrix has dimensions `M = m + 1`.
# - `n::Int64`: The number of columns in the sop-matrix. Note that the data matrix has dimensions `N = n + 1`.
# - `alpha::Float64`: The significance level.
# - `chart_choice::Int64`: The choice of chart. The options are:

# # Examples
# ```julia-repl
# # compute approximate critical value for chart 1 
# crit_val_sop(10, 10, 0.05, 1, true)
# ```
# """
# function crit_val_sop(m, n, alpha, chart_choice)

#   if chart_choice == 1
#     quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 1 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
#   elseif chart_choice == 2
#     quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 1 / 9 * (1 - 1 / (2 * m) - 1 / (2 * n)))
#   elseif chart_choice == 3
#     quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 9 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
#   elseif chart_choice == 4
#     quantile(Normal(0, 1), 1 - alpha / 2) * sqrt(2 / 3 + 2 / 45 * (1 - 1 / (2 * m) - 1 / (2 * n)))
#   end

# end
