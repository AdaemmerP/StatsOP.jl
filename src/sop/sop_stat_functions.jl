#===============================================

Multiple Dispatch for 'stat_sop()':
  1. data is only one picture -> data::Matrix{T}
  2. data is a three dimensional array -> data::Array{T, 3}

================================================#

# 1. Method to compute test statistic for one picture without refinement
"""
  function stat_sop(
    data::Union{SubArray,Array{T,2}}, d1::Int, d2::Int;    
    chart_choice=3, 
    add_noise::Bool=false,
    noise_dist::UnivariateDistribution=Uniform(0, 1)
) where {T<:Real}

Computes the test statistic for a single picture and chosen test statistic. 
`chart_coice` is an integer value for the chart choice. The options are 1-4.

  The input parameters are:

- `data::Union{SubArray,Array{T,2}}`: A 2D array of data.
- `d1::Int`: The delay value for the rows.
- `d2::Int`: The delay value for the columns.
- `chart_choice::Int`: The choice of chart. The options are 1-4.
- `add_noise::Bool`: A boolean value to add noise to the data.
- `noise_dist::UnivariateDistribution`: The distribution for the noise.  

# Examples
```julia-repl
data = rand(20, 20);

stat_sop(data, 2)
```
"""
function stat_sop(
  data::Union{SubArray,Array{T,2}},
  d1::Int, d2::Int;
  chart_choice,
  refinement::Union{Nothing,RefinedType}=nothing,
  add_noise::Bool=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1)
) where {T<:Real}

  # TODO Check input parameters


  # Pre-allocate  
  if isnothing(refinement) #&& chart_choice in 1:4
    p_hat = zeros(3) # classical approach
  else
    p_hat = zeros(6) # refined approach
  end

  lookup_array_sop = compute_lookup_array_sop()
  sop = zeros(4)
  win = zeros(Int, 4)
  sop_freq = zeros(Int, 24) # factorial(4)  

  # Compute m and n based on data
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement=refinement)

  # Add noise?
  if add_noise
    data = data .+ rand(noise_dist, size(data, 1), size(data, 2))
  end

  # Compute frequencies of sops    
  sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

  # Fill 'p_hat' with sop-frequencies and compute relative frequencies
  fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)

  # Compute test statistic
  stat = chart_stat_sop(p_hat, chart_choice)

  return stat
end

# 2. Method to compute test statistic for multiple pictures
"""
   stat_sop(
      data::Array{T,3}, 
      lam, 
      d1::Int, 
      d2::Int; 
      chart_choice=3, 
      add_noise::Bool=false,
      noise_dist::UnivariateDistribution=Uniform(0, 1),
      type_freq_init::Union{Float64,Array{Float64,2}}=1 / 3
) 

Computes the test statistics for a 3D array of data, a given lambda value, and a given chart choice. 

The input parameters are:

- `data::Array{Float64,3}`: A 3D array of data.
- `lam::Float64`: The lambda value for the EWMA.
- `d1::Int`: The delay value for the rows.
- `d2::Int`: The delay value for the columns.
- `chart_choice::Int`: The choice of chart. The options are 1-4.
- `add_noise::Bool`: A boolean value to add noise to the data.
- `noise_dist::UnivariateDistribution`: The distribution for the noise.
- `type_freq_init::Union{Float64,Array{Float64,2}}`: The initial type frequencies.
"""
function stat_sop(
  data::Array{T,3},
  lam,
  d1::Int,
  d2::Int;
  chart_choice=TauTilde(),
  refinement::Union{Nothing,RefinedType}=nothing,
  add_noise::Bool=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1),
  type_freq_init::Union{Float64,Array{Float64,2}}=1 / 3
) where {T<:Real}

  # TODO Check input parameters


  # Compute lookup cube
  lookup_array_sop = compute_lookup_array_sop()

  # Pre-allocate
  if refinement == nothing
    p_hat = zeros(3) # classical approach
  else
    p_hat = zeros(6) # refined approach
  end

  sop = zeros(4)
  p_ewma = zeros(3)
  p_ewma .= type_freq_init
  stats_all = zeros(size(data, 3))
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement=refinement)

  # Compute m and n based on data
  data_tmp = similar(data[:, :, 1])
  rand_tmp = similar(data_tmp)
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  for i = axes(data, 3)

    # add noise?
    if add_noise
      data_tmp .= view(data, :, :, i) .+ rand!(noise_dist, rand_tmp)
    else
      data_tmp .= view(data, :, :, i)
    end

    # Compute frequencies of sops    
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

    # Fill 'p_hat' with sop-frequencies and compute relative frequencies
    fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)

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
