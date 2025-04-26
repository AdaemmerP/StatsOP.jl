
# Compute test SOP-BP-statistic for one picture
function stat_sop_bp(
  data::Union{SubArray,Array{T,2}}, w::Int; 
  chart_choice::Int=3, 
  refinement::Int=0,
  add_noise::Bool=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1)
) where {T<:Real}

  # Check input parameters
  @assert 1 <= chart_choice <= 7 "chart_choice must be between 1 and 7"
  if chart_choice in 1:4
    @assert refinement == 0 "refinement must be 0 for chart_choice 1-4"
  end
  
  # Pre-allocate
  if refinement == 0
    # classical approach
    p_hat = zeros(3)
  else
    # refined approach
    p_hat = zeros(6)
  end
  lookup_array_sop = compute_lookup_array_sop()
  sop = zeros(4)
  sop_freq = zeros(Int, 24) # factorial(4) = 24
  win = zeros(Int, 4)
  bp_stat = 0.0

  # Get image size and get all d1-d2-combinations  
  M_rows = size(data, 1)
  N_cols = size(data, 2)
  d1_d2_combinations = Iterators.product(1:w, 1:w)

  # Pre-allocate indexes to compute sum of frequencies
  index_sop = create_index_sop(refinement=refinement)
  #s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  #s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  #s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # Add noise?
  if add_noise
    data = data .+ rand(noise_dist, size(data, 1), size(data, 2))
  end

  for (d1, d2) in d1_d2_combinations

    m = M_rows - d1
    n = N_cols - d2

    # Compute sum of frequencies for each pattern group
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

    # Fill 'p_hat' with sop-frequencies and compute relative frequencies
    fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, index_sop) # s_1, s_2, s_3)

    # Compute test statistic
    stat = chart_stat_sop(p_hat, chart_choice)
    bp_stat += stat^2

    # Reset win, sop_freq and p_hat
    # fill!(win, 0)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)
  end

  return bp_stat

end

# Compute "SOP-EWMA-BP-Statistic" based on sequential images
function stat_sop_bp(
  data::Array{T,3},
  lam,
  w::Int;
  chart_choice=3,
  refinement::Int=0,
  add_noise=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1),
  stat_ic::Union{Float64,Vector{Float64}}=0.0,
  type_freq_init::Union{Float64,Array{Float64,3}}=1 / 3
) where {T<:Real}

  # Check input parameters
  @assert 1 <= chart_choice <= 7 "chart_choice must be between 1 and 7"
  if chart_choice in 1:4
    @assert refinement == 0 "refinement must be 0 for chart_choice 1-4"
  elseif chart_choice in 5:7
    @assert 1 <= refinement <= 3 "refinement must be 1-3 for chart_choices 5-7"
  end

  # Number of d1-d2 combinations
  length_d1d2 = length(Iterators.product(1:w, 1:w))

  # If 'type_freq_init' is a 3d Array: verify that the first and second dimensions are equal to 3 and 1, respectively
  if typeof(type_freq_init) == Array{Float64,3} &&
     size(type_freq_init, 1) != 3 &&
     size(type_freq_init, 2) != 1
    throw(ArgumentError("First dimension of 'type_freq_init' must be equal to 3 and second dimension must be equal to 1"))
  end

  # If 'type_freq_init' is a 3d Array: verify that the third dimension equals the number of d1-d2 combinations  
  if typeof(type_freq_init) == Array{Float64,3} &&
     size(type_freq_init, 3) != length_d1d2
    throw(ArgumentError("Third dimension of 'type_freq_init' must be equal to the number of d1-d2 combinations"))
  end

  # If 'stat_ic' is a vector: verify that the length is equal to the number of d1-d2 combinations
  if length(stat_ic) > 1 && length(stat_ic) != length_d1d2
    throw(ArgumentError("Length of 'stat_ic' must be equal to the number of d1-d2 combinations"))
  end

  # Pre-allocate for BP-computations
  if refinement == 0
    # classical approach
    p_ewma_all = zeros(3, 1, length_d1d2)
  else
    # refined approach
    p_ewma_all = zeros(6, 1, length_d1d2)
  end
  bp_stats_all = zeros(size(data, 3))
  p_ewma_all .= type_freq_init
  stat_ic_vec = zeros(length_d1d2)
  stat_ic_vec .= stat_ic

  # Add noise?
  if add_noise
    data = data .+ rand(noise_dist, size(data, 1), size(data, 2), size(data, 3))
  end

  # Compute p_array (parallelized)
  # rows     -> images 
  # columns  -> type-frequencies 
  # 3rd dims -> d1-d2 combinations
  p_array = compute_p_array_bp(data, w; chart_choice=chart_choice, refinement=refinement)

  # Pre-allocate vector for each d1-d2 statistic
  stat = similar(stat_ic_vec)

  # Loop over all images (can not be parallelized because EWMA is dependent)
  for i = axes(data, 3)

    # Iterate over d1-d2 combination (parallelized) 
    Threads.@threads for j in 1:length_d1d2

      # Apply EWMA to p-vectors
      @views @. p_ewma_all[:, :, j] = (1 - lam) * p_ewma_all[:, :, j] + lam * p_array[i, :, j]

      # Compute test statistic            
      @views stat[j] = chart_stat_sop(p_ewma_all[:, :, j], chart_choice)

    end

    # Compute BP-statistic for each image
    @. stat = (stat - stat_ic_vec)^2
    bp_stats_all[i] = sum(stat)

    # Reset vector with individual statistics to zero 
    fill!(stat, 0.0)

  end

  return bp_stats_all

end
