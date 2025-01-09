
# Compute test SOP-BP-statistic for one picture
function stat_sop_bp(
  data::Union{SubArray,Array{T, 2}}, w::Int; chart_choice=3, add_noise::Bool=false
) where {T<:Real}

  # Pre-allocate
  lookup_array_sop = compute_lookup_array_sop()
  p_hat = zeros(3)
  sop = zeros(4)
  sop_freq = zeros(Int, 24) # factorial(4) = 24
  win = zeros(Int, 4)
  bp_stat = 0.0

  # Get image size and get all d1-d2-combinations  
  M_rows = size(data, 1)
  N_cols = size(data, 2)
  d1_d2_combinations = Iterators.product(1:w, 1:w)

  # Pre-allocate indexes to compute sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  # Add noise?
  if add_noise
    data .= data .+ rand(size(data, 1), size(data, 2))
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

    # Reset win, sop_freq and p_hat
    # fill!(win, 0)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)
  end

  return bp_stat

end

# Compute "SOP-EWMA-BP-Statstic" based on sequential images
function stat_sop_bp(
  data::Array{T,3}, lam, w::Int;
  chart_choice=3, add_noise=false, 
) where {T<:Real}

  # Compute 4 dimensional cube to lookup sops
  lookup_array_sop = compute_lookup_array_sop()
  p_hat = zeros(3)
  sop = zeros(4)

  # Pre-allocate for BP-computations
  d1_d2_combinations = Iterators.product(1:w, 1:w)
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
    for (j, (d1, d2)) in enumerate(d1_d2_combinations)

      m = M_rows - d1
      n = N_cols - d2

      # Compute frequencies of sops    
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

      # Fill 'p_hat' with sop-frequencies and compute relative frequencies
      fill_p_hat!(p_hat, chart_choice, sop_freq, m, n, s_1, s_2, s_3)

      # Apply EWMA to p-vectors
      @views p_ewma_all[:, :, j] .= (1 - lam) .* p_ewma_all[:, :, j] .+ lam .* p_hat

      # Compute test statistic            
      @views stat = chart_stat_sop(p_ewma_all[:, :, j], chart_choice)

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
