# Compute a lookup array for sorting permutations of 4 elements
# Use dictionary
function compute_lookup_array_sop()

  # Create ranks
  ranks = collect(permutations(1:4))

  # Convert to sortperm values
  perm_ranks = sortperm.(ranks)

  # Create lookup dictionary
  lookup_dict = Dict{Vector{Int},Int}()
  for (i, j) in enumerate(perm_ranks)
    lookup_dict[j] = i
  end

  return lookup_dict

end


#--- Compute absolute frequencies of sops
function sop_frequencies_lehmer!(
  m, n, d1, d2, data, sop, win, sop_freq, used
)

  # Loop through data to fill sop vector
  for j in 1:n
    for i in 1:m

      sop[1] = data[i, j]
      sop[2] = data[i, j+d2]
      sop[3] = data[i+d1, j]
      sop[4] = data[i+d1, j+d2]

      # Order 'sop' in-place and save results in 'win'
      sortperm!(win, sop)

      # Convert permutation to lehmer index
      index = perm_to_lehm_idx!(win, used)
      fill!(used, 0) # reset used

      # Add 1 to index position
      sop_freq[index] += 1
    end
  end

  return sop_freq

end


# Fill p_hat with the sum of frequencies and compute relative frequencies
function fill_p_hat_lehmer!(
  p_hat, chart_choice, refinement, sop_freq, m, n
)

  # For classical appraoch
  if isnothing(refinement)
    if typeof(chart_choice) == TauHat # previously chart_choice == 1
      # Relevant lehmer indices for TauHat
      s_1 = (1, 3, 8, 11, 14, 17, 22, 24)
      for i in s_1
        p_hat[1] += sop_freq[i]
      end

    elseif typeof(chart_choice) == KappaHat # previously chart_choice == 2
      # Relevant lehmer indices for KappaHat
      s_2 = (2, 4, 7, 12, 13, 18, 21, 23)
      s_3 = (5, 6, 9, 10, 15, 16, 19, 20)
      for (i, j) in zip(s_2, s_3)
        p_hat[2] += sop_freq[i]
        p_hat[3] += sop_freq[j]
      end

    elseif typeof(chart_choice) == TauTilde # previously chart_choice == 3
      # Relevant lehmer indices for TauTilde
      s_3 = (5, 6, 9, 10, 15, 16, 19, 20)
      for i in s_3
        p_hat[3] += sop_freq[i]
      end

    elseif typeof(chart_choice) == KappaTilde # previously chart_choice == 4
      # Relevant lehmer indices for indices 
      s_1 = (1, 3, 8, 11, 14, 17, 22, 24)
      s_2 = (2, 4, 7, 12, 13, 18, 21, 23)
      for (i, j) in zip(s_1, s_2)
        p_hat[1] += sop_freq[i]
        p_hat[2] += sop_freq[j]
      end

    elseif typeof(chart_choice) in (Shannon, ShannonExtropy, DistanceToWhiteNoise)
      # Relevant lehmer indices for Shannon, ShannonExtropy, DistanceToWhiteNoise
      s_1 = (1, 3, 8, 11, 14, 17, 22, 24)
      s_2 = (2, 4, 7, 12, 13, 18, 21, 23)
      s_3 = (5, 6, 9, 10, 15, 16, 19, 20)
      for (i, j, k) in zip(s_1, s_2, s_3)
        p_hat[1] += sop_freq[i]
        p_hat[2] += sop_freq[j]
        p_hat[3] += sop_freq[k]
      end
    end

    # For refined computations  
  elseif typeof(refinement) == RefinedType
    for (i, j, k, l, m, n) in zip(
      s_all[1], s_all[2], s_all[3], s_all[4], s_all[5], s_all[6]
    )
      p_hat[1] += sop_freq[i]
      p_hat[2] += sop_freq[j]
      p_hat[3] += sop_freq[k]
      p_hat[4] += sop_freq[l]
      p_hat[5] += sop_freq[m]
      p_hat[6] += sop_freq[n]
    end


  end

  # Compute relative frequencies
  p_hat ./= m * n

end


################################################################################
#                     Compute test stats using Lehmer                          #
################################################################################

function chart_stat_sop_lehmer(p_vec, ::TauHat)
  return p_vec[1] - 1.0 / 3.0
end

function chart_stat_sop_lehmer(p_vec, ::KappaHat)
  return p_vec[2] - p_vec[3]
end

function chart_stat_sop_lehmer(p_vec, ::TauTilde)
  return p_vec[3] - 1.0 / 3.0
end

function chart_stat_sop_lehmer(p_vec, ::KappaTilde)
  return p_vec[1] - p_vec[2]
end

function chart_stat_sop_lehmer(p_vec, ::Shannon)
  # H-chart: Equation (7), Weiss and Kim
  chart_val = 0.0
  q = length(p_vec)
  for i in axes(p_vec, 1)
    p_vec[i] > 0 && (chart_val -= p_vec[i] * log(p_vec[i])) # to avoid log(0)
  end
  # Re-scale
  return (-2) / q * (chart_val - log(q))
end

function chart_stat_sop_lehmer(p_vec, ::ShannonExtropy)
  # Hex-chart: Equation (7), Weiss and Kim
  chart_val = 0.0
  q = length(p_vec)
  for i in axes(p_vec, 1)
    p_vec[i] < 1 && (chart_val -= (1 - p_vec[i]) * log(1 - p_vec[i])) # to avoid log of negative value
  end
  # Re-scale
  return (-2) * (1 - 1 / q) * (chart_val - (q - 1) * log(q / (q - 1)))
end

function chart_stat_sop_lehmer(p_vec, ::DistanceToWhiteNoise)
  # Δ-chart: Equation (7), Weiss and Kim
  chart_val = 0.0
  q = length(p_vec)
  for i in axes(p_vec, 1)
    chart_val += (p_vec[i] - 1 / q)^2
  end
  return chart_val
end




