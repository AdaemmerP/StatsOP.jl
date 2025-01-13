# SACF-BP-statistic for one images
"""
    stat_sacf_bp(
  data::Union{SubArray,Matrix{T}}, d1_vec::Vector{Int}, d2_vec::Vector{Int}
) where {T<:Real}


Compute the BP-spatial autocorrelation function (BP-SACF) for multiple delay combinations (d1, d2) for a single picture.

- `data`: The data matrix.
- `d1_vec::Vector{Int}`: The vector of first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The vector of second (column) delays for the spatial process.
"""
function stat_sacf_bp(data::Union{SubArray,Array{T,2}}, w::Int) where {T<:Real}

  # Compute all relevant h1-h2 combinations
  set_1 = Iterators.product(1:w, 0:w)
  set_2 = Iterators.product(-w:0, 1:w)
  h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))

  # pre-allocate
  X_centered = data .- mean(data)
  bp_stat = 0.0

  for (h1, h2) in h1_h2_combinations
    bp_stat += 2 * sacf(X_centered, h1, h2)^2
  end

  return bp_stat

end

# EWMA SACF-BP-statistic for multiple images
"""
    stat_sacf_bp(lam, data::Array{T,3}, d1_vec::Vector{Int}, d2_vec::Vector{Int}) where {T<:Real}

Compute the EWMA-BP-spatial autocorrelation function (EWMA-BP-SACF) for multiple images.

- `lam`: The smoothing parameter for the SACF.
- `data`: The data matrix.
- `d1_vec::Vector{Int}`: The vector of first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The vector of second (column) delays for the spatial process.
"""
function stat_sacf_bp(data::Array{T,3}, lam, w::Int) where {T<:Real}

  # Compute all relevant h1-h2 combinations
  set_1 = Iterators.product(1:w, 0:w)
  set_2 = Iterators.product(-w:0, 1:w)
  h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))

  # pre-allocate
  X_centered = zeros(size(data[:, :, 1]))
  rho_hat_all = zeros(length(h1_h2_combinations))
  bp_stats = zeros(size(data, 3))

  # compute sequential BP-statistic
  for i in axes(data, 3)

    # Center the data
    X_centered .= view(data, :, :, i) .- mean(view(data, :, :, i))

    # Compute the BP-statistic       
    bp_stat = 0.0  # Initialize BP-sum
    for (j, (h1, h2)) in enumerate(h1_h2_combinations)
      rho_hat_all[j] = (1 - lam) * rho_hat_all[j] + lam * sacf(X_centered, h1, h2)
      bp_stat += 2 * rho_hat_all[j]^2
    end

    bp_stats[i] = bp_stat

  end

  return bp_stats

end

