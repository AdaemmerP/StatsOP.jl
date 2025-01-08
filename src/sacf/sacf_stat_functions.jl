# ---------------------------------------------------------------------------#
# -- Full SACF matrix and particular SACF for particular delay-combination - # 
# ---------------------------------------------------------------------------#
# function sacf(data, cdata, covs, d1::Int, d2::Int)

#   cdata .= data .- mean(data)
#   M = size(cdata, 1)
#   N = size(cdata, 2)

#   # Loop to compute all sums of relevant products
#   for k in (0, d2) # 0:d2 -> we only need 0-0 and one particular combination
#     for l in (0, d1) 0:d1 -> we only need 0-0 and one particular combination
#       # for (k, l) in zip((0, d2), (0, d1)) 
#       for j in 1:(N-l)
#         for i in 1:(M-k)
#           covs[l+1, k+1] += cdata[i+k, j+l] * cdata[i, j]
#         end
#       end
#     end
#   end

#   # Normalize by the number of elements
#   covs .= covs ./ (M * N)

#   # Compute the SACF value
#   sacf_val = covs[1+d1, 1+d2] / covs[1, 1]

#   # Return the SACF value
#   return sacf_val
# end

# ---------------------------------------------------------------------------#
# --------------------------- SACF(1, 1)         --------------------------- #
# ---------------------------------------------------------------------------#

# function sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)

#   # sizes and demeaned data
#   M = size(data, 1)
#   N = size(data, 2)
#   x_bar = mean(data)
#   cdata .= data .- x_bar

#   # slices and multiplications
#   @views cx_t = cdata[2:M, 2:N]
#   @views cx_t1 = cdata[1:(M-1), 1:(N-1)]
#   cx_t_cx_t1 .= cx_t .* cx_t1
#   cdata_sq .= cdata .^ 2

#   # return ρ(1, 1)
#   if allequal(data)
#     return 0.0
#   else
#     return sum(cx_t_cx_t1) / (sum(cdata_sq))
#   end
# end


"""
    sacf(X_centered, d1::Int, d2::Int)

- `X_centered`: The centered (de-meaned) data matrix.  
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
"""
function sacf(X_centered, d1::Int, d2::Int)

  M = size(X_centered, 1)
  N = size(X_centered, 2)

  # Lag 0x0
  @views cov_00 = dot(X_centered[1:M, 1:N], X_centered[1:M, 1:N]) / (M * N)
  # Lag d1xd2
  @views cov_d1d2 = dot(X_centered[1:(M-d1), 1:(N-d2)], X_centered[(1+d1):M, (1+d2):N]) / (M * N)

  # Return the SACF value
  if allequal(X_centered)
    return 1.0
  else
    return cov_d1d2 / cov_00
  end
end

# Compute SACF for one picture
"""
    stat_sacf(data::Union{SubArray,Matrix{T}}, d1::Int, d2::Int) where {T<:Real}

Compute the spatial autocorrelation for a delay combination (d1, d2) for a single picture.
  
- `data`: The data matrix.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
"""
function stat_sacf(data::Union{SubArray,Array{T,2}}, d1::Int, d2::Int) where {T<:Real}

  # pre-allocate
  X_centered = data .- mean(data)

  return sacf(X_centered, d1, d2)

end

# Compute SACF-BP-Statistik for one picture
"""
    stat_sacf(
  data::Union{SubArray,Matrix{T}}, d1_vec::Vector{Int}, d2_vec::Vector{Int}
) where {T<:Real}


Compute the BP-spatial autocorrelation function (BP-SACF) for multiple delay combinations (d1, d2) for a single picture.

- `data`: The data matrix.
- `d1_vec::Vector{Int}`: The vector of first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The vector of second (column) delays for the spatial process.
"""
function stat_sacf_bp(
  data::Union{SubArray,Array{T,2}}, w::Int
) where {T<:Real}

  # Compute all relevant h1-h2 combinations
  # h1_h2_combinations = Iterators.product(-w:w, -w:w)
  set_1 = Iterators.product(1:w, 0:w) 
  set_2 = Iterators.product(-w:0, 1:w)
  h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))

  # pre-allocate
  X_centered = data .- mean(data)
  bp_stat = 0.0

  for (h1, h2) in h1_h2_combinations
    @show h1, h2
    bp_stat += 2 * sacf(X_centered, h1, h2)^2
  end

  # bp_stat -= 1.0
  # bp_stat = 2 * bp_stat

  return bp_stat

end

# Compute SACF for multiple images
"""
    stat_sacf(lam, data::Array{T,3}, d1::Int, d2::Int) where {T<:Real}

Compute the spatial autocorrelation function (SACF) for a delay combination (d1, d2) for multiple images.

- `lam`: The smoothing parameter for the SACF.
- `data`: The data matrix.
- `d1::Int`: The first (row) delay for the spatial process.  
"""
function stat_sacf(lam, data::Array{T,3}, d1::Int, d2::Int) where {T<:Real}

  # pre-allocate
  data_tmp = similar(data[:, :, 1])
  X_centered = zeros(size(data_tmp))
  rho_hat = 0.0
  sacf_vals = zeros(size(data, 3))

  # loop over images
  for i in axes(data, 3)
    data_tmp .= view(data, :, :, i)
    X_centered .= data_tmp .- mean(data_tmp)
    rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)
    sacf_vals[i] = rho_hat
  end

  return sacf_vals

end

# Compute B-Statistik for multiple images
"""
    stat_sacf(lam, data::Array{T,3}, d1_vec::Vector{Int}, d2_vec::Vector{Int}) where {T<:Real}

Compute the EWMA-BP-spatial autocorrelation function (EWMA-BP-SACF) for multiple images.

- `lam`: The smoothing parameter for the SACF.
- `data`: The data matrix.
- `d1_vec::Vector{Int}`: The vector of first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The vector of second (column) delays for the spatial process.
"""
function stat_sacf(
  lam, data::Array{T,3}, d1_vec::Vector{Int}, d2_vec::Vector{Int}
) where {T<:Real}

  # ensure that 0 is not included in the d1_vec and d2_vec
  if 0 in d1_vec || 0 in d2_vec
    throw(ArgumentError("0 should not be included in d1_vec or d2_vec"))
  end

  # Compute all d1-d2 combinations
  d1_d2_combinations = Iterators.product(d1_vec, d2_vec)

  # pre-allocate
  X_centered = zeros(size(data[:, :, 1]))
  rho_hat_all = zeros(length(d1_d2_combinations))
  bp_stats = zeros(size(data, 3))
  bp_stat = 0.0

  # compute sequential BP-statistic
  for i in axes(data, 3)

    X_centered .= view(data, :, :, i) .- mean(view(data, :, :, i))

    for (j, (d1, d2)) in enumerate(d1_d2_combinations)
      rho_hat_all[j] = (1 - lam) * rho_hat_all[j] + lam * sacf(X_centered, d1, d2)
      bp_stat += 2 * rho_hat_all[j]^2
    end

    bp_stats[i] = bp_stat

  end

  return bp_stats

end
