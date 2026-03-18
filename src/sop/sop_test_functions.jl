function qup22_sop_value(refinement::Bool, alpha)


  ev = [2 / 5, 16 / 45]
  quantile(GeneralizedChisq(ev, ones(length(ev)), zeros(length(ev)), 0.0, 0.0), 1 - alpha)

end

function qup22_sop_value(refinement::RotationType, alpha)

  ev = [1 / 5, 8 / 45, 13 / 90, 2 / 15]
  quantile(GeneralizedChisq(ev, ones(length(ev)), zeros(length(ev)), 0.0, 0.0), 1 - alpha)

end

function qup22_sop_value(refinement::DirectionType, alpha)


  ev = [4 / 15, 1 / 5, 8 / 45, 19 / 630]
  quantile(GeneralizedChisq(ev, [1, 1, 2, 1], zeros(length(ev)), 0.0, 0.0), 1 - alpha)

end

function qup22_sop_value(refinement::DiagonalType, alpha)


  ev = [1 / 5, 8 / 45, 19 / 126, 4 / 45]
  quantile(GeneralizedChisq(ev, ones(length(ev)), zeros(length(ev)), 0.0, 0.0), 1 - alpha)

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

# --- 3. Multiple Dispatch Implementation of crit_val_sop() ---

# ==========================================================================
# PART A: NO REFINEMENT (chart::ChartMetric, ::NoRefinement)
# The dispatch logic is split based on the calculation method:
# 1. Tau/Kappa metrics (need the 'approximate' flag)
# 2. Information metrics (fixed critical values, ignore 'approximate')
# ==========================================================================

function crit_val_sop(M, N, alpha, d1::Int, d2::Int, ::TauHat, ::Bool=false)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 9 + 1 / 45 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

function crit_val_sop(M, N, alpha, d1::Int, d2::Int, ::KappaHat, ::Bool=false)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 3 + 1 / 9 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

function crit_val_sop(M, N, alpha, d1::Int, d2::Int, ::TauTilde, ::Bool=false)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 9 + 2 / 45 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

function crit_val_sop(M, N, alpha, d1::Int, d2::Int, ::KappaTilde, ::Bool=false)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 3 + 2 / 45 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

# A2. Dispatch for Entropy metrics
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::Bool=false
)

  m = M - d1
  n = N - d2

  # Note: Original logic simplified to use the ternary operator structure
  # crit_const = ifelse(alpha == 0.1, 3.487299,
  # ifelse(alpha == 0.05, 2.265401,
  # 1.740201)) # alpha == 0.01

  return qup22_sop_value(refinement, alpha) / (m * n)
end

# A2. Dispatch for Refined metrics
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::Union{RotationType,DirectionType,DiagonalType}
)

  m = M - d1
  n = N - d2

  return qup22_sop_value(refinement, alpha) / (m * n)

  # return ifelse(alpha == 0.1, 2.210104 / (m * n), ifelse(alpha == 0.05, 1.566739 / (m * n), 1.279915 / (m * n)))
end

# # A2. Dispatch for DirectionType refinement
# function crit_val_sop(
#   M, N, alpha, d1::Int, d2::Int,
#   chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
#   refinement::DirectionType
# )

#   m = M - d1
#   n = N - d2

#   return ifelse(alpha == 0.1, 2.813519 / (m * n), ifelse(alpha == 0.05, 1.999264 / (m * n), 1.637740 / (m * n)))
# end

# # A2. Dispatch for DiagonalType refinement
# function crit_val_sop(
#   M, N, alpha, d1::Int, d2::Int,
#   chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
#   refinement::DiagonalType
# )

#   m = M - d1
#   n = N - d2

#   return ifelse(alpha == 0.1, 2.133017 / (m * n), ifelse(alpha == 0.05, 1.497222 / (m * n), 1.216170 / (m * n)))
# end



# ---- User-facing wrapper ----
function test_sop(
  data, alpha, d1::Int, d2::Int;
  chart_choice,
  refinement::Union{Bool,RotationType,DirectionType,DiagonalType}=false,
  add_noise::Bool=false
)
  return test_sop(data, alpha, d1, d2, chart_choice; refinement=refinement, add_noise=add_noise)
end

# ---- Internal Method 1: Tau/Kappa - two-sided test, no rescaling ----
function test_sop(
  data, alpha, d1::Int, d2::Int,
  chart_choice::Union{TauHat,KappaHat,TauTilde,KappaTilde};
  refinement::Bool=false,
  add_noise::Bool=false
)
  M = size(data, 1)
  N = size(data, 2)

  # Compute critical value
  crit_val = crit_val_sop(M, N, alpha, d1, d2, chart_choice, refinement)

  # Compute test statistic (no rescaling needed)
  test_stat = stat_sop(data, d1, d2; chart_choice=chart_choice, refinement=refinement, add_noise=add_noise)[1]

  # Two-sided test
  return (test_stat, crit_val, abs(test_stat) > crit_val)
end

# ---- Internal Method 2: Entropy - one-sided test, rescaling needed ----
function test_sop(
  data, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise};
  refinement::Union{Bool,RotationType,DirectionType,DiagonalType}=false,
  add_noise::Bool=false
)
  M = size(data, 1)
  N = size(data, 2)
  m = M - d1
  n = N - d2

  # Compute critical value
  crit_val = crit_val_sop(M, N, alpha, d1, d2, chart_choice, refinement)

  # Compute p_vec, raw statistic, rescale, multiply by m*n
  raw = stat_sop(data, d1, d2; chart_choice=chart_choice, refinement=refinement, add_noise=add_noise)
  val = raw[1]
  q = length(raw[2])
  test_stat = m * n * rescale_sop(val, q, chart_choice)

  # One-sided test
  return (test_stat, crit_val, test_stat > crit_val)
end



# # Dispatch on no refinement
# function crit_val_sop(
#   M, N, alpha, d1::Int, d2::Int,
#   chart_choice::Union{TauHat,KappaHat,TauTilde,KappaTilde},
#   refinement::Bool=false
# )

#   m = M - d1
#   n = N - d2

#   # if approximate
#   #   # --- Approximate calculation ---
#   #   if typeof(chart_choice) == TauHat
#   #     term = sqrt(4 / 15) / sqrt(m * n)
#   #   elseif typeof(chart_choice) == KappaHat
#   #     term = sqrt(7 / 9) / sqrt(m * n)
#   #   elseif typeof(chart_choice) == TauTilde
#   #     term = sqrt(4 / 15) / sqrt(m * n)
#   #   elseif typeof(chart_choice) == KappaTilde
#   #     term = sqrt(32 / 45) / sqrt(m * n)
#   #   end
#   # else

#   # --- No approximation (exact formula with correction term) ---
#   correction = 1 - 1 / (2 * m) - 1 / (2 * n)
#   if typeof(chart_choice) == TauHat
#     term = sqrt(2 / 9 + 1 / 45 * correction) / sqrt(m * n)
#   elseif typeof(chart_choice) == KappaHat
#     term = sqrt(2 / 3 + 1 / 9 * correction) / sqrt(m * n)
#   elseif typeof(chart_choice) == TauTilde
#     term = sqrt(2 / 9 + 2 / 45 * correction) / sqrt(m * n)
#   elseif typeof(chart_choice) == KappaTilde
#     term = sqrt(2 / 3 + 2 / 45 * correction) / sqrt(m * n)
#   end
#   #end

#   return quantile(Normal(0, 1), 1 - alpha / 2) * term
# end
