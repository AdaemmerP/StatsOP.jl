
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
# Checks alpha validity for refinement cases, common to all refined methods.
function _check_alpha(alpha)
  @assert alpha in (0.1, 0.05, 0.01) "alpha must be 0.1, 0.05 or 0.01 for refined types."
end

# --- 3. Multiple Dispatch Implementation of crit_val_sop() ---

# ==========================================================================
# PART A: NO REFINEMENT (chart::ChartMetric, ::NoRefinement)
# The dispatch logic is split based on the calculation method:
# 1. Tau/Kappa metrics (need the 'approximate' flag)
# 2. Information metrics (fixed critical values, ignore 'approximate')
# ==========================================================================

# Dispatch on no refinement
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int,
  chart_choice::Union{TauHat,KappaHat,TauTilde,KappaTilde},
  refinement::Bool=false;
  approximate::Bool=false
)

  m = M - d1
  n = N - d2

  if approximate
    # --- Approximate calculation ---
    if typeof(chart_choice) == TauHat
      term = sqrt(4 / 15) / sqrt(m * n)
    elseif typeof(chart_choice) == KappaHat
      term = sqrt(7 / 9) / sqrt(m * n)
    elseif typeof(chart_choice) == TauTilde
      term = sqrt(4 / 15) / sqrt(m * n)
    elseif typeof(chart_choice) == KappaTilde
      term = sqrt(32 / 45) / sqrt(m * n)
    end
  else

    # --- No approximation (exact formula with correction term) ---
    correction = 1 - 1 / (2 * m) - 1 / (2 * n)
    if typeof(chart_choice) == TauHat
      term = sqrt(2 / 9 + 1 / 45 * correction) / sqrt(m * n)
    elseif typeof(chart_choice) == KappaHat
      term = sqrt(2 / 3 + 1 / 9 * correction) / sqrt(m * n)
    elseif typeof(chart_choice) == TauTilde
      term = sqrt(2 / 9 + 2 / 45 * correction) / sqrt(m * n)
    elseif typeof(chart_choice) == KappaTilde
      term = sqrt(2 / 3 + 2 / 45 * correction) / sqrt(m * n)
    end
  end

  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

# A2. Dispatch for Entropy metrics
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::Bool=false;
  approximate::Bool=false # Note: approximate is ignored here but kept for signature consistency
)

  m = M - d1
  n = N - d2

  # Note: Original logic simplified to use the ternary operator structure
  crit_const = ifelse(alpha == 0.1, 3.487299,
    ifelse(alpha == 0.05, 2.265401,
      1.740201)) # alpha == 0.01

  return crit_const / (m * n)
end

# A2. Dispatch for Refined metrics
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::RotationType;
  approximate::Bool=false
)

  m = M - d1
  n = N - d2

  return ifelse(alpha == 0.1, 2.210104 / (m * n), ifelse(alpha == 0.05, 1.566739 / (m * n), 1.279915 / (m * n)))
end

# A2. Dispatch for DirectionType refinement
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::DirectionType;
  approximate::Bool=false
)

  m = M - d1
  n = N - d2

  return ifelse(alpha == 0.1, 2.813519 / (m * n), ifelse(alpha == 0.05, 1.999264 / (m * n), 1.637740 / (m * n)))
end

# A2. Dispatch for DiagonalType refinement
function crit_val_sop(
  M, N, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::DiagonalType;
  approximate::Bool=false
)

  m = M - d1
  n = N - d2

  return ifelse(alpha == 0.1, 2.133017 / (m * n), ifelse(alpha == 0.05, 1.497222 / (m * n), 1.216170 / (m * n)))
end



# Function for hypothesis testing
function test_sop(
  data, alpha, d1::Int, d2::Int; chart_choice, refinement::Bool=false, add_noise::Bool=false, approximate::Bool=false
)

  # sizes
  M = size(data, 1)
  N = size(data, 2)

  # compute critical value 
  crit_val = crit_val_sop(
    M, N, alpha, d1, d2, chart_choice, refinement; approximate=approximate
  )

  # compute test statistic
  test_stat = stat_sop(
    data,
    d1,
    d2;
    chart_choice,
    refinement,
    add_noise
  )

  # return test result
  return (test_stat, crit_val, abs(test_stat) > crit_val)

end