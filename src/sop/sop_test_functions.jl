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


# --- 3. Multiple Dispatch Implementation of crit_val_sop() ---

# ==========================================================================
# PART A: NO REFINEMENT (chart::ChartMetric, ::NoRefinement)
# The dispatch logic is split based on the calculation method:
# 1. Tau/Kappa metrics (need the 'approximate' flag)
# 2. Information metrics (fixed critical values, ignore 'approximate')
# ==========================================================================

"""
    crit_val_sop(M, N, alpha, d1, d2, chart_choice, refinement=false)

Compute the critical value for the asymptotic test based on spatial ordinal patterns
(SOPs); see [`test_sop`](@ref).

- `M::Int`: number of rows of the data matrix. The SOP matrix has `m = M - d1` rows.
- `N::Int`: number of columns of the data matrix. The SOP matrix has `n = N - d2` columns.
- `alpha::Float64`: significance level.
- `d1::Int`: row delay.
- `d2::Int`: column delay.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`, `Shannon()`, `ShannonExtropy()`,
  `DistanceToWhiteNoise()`.
- `refinement`: `false` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`
  (only for the entropy-type charts).

# Examples
```julia-repl
crit_val_sop(11, 11, 0.05, 1, 1, TauHat())
```
"""
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



# --- Result type for SOP asymptotic test ---

"""
    SOPTestResult

Result of the asymptotic test based on spatial ordinal patterns [`test_sop`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `stat::Float64`: value of the test statistic.
- `asymp_crit::Float64`: asymptotic critical value.
- `asymp_pval::Float64`: asymptotic p-value.
- `asymp_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
"""
struct SOPTestResult{C}
  chart::C
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::SOPTestResult)
  println(io, "SOPTestResult")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

# GenChisq null distributions for entropy-chart p-values
_gc_sop(::Bool)          = GeneralizedChisq([2/5, 16/45],              ones(2),      zeros(2), 0.0, 0.0)
_gc_sop(::RotationType)  = GeneralizedChisq([1/5, 8/45, 13/90, 2/15], ones(4),      zeros(4), 0.0, 0.0)
_gc_sop(::DirectionType) = GeneralizedChisq([4/15, 1/5, 8/45, 19/630],[1, 1, 2, 1], zeros(4), 0.0, 0.0)
_gc_sop(::DiagonalType)  = GeneralizedChisq([1/5, 8/45, 19/126, 4/45],ones(4),      zeros(4), 0.0, 0.0)

# Asymptotic p-value for any chart type.
# For Tau/Kappa (Normal): derive SE from crit_val.
# For entropy charts (GenChisq): m_pat * n_pat * test_stat ~ GenChisq under H₀.
function _sop_asymp_pval(chart_choice, test_stat, crit_val, refinement, alpha, m_pat, n_pat)
  if chart_choice isa Union{TauHat, KappaHat, TauTilde, KappaTilde}
    z_alpha = quantile(Normal(0, 1), 1 - alpha / 2)
    return 2.0 * (1.0 - cdf(Normal(), abs(test_stat) * z_alpha / crit_val))
  else
    return 1.0 - cdf(_gc_sop(refinement), m_pat * n_pat * test_stat)
  end
end

"""
    test_sop(data, alpha, d1, d2; chart_choice, refinement=false, add_noise=false)

Perform the asymptotic hypothesis test for spatial dependence based on spatial ordinal
patterns (SOPs) and return a [`SOPTestResult`](@ref) with the test statistic, the
asymptotic critical value, the p-value, and the reject decision.

- `data`: data matrix (spatial field).
- `alpha`: significance level.
- `d1::Int`: row delay.
- `d2::Int`: column delay.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()` (two-sided test), or `Shannon()`,
  `ShannonExtropy()`, `DistanceToWhiteNoise()` (one-sided, upper-tail test with
  rescaled statistic).
- `refinement`: `false` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`
  (only for the entropy-type charts).
- `add_noise::Bool=false`: add uniform noise to the data to break ties (recommended for
  discrete-valued data).
"""
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
  m_pat = M - d1
  n_pat = N - d2
  crit_val  = crit_val_sop(M, N, alpha, d1, d2, chart_choice, refinement)
  test_stat = stat_sop(data, d1, d2; chart_choice=chart_choice, refinement=refinement, add_noise=add_noise)[1]
  p_val     = _sop_asymp_pval(chart_choice, test_stat, crit_val, refinement, alpha, m_pat, n_pat)
  return SOPTestResult(chart_choice, test_stat, crit_val, p_val, abs(test_stat) > crit_val)
end

# ---- Internal Method 2: Entropy - one-sided (upper-tail) test, rescaling needed ----
function test_sop(
  data, alpha, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise};
  refinement::Union{Bool,RotationType,DirectionType,DiagonalType}=false,
  add_noise::Bool=false
)
  M = size(data, 1)
  N = size(data, 2)
  m_pat = M - d1
  n_pat = N - d2
  crit_val  = crit_val_sop(M, N, alpha, d1, d2, chart_choice, refinement)
  raw       = stat_sop(data, d1, d2; chart_choice=chart_choice, refinement=refinement, add_noise=add_noise)
  test_stat = m_pat * n_pat * rescale_sop(raw[1], length(raw[2]), chart_choice)
  p_val     = _sop_asymp_pval(chart_choice, test_stat, crit_val, refinement, alpha, m_pat, n_pat)
  return SOPTestResult(chart_choice, test_stat, crit_val, p_val, test_stat > crit_val)
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
