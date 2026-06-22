# Quantile computation for the generalized chi-squared distribution
function qup3_op_value(alpha)

  ev = [(2 + sqrt(2)) / 12, 2 / 15, 1 / 10, (2 - sqrt(2)) / 12]
  quantile(GeneralizedChisq(ev, ones(length(ev)), zeros(length(ev)), 0.0, 0.0), 1 - alpha)

  # if alpha == 0.01
  #   return 2.267254
  # elseif alpha == 0.05
  #   return 1.484225
  # elseif alpha == 0.1
  #   return 1.162639
  # end

end

# 1.) Method for Shannon
function crit_val_op(::Shannon, m, n_patterns; alpha=0.05)
  if m == 2
    # H-chart (m=2)
    @assert m in (2, 3) "Wrong m value for Shannon chart."
    qup2 = quantile(Chisq(1), 1 - alpha) / 6
    return log(2) - qup2 / n_patterns
  elseif m == 3
    # H-chart (m=3)
    qup3 = qup3_op_value(alpha)
    return log(6) - 3 * qup3 / n_patterns
  else
    throw(ArgumentError("Unsupported m value for Shannon chart: $m. Use 2 or 3."))
  end
end

# 2.) Method for ShannonExtropy 
function crit_val_op(::ShannonExtropy, m, n_patterns; alpha=0.05)
  @assert m == 3 "ShannonExtropy test only supports m = 3."

  # Hex-chart (m=3)
  qup3 = qup3_op_value(alpha)
  return 5 * log(6 / 5) - 3 * qup3 / 5 / n_patterns
end

# 3.) Method for DistanceToWhiteNoise 
function crit_val_op(::DistanceToWhiteNoise, m, n_patterns; alpha=0.05)
  @assert m in (2, 3) "Wrong m value for DistanceToWhiteNoise chart."

  if m == 2
    # Δ-chart (m=2)
    qup2 = quantile(Chisq(1), 1 - alpha)
    return qup2 / n_patterns
  elseif m == 3
    # Δ-chart (m=3)
    qup3 = qup3_op_value(alpha)
    return qup3 / n_patterns
  end
end

# 4.) Method for UpDownBalance 
function crit_val_op(::UpDownBalance, m, n_patterns; alpha=0.05)
  @assert m in (2, 3) "Wrong m value for UpDownBalance chart."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # β-chart (The calculation is the same for m=2 and m=3)
  return z2 * sqrt(1 / 3 / n_patterns)
end

# 5.) Method for Persistence
function crit_val_op(::Persistence, m, n_patterns; alpha=0.05)
  @assert m == 3 "Persistence chart only supports m = 3."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # τ-chart (m=3)
  return z2 * sqrt(8 / 45 / n_patterns)
end

# 6.) Method for RotationalAsymmetry
function crit_val_op(::RotationalAsymmetry, m, n_patterns; alpha=0.05)
  @assert m == 3 "RotationalAsymmetry chart only supports m = 3."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # γ-chart (m=3)
  return z2 * sqrt(2 / 5 / n_patterns)
end

# 7.) Method for UpDownScaling
function crit_val_op(::UpDownScaling, m, n_patterns; alpha=0.05)
  @assert m == 3 "UpDownScaling chart only supports m = 3."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # δ-chart (m=3)
  return z2 * sqrt(2 / 3 / n_patterns)
end

# This helper function performs the calculations common to all chart types.
# It takes the specific chart type instance (e.g., ::Shannon) as an argument.
function _common_chart_calculations(ts, chart_type; m::Int=3, d::Int=1, alpha=0.05)

  # Number of patterns when d is integer      
  n_patterns = length(ts) - (m - 1) * d

  # Compute p vectors. We assume stat_op is dispatched on the chart_type.
  p_vec = stat_op(ts; chart_choice=chart_type, m=m, d=d)[2]

  # Compute test statistic and critical value. 
  # Assume chart_stat_op and crit_val_op are also dispatched on the type.
  test_stat = chart_stat_op(p_vec, chart_type)
  crit_val = crit_val_op(chart_type, m, n_patterns; alpha=alpha)

  return (test_stat, crit_val)
end


# --- 3. Result type for asymptotic test ---

struct OPTestResult{C}
  chart::C
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::OPTestResult)
  println(io, "OPTestResult")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

# --- 4. Asymptotic p-value computation ---

# Shared generalized chi-squared null (m=3 case for Δ, H, Hex)
const _gc_op = GeneralizedChisq(
  [(2 + sqrt(2)) / 12, 2 / 15, 1 / 10, (2 - sqrt(2)) / 12],
  ones(4), zeros(4), 0.0, 0.0
)

function _asymp_pval(chart, stat::Float64, n_pat::Int, m::Int)::Float64
  if chart isa Union{Persistence, UpDownBalance, RotationalAsymmetry, UpDownScaling}
    se = chart isa Persistence         ? sqrt(8 / 45 / n_pat) :
         chart isa UpDownBalance       ? sqrt(1 / 3  / n_pat) :
         chart isa RotationalAsymmetry ? sqrt(2 / 5  / n_pat) : sqrt(2 / 3 / n_pat)
    return 2.0 * (1.0 - cdf(Normal(), abs(stat) / se))
  elseif chart isa DistanceToWhiteNoise
    T = n_pat * stat
    return m == 2 ? 1.0 - cdf(Chisq(1), T) : 1.0 - cdf(_gc_op, T)
  elseif chart isa Shannon
    T = m == 2 ? 6 * n_pat * (log(2) - stat) : n_pat * (log(6) - stat) / 3
    return m == 2 ? 1.0 - cdf(Chisq(1), T) : 1.0 - cdf(_gc_op, T)
  else  # ShannonExtropy (m=3 only)
    return 1.0 - cdf(_gc_op, 5 * n_pat * (5 * log(6 / 5) - stat) / 3)
  end
end

# --- 5. Implementation of test_op() ---

# Dispatch on chart type to determine rejection direction
reject(::Union{Shannon,ShannonExtropy}, test_stat, crit_val) = test_stat < crit_val
reject(::DistanceToWhiteNoise, test_stat, crit_val) = test_stat > crit_val
reject(::Union{UpDownBalance,Persistence,RotationalAsymmetry,UpDownScaling}, test_stat, crit_val) = abs(test_stat) > crit_val

function test_op(ts; chart_choice, m::Int=3, d::Int=1, alpha=0.05)
  n_pat = length(ts) - (m - 1) * d
  test_stat, crit_val = _common_chart_calculations(ts, chart_choice; m=m, d=d, alpha=alpha)
  p_val = _asymp_pval(chart_choice, test_stat, n_pat, m)
  return OPTestResult(chart_choice, test_stat, crit_val, p_val, reject(chart_choice, test_stat, crit_val))
end








