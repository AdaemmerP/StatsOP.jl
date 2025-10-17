# Common qup3 calculation for m=3
function qup3_value(alpha)
  @assert alpha in (0.01, 0.05, 0.1) "Wrong alpha level. Choose 0.01, 0.05 or 0.1."
  if alpha == 0.01
    return 2.267254
  elseif alpha == 0.05
    return 1.484225
  elseif alpha == 0.1
    return 1.162639
  end
end

# 1.) Method for Shannon
function crit_val_op(::Shannon, m, n_patterns; alpha=0.05)
  if m == 2
    # H-chart (m=2)
    @assert m in (2, 3) "Wrong m value for Shannon chart."
    qup2 = quantile(Chisq(1), 1 - alpha) / 6
    return log(2) - qup2(alpha) / n_patterns
  elseif m == 3
    # H-chart (m=3)
    qup3 = qup3_value(alpha)
    return log(6) - 3 * qup3 / n_patterns
  else
    throw(ArgumentError("Unsupported m value for Shannon chart: $m. Use 2 or 3."))
  end
end

# 2.) Method for ShannonExtropy 
function crit_val_op(::ShannonExtropy, m, n_patterns; alpha=0.05)
  @assert m == 3 "ShannonExtropy test only supports m = 3."

  # Hex-chart (m=3)
  qup3 = qup3_value(alpha)
  return 5 * log(6 / 5) - 3 * qup3 / 5 / n_patterns
end

# 3.) Method for DistanceToWhiteNoise 
function crit_val_op(::DistanceToWhiteNoise, m, n_patterns; alpha=0.05)
  @assert m in (2, 3) "Wrong m value for DistanceToWhiteNoise chart."

  if m == 2
    # Δ-chart (m=2)
    qup2 = quantile(Chisq(1), 1 - alpha)
    return qup2(alpha) / n_patterns
  elseif m == 3
    # Δ-chart (m=3)
    qup3 = qup3_value(alpha)
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


# --- 3. Multiple Dispatch Implementation of test_op() ---

# Method 1 & 2 (Original chart_choice 1 and 2): Test decision is test_stat < crit_val
function test_op(ts, chart_choice::Shannon; m::Int=3, d::Int=1, alpha=0.05)
  test_stat, crit_val = _common_chart_calculations(ts, Shannon(); m=m, d=d, alpha=alpha)
  # Decision: test_stat < crit_val
  return (test_stat, crit_val, test_stat < crit_val)
end

function test_op(ts, chart_choice::ShannonExtropy; m::Int=3, d::Int=1, alpha=0.05)
  test_stat, crit_val = _common_chart_calculations(ts, ShannonExtropy(); m=m, d=d, alpha=alpha)
  # Decision: test_stat < crit_val
  return (test_stat, crit_val, test_stat < crit_val)
end


# Method 3 (Original chart_choice 3): Test decision is test_stat > crit_val
function test_op(ts, chart_choice::DistanceToWhiteNoise; m::Int=3, d::Int=1, alpha=0.05)
  test_stat, crit_val = _common_chart_calculations(ts, DistanceToWhiteNoise(); m=m, d=d, alpha=alpha)
  # Decision: test_stat > crit_val
  return (test_stat, crit_val, test_stat > crit_val)
end


# Methods 4 through 7 (Original chart_choice 4:7): Test decision is abs(test_stat) > crit_val
function test_op(ts, chart_choice::UpDownBalance; m::Int=3, d::Int=1, alpha=0.05)
  test_stat, crit_val = _common_chart_calculations(ts, UpDownBalance(); m=m, d=d, alpha=alpha)
  # Decision: abs(test_stat) > crit_val
  return (test_stat, crit_val, abs(test_stat) > crit_val)
end

function test_op(ts, chart_choice::Persistence; m::Int=3, d::Int=1, alpha=0.05)
  test_stat, crit_val = _common_chart_calculations(ts, Persistence(); m=m, d=d, alpha=alpha)
  # Decision: abs(test_stat) > crit_val
  return (test_stat, crit_val, abs(test_stat) > crit_val)
end

function test_op(ts, chart_choice::RotationalAsymmetry; m::Int=3, d::Int=1, alpha=0.05)
  test_stat, crit_val = _common_chart_calculations(ts, RotationalAsymmetry(); m=m, d=d, alpha=alpha)
  # Decision: abs(test_stat) > crit_val
  return (test_stat, crit_val, abs(test_stat) > crit_val)
end

function test_op(ts, chart_choice::UpDownScaling; m::Int=3, d::Int=1, alpha=0.05)
  test_stat, crit_val = _common_chart_calculations(ts, UpDownScaling(); m=m, d=d, alpha=alpha)
  # Decision: abs(test_stat) > crit_val
  return (test_stat, crit_val, abs(test_stat) > crit_val)
end










# function test_op(ts; chart_choice, m=3, d=1, alpha=0.05)
#   # lam is not needed in this function, yet stat_op is generalized

#   # Check that the chart_choice is valid
#   if m == 3
#     @assert 1 <= chart_choice <= 7 "Wrong number for test statistic."
#   end

#   # Number of patterns when d is integer      
#   n_patterns = length(ts) - (m - 1) * d

#   # Compute p vectors
#   p_vec = stat_op(ts; chart_choice=chart_choice, m=m, d=d)[2]

#   # Compute test statistic and critical value
#   test_stat = chart_stat_op(p_vec, chart_choice)
#   crit_val = crit_val_op(chart_choice, m, n_patterns; alpha=alpha)

#   # Return tuple with test statistic, critical value and test decision
#   if chart_choice == 1
#     return (test_stat, crit_val, test_stat < crit_val)

#   elseif chart_choice == 2
#     return (test_stat, crit_val, test_stat < crit_val)

#   elseif chart_choice == 3
#     return (test_stat, crit_val, test_stat > crit_val)

#   elseif chart_choice in 4:7
#     return (test_stat, crit_val, abs(test_stat) > crit_val)

#   end

# end
