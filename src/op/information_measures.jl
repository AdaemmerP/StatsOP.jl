export Shannon,
  ShannonExtropy,
  DistanceToWhiteNoise,
  UpDownBalance,
  Persistence,
  RotationalAsymmetry,
  UpDownScaling,
  chart_stat_op

struct DistanceToWhiteNoise <: Entropy end
struct UpDownBalance <: ComplexityEstimator end
struct Persistence <: ComplexityEstimator end
struct RotationalAsymmetry <: ComplexityEstimator end
struct UpDownScaling <: ComplexityEstimator end

# H-chart: Equation (3), page 342, Weiss and Testik (2023)
function chart_stat_op(p_vec, ::Shannon)
  value = 0.0
  for i in axes(p_vec, 1)
    p_vec[i] > 0 && (value -= p_vec[i] * log(p_vec[i])) # to avoid log(0)
  end
  return value
end

# Hex-chart: Equation (3), page 342, Weiss and Testik (2023), Equation (15), page 6 in the paper
function chart_stat_op(p_vec, ::ShannonExtropy)
  value = 0.0
  for i in axes(p_vec, 1)
    p_vec[i] < 1 && (value -= (1 - p_vec[i]) * log(1 - p_vec[i])) # to avoid log of negative value
  end
  return value
end

# Δ-chart: Equation (3), page 342, Weiss and Testik (2023)
function chart_stat_op(p_vec, ::DistanceToWhiteNoise)
  op_length = length(p_vec)
  value = 0.0
  for i in axes(p_vec, 1)
    value += (p_vec[i] - 1 / op_length)^2
  end
  return value
end

# β-chart: Bandt (2019), equation (3)
function chart_stat_op(p_vec, ::UpDownBalance)
  if length(p_vec) == 2
    # β-chart for op_length of 2
    return p_vec[2] - p_vec[1]
  else
    return p_vec[1] - p_vec[6]
  end
end

# τ-chart: Bandt (2019), equation (4)
function chart_stat_op(p_vec, ::Persistence)
  return p_vec[1] + p_vec[6] - (1 / 3)
end

# γ-chart: Bandt (2019), equation (5)
function chart_stat_op(p_vec, ::RotationalAsymmetry)
  return p_vec[3] + p_vec[4] - p_vec[2] - p_vec[5]
end

# δ-chart: Bandt (2019), equation (6)
function chart_stat_op(p_vec, ::UpDownScaling)
  return p_vec[2] + p_vec[3] - p_vec[4] - p_vec[5]
end
