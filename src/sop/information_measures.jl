export chart_stat_sop, TauHat, KappaHat, TauTilde, KappaTilde, Shannon, ShannonExtropy, DistanceToWhiteNoise

# Build concrete types for information measures for SOPs
struct TauHat <: ComplexityEstimator end
struct KappaHat <: ComplexityEstimator end
struct TauTilde <: ComplexityEstimator end
struct KappaTilde <: ComplexityEstimator end

# Build Refinement Types
abstract type RefinedType end
struct RotationType <: RefinedType end
struct DirectionType <: RefinedType end
struct DiagonalType <: RefinedType end


function chart_stat_sop(p_vec, ::TauHat)
  return p_vec[1] - 1.0 / 3.0
end

function chart_stat_sop(p_vec, ::KappaHat)
  return p_vec[2] - p_vec[3]
end

function chart_stat_sop(p_vec, ::TauTilde)
  return p_vec[3] - 1.0 / 3.0
end

function chart_stat_sop(p_vec, ::KappaTilde)
  return p_vec[1] - p_vec[2]
end

function chart_stat_sop(p_vec, ::Shannon)
  # H-chart: Equation (7), Weiss and Kim
  chart_val = 0.0
  q = length(p_vec)
  for i in axes(p_vec, 1)
    p_vec[i] > 0 && (chart_val -= p_vec[i] * log(p_vec[i])) # to avoid log(0)
  end
  # Re-scale
  return (-2) / q * (chart_val - log(q))
end

function chart_stat_sop(p_vec, ::ShannonExtropy)
  # Hex-chart: Equation (7), Weiss and Kim
  chart_val = 0.0
  q = length(p_vec)
  for i in axes(p_vec, 1)
    p_vec[i] < 1 && (chart_val -= (1 - p_vec[i]) * log(1 - p_vec[i])) # to avoid log of negative value
  end
  # Re-scale
  return (-2) * (1 - 1 / q) * (chart_val - (q - 1) * log(q / (q - 1)))
end

function chart_stat_sop(p_vec, ::DistanceToWhiteNoise)
  # Δ-chart: Equation (7), Weiss and Kim
  chart_val = 0.0
  q = length(p_vec)
  for i in axes(p_vec, 1)
    chart_val += (p_vec[i] - 1 / q)^2
  end
  return chart_val
end