
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


function chart_stat_sop(p_vec, chart_type::Shannon{T}) where {T}
  return chart_stat_op(p_vec, chart_type)
end

function chart_stat_sop(p_vec, chart_type::ShannonExtropy{T}) where {T}
  return chart_stat_op(p_vec, chart_type)
end

function chart_stat_sop(p_vec, chart_type::DistanceToWhiteNoise)
  return chart_stat_op(p_vec, chart_type)
end

# Rescaling (Theorem 2.1 / Corollaries 2.2, 3.1.1, 3.2.1, 3.3.1, Weiß and Kim 2025)
function rescale_sop(val, q, ::Shannon{T}) where {T}
  return (-2 / q) * (val - log(q))
end

function rescale_sop(val, q, ::ShannonExtropy{T}) where {T}
  return (-2) * (1 - 1 / q) * (val - (q - 1) * log(q / (q - 1)))
end

function rescale_sop(val, q, ::DistanceToWhiteNoise)
  return val
end




