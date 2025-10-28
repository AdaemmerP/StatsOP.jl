export D_Chart,
  G_Chart,
  chart_stat_gop


# Define abstract type for Information Measures
struct D_Chart <: InformationMeasure end
struct G_Chart <: ComplexityEstimator end

# Define functions for Information Measures
function chart_stat_gop(p_p0, ::D_Chart)
  # D-chart: Equation (18), page 7 in the paper
  return dot(p_p0, p_p0)

end

function chart_stat_gop(p_p0, G1G, ::G_Chart)

  # G-chart: Equation (20), page 7 in the paper
  return dot(p_p0, G1G, p_p0)

end

function chart_stat_gop(p_p0, ::Persistence)

  # Persistence for ordinal patterns
  idx = (1, 6, 8, 10, 11, 13)

  stat = 0.0
  for i in idx
    stat += p_p0[i]^2
  end

  return stat
end
