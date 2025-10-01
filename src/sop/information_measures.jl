export chart_stat_sop, TauHat, KappaHat, TauTilde, KappaTilde

# Build concrete types for information measures for SOPs
struct TauHat <: InformationMeasure end
struct KappaHat <: InformationMeasure end
struct TauTilde <: InformationMeasure end
struct KappaTilde <: InformationMeasure end

# Build Refinement Types
abstract type RefinedType end
struct RotationType <: RefinedType end
struct DirectionType <: RefinedType end
struct DiagonalType <: RefinedType end


"""
    chart_stat_sop(p_ewma, chart_choice)

Compute the the test statistic for spatial ordinal patterns. The first input is 
a vector with three values, based on SOP counts. The second input is the chart.     

"""
function chart_stat_sop(p_vec, chart_choice::InformationMeasure)

  @assert 1 <= chart_choice <= 7 "chart_choice must be between 1 and 7"

  # Initialize test statistic
  chart_val = 0

  # Test statistics on unrefined types
  if chart_choice == 1
    chart_val = p_vec[1] - 1.0 / 3.0
  elseif chart_choice == 2
    chart_val = p_vec[2] - p_vec[3]
  elseif chart_choice == 3
    chart_val = p_vec[3] - 1.0 / 3.0
  elseif chart_choice == 4
    chart_val = p_vec[1] - p_vec[2]

    # Test statistics for refined types  
  elseif chart_choice == 5
    # H-chart: Equation (7), Weiss and Kim
    q = length(p_vec)
    for i in axes(p_vec, 1)
      p_vec[i] > 0 && (chart_val -= p_vec[i] * log(p_vec[i])) # to avoid log(0)          
    end
    # Re-scale
    chart_val = (-2) / q * (chart_val - log(q))

  elseif chart_choice == 6
    # Hex-chart: Equation (7), Weiss and Kim
    q = length(p_vec)
    for i in axes(p_vec, 1)
      p_vec[i] < 1 && (chart_val -= (1 - p_vec[i]) * log(1 - p_vec[i])) # to avoid log of negative value    
    end
    # Re-scale
    chart_val = (-2) * (1 - 1 / q) * (chart_val - (q - 1) * log(q / (q - 1)))

  else # chart_choice == 7
    # Δ-chart: Equation (7), Weiss and Kim
    q = length(p_vec)
    for i in axes(p_vec, 1)
      chart_val += (p_vec[i] - 1 / q)^2
    end

  end

  return chart_val

end


