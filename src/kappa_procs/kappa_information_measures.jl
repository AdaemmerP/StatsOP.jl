export chart_stat_qual,
  KappaN1,
  KappaN2,
  KappaO1,
  KappaO2

struct KappaN1 <: InformationMeasure end
struct KappaN2 <: InformationMeasure end
struct KappaO1 <: InformationMeasure end
struct KappaO2 <: InformationMeasure end

function chart_stat_qual(q, Q, ::Union{KappaN1,KappaO1,KappaN2,KappaO2})

  # Sum for numerator and denominator part
  numerator_sum = 0.0
  denominator_sum = 0.0
  for i in axes(q, 1)
    numerator_sum += q[i]^2
    denominator_sum += q[i] * (1 - q[i])
  end

  return (Q - numerator_sum) / denominator_sum

end

# function chart_stat_qual(p_or_f, Q, ::Union{})

#   # Sum for numerator and denominator part
#   numerator_sum = 0.0
#   denominator_sum = 0.0
#   for i in axes(p_or_f, 1)
#     numerator_sum += p_or_f[i]^2
#     denominator_sum += p_or_f[i] * (1 - p_or_f[i])
#   end

#   return (Q - numerator_sum) / denominator_sum

# end