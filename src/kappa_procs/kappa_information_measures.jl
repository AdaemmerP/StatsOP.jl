export chart_stat_qual,
  KNominal,
  KOrdinal

struct KappaN1 <: InformationMeasure end
struct KappaN2 <: InformationMeasure end
struct KOrdinal1 <: InformationMeasure end
struct KOrdinal2 <: InformationMeasure end

function chart_stat_qual(q, Q, ::KappaN1)

  # Sum for numerator part
  numerator_sum = 0.0
  for i in axes(q, 1)
    numerator_sum += q[i]^2
  end

  # Sum for denominator part
  denominator_sum = 0.0
  for i in axes(q, 1)
    denominator_sum += q[i] * (1 - q[i])
  end

  return (Q - numerator_sum) / denominator_sum

end

function chart_stat_qual(p, Q, ::KappaN2)

  # Sum for numerator part
  numerator_sum = 0.0
  for i in axes(p, 1)
    numerator_sum += p[i]^2
  end

  # Sum for denominator part
  denominator_sum = 0.0
  for i in axes(p, 1)
    denominator_sum += p[i] * (1 - p[i])
  end

  return (Q - numerator_sum) / denominator_sum

end