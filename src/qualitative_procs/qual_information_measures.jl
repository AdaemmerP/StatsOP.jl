export chart_stat_qual,
  KNominal,
  KOrdinal

struct KNominal <: InformationMeasure end
struct KOrdinal <: InformationMeasure end

function chart_stat_qual(q, Q, ::KNominal)

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