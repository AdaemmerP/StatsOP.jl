# Function to compute D-chart and Persistence 
function stat_ual(
  data, null_dist, chart_choice::Union{KNominal,KOrdinal}
)

  # Compute lookup array and number of ops
  lookup_array_gop = compute_lookup_array_gop()
  p = zeros(13)
  bin = zeros(Int, 13)
  win = zeros(Int, m)
  ix = zeros(Int, m)
  p_p0 = zeros(13)
  p0 = zeros(13)

  fill_p0!(p0, null_dist)
  number_of_patterns = length(data) - (m - 1) * d

  for i in 1:number_of_patterns # enumerate(dindex_ranges)

    # create unit range for indexing data
    unit_range = range(i; step=d, length=m)
    # view of data
    x_seq = view(data, unit_range)

    # compute ordinal pattern based on permutations
    competerank!(win, x_seq, ix)

    # Binarization of ordinal pattern
    j, k, l = win
    bin[lookup_array_gop[j, k, l]] += 1

  end

  # Test statistic
  p = bin ./ sum(bin)
  @. p_p0 = p - p0

  return chart_stat_gop(p_p0, chart_choice)

end
