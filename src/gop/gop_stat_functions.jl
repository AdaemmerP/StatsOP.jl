
# ---------------------------------------------------------------------------- #
# ---------------      Methods for non-sequential testing   ------------------ #
# ---------------------------------------------------------------------------- #
"""
    stat_gop(data, null_dist, chart_choice, m=3, d=1)
    stat_gop(data, null_dist, lam, chart_choice, m=3, d=1)

Compute the chart statistic based on generalized ordinal patterns (GOPs) for the
discrete-valued time series `data`. GOPs extend ordinal patterns by accounting for ties;
see Weiß and Schnurr (2024).

The first method computes the statistic once for the whole series and returns a scalar.
The second method additionally applies EWMA smoothing with parameter `lam` and returns
the vector of sequentially computed chart statistics.

- `data`: discrete-valued time series.
- `null_dist::DiscreteUnivariateDistribution`: in-control (null) distribution, used to
  compute the in-control GOP distribution via [`fill_p0!`](@ref).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `chart_choice`: [`D_Chart`](@ref)`()` or `Persistence()`.
- `m::Int=3`: length of the ordinal patterns (currently only `3` is supported).
- `d=1`: delay between observations of a pattern.
"""
function stat_gop(
  data, null_dist::DiscreteUnivariateDistribution, chart_choice::Union{D_Chart,Persistence}, m::Int=3, d=1
)

  # Compute lookup array and number of ops
  lookup_array_gop = _LOOKUP_GOP
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


# ---------------------------------------------------------------------------- #
# ---------------      Methods for sequential testing     -------------------- #
# ---------------------------------------------------------------------------- #
# Function to compute chart statistic
function stat_gop(data, null_dist::DiscreteUnivariateDistribution, lam, chart_choice::Union{D_Chart,Persistence}, m::Int=3, d=1)

  # Compute lookup array and number of ops
  lookup_array_gop = _LOOKUP_GOP
  p = zeros(13)
  bin = zeros(Int, 13)
  win = zeros(Int, m)
  ix = zeros(Int, m)
  p_p0 = zeros(13)
  p0 = zeros(13)

  fill_p0!(p0, null_dist)
  number_of_patterns = length(data) - (m - 1) * d
  stats_all = zeros(number_of_patterns)

  # initialze EWMA statistic, Equation (17), in the paper
  p .= p0

  for i in 1:number_of_patterns # enumerate(dindex_ranges)

    # create unit range for indexing data
    unit_range = range(i; step=d, length=m)
    # view of data
    x_seq = view(data, unit_range) # x_seq .= view(data, j) 

    # compute ordinal pattern based on permutations
    competerank!(win, x_seq, ix)

    # Binarization of ordinal pattern
    bin[lookup_array_gop[win[1], win[2], win[3]]] = 1
    # Compute EWMA statistic
    @. p = lam * bin .+ (1 - lam) * p
    # statistic based on smoothed p-estimate

    @. p_p0 = p - p0
    stat = chart_stat_gop(p_p0, chart_choice)

    # Save temporary test statistic
    stats_all[i] = stat

    # Reset binarization vector
    fill!(bin, 0)
  end

  return stats_all

end
