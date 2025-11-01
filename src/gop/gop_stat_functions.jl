export stat_gop


# ---------------------------------------------------------------------------- #
# ---------------      Methods for non-sequential testing   ------------------ #
# ---------------------------------------------------------------------------- #
# Function to compute D-chart and Persistence 
function stat_gop(
  data, null_dist::Union{Binomial,Poisson}, chart_choice::Union{D_Chart,Persistence}, m::Int=3, d=1
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

# Function to compute D-chart statistic
function stat_gop(
  data, null_dist::Union{Binomial,Poisson}, chart_choice::G_Chart, m::Int=3, d=1
)

  # Compute lookup array and number of ops
  lookup_array_gop = compute_lookup_array_gop()
  p = zeros(13)
  bin = zeros(Int, 13)
  win = zeros(Int, m)
  ix = zeros(Int, m)
  p_p0 = zeros(13)
  p0 = zeros(13)
  g_p = similar(p)
  G = [
    0 0 0 0 0 1 0 0 1 0 0 1 0;
    1 0 0 0 0 0 1 0 0 1 0 0 0;
    0 1 1 1 1 0 0 1 0 0 1 0 1
  ]
  G1G = G' * G

  fill_p0!(p0, null_dist)
  number_of_patterns = length(data) - (m - 1) * d

  for i in 1:number_of_patterns

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

  return chart_stat_gop(p_p0, g_p, G1G, chart_choice)

end


# ---------------------------------------------------------------------------- #
# ---------------      Methods for sequential testing     -------------------- #
# ---------------------------------------------------------------------------- #
# Function to compute chart statistic
function stat_gop(data, null_dist::Union{Binomial,Poisson}, lam, chart_choice::Union{D_Chart,Persistence}, m::Int=3, d=1)

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

# Function to compute chart statistic
function stat_gop(data, null_dist::Union{Binomial,Poisson}, lam, chart_choice::G_Chart, m::Int=3, d=1)

  # Compute lookup array and number of ops
  lookup_array_gop = compute_lookup_array_gop()
  p = zeros(13)
  bin = zeros(Int, 13)
  win = zeros(Int, m)
  ix = zeros(Int, m)
  p_p0 = zeros(13)
  p0 = zeros(13)
  G = [
    0 0 0 0 0 1 0 0 1 0 0 1 0;
    1 0 0 0 0 0 1 0 0 1 0 0 0;
    0 1 1 1 1 0 0 1 0 0 1 0 1
  ]
  G1G = G' * G
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
    # # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
    @. p = lam * bin .+ (1 - lam) * p
    # statistic based on smoothed p-estimate
    @. p_p0 = p - p0
    stat = chart_stat_gop(p_p0, G1G, chart_choice)
    # Save temporary test statistic
    stats_all[i] = stat

    # Reset binarization vector
    fill!(bin, 0)
  end

  return stats_all

end
