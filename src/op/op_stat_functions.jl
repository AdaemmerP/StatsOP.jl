
# TODO: Write one function for sequential charts, using λ, 
# TODO: Write one function for one chart, without λ,    

# # Function to build MVector for 'win'
# # This vector will be always filled in-place and then converted to an SVector
# # This makes the code fully generalizable for any m 
# function build_win(::Val{M}) where {M}
#   return MVector{M,Int}(undef)
# end

# # User calls this function:
# function stat_op(data; chart_choice::InformationMeasure, m::Int=3, d::Int=1)
#   # We aim to convert the runtime value 'm' into a compile-time type 'Val(m)'.
#   # Therefore we have two functions
#   return _stat_op_core(Val(m), data, chart_choice, d)
# end

# # Function to chart statistic and relative frequencies of ordinal patterns
# function _stat_op_core(::Val{M}, data, chart_choice, d::Int) where {M}

#   # Set m
#   m = M

#   # Compute lookup array and number of ops
#   lookup_array_op = compute_lookup_array_op(m=m)
#   m_fact = factorial(m)

#   p_vec = Vector{Float64}(undef, m_fact)
#   p_count = zeros(Int, m_fact)
#   fill!(p_vec, 1 / m_fact)
#   bin = Vector{Int64}(undef, m_fact)
#   win = build_win(Val(m))

#   # Loop over all possible ordinal patterns
#   for range_start in 1:(length(data)-(m-1)*d) #for (i, j) in enumerate(dindex_ranges)

#     # Reset binarization vector
#     fill!(bin, 0)

#     # create unit range for indexing data
#     unit_range = range(range_start; step=d, length=m)

#     x_long = view(data, unit_range)

#     # compute ordinal pattern based on permutations    
#     sortperm!(win, x_long)
#     win_static = SVector(win)

#     # Binarization of ordinal pattern
#     bin[lookup_array_op[win_static...]] = 1
#     # bin[lookup_array_op[win[1], win[2], win[3]]] = 1
#     # if m == 2
#     #   bin[lookup_array_op[win[1], win[2]]] = 1
#     # elseif m == 3
#     #   bin[lookup_array_op[win[1], win[2], win[3]]] = 1
#     # elseif m == 4
#     #   bin[lookup_array_op[win[1], win[2], win[3], win[4]]] = 1
#     # elseif m == 5
#     #   bin[lookup_array_op[win[1], win[2], win[3], win[4], win[5]]] = 1
#     # end

#     @. p_count += bin

#   end

#   p_rel = p_count ./ sum(p_count) #length(dindex_ranges)
#   stat = chart_stat_op(p_rel, chart_choice)
#   return stat # [stat, p_rel]

# end

# Function that uses the lehmer code to compute the ordinal pattern
function stat_op(data; chart_choice, m::Int=3, d::Int=1, add_noise::Bool=false)

  # pre-allocate
  m_fact = factorial(m)
  p_vec = Vector{Float64}(undef, m_fact)
  p_count = zeros(Int, m_fact)
  fill!(p_vec, 1 / m_fact)
  bin = zeros(Int, m_fact)
  win = zeros(Int, m)
  idx_used = zeros(Int, m)
  number_of_patterns = length(data) - (m - 1) * d

  if add_noise
    data = data .+ rand(length(data))
  end

  # Loop over all possible ordinal patterns
  for i in 1:number_of_patterns #for (i, j) in enumerate(dindex_ranges)

    # Reset binarization vector
    fill!(bin, 0)

    # create unit range for indexing data
    unit_range = range(i; step=d, length=m)
    x_long = view(data, unit_range)

    # compute ordinal pattern based on permutations    
    sortperm!(win, x_long)

    # Convert permutation to lehmer index
    index = perm_to_lehm_idx!(win, idx_used)
    fill!(idx_used, 0) # reset idx_used

    # Binarization of ordinal pattern
    bin[index] = 1

    @. p_count += bin

  end

  p_rel = p_count ./ sum(p_count) #length(dindex_ranges)
  stat = chart_stat_op(p_rel, chart_choice)
  return [stat, p_rel]

end



# Function to compute EWMA chart statistic
function stat_op(
  data, lam; chart_choice, m::Int=3, d::Int=1, add_noise::Bool=false
)

  # lookup_array_op = compute_lookup_array_op(m=m)
  m_fact = factorial(m)

  p_vec = Vector{Float64}(undef, m_fact)
  p_count = zeros(Int, m_fact)
  fill!(p_vec, 1 / m_fact)
  bin = Vector{Int64}(undef, m_fact)
  win = Vector{Int64}(undef, m)
  idx_used = zeros(Int, m)
  number_of_patterns = length(data) - (m - 1) * d
  stats_all = zeros(Float64, number_of_patterns)

  if add_noise
    data = data .+ rand(length(data))
  end

  for i in 1:number_of_patterns # (i, j) in enumerate(dindex_ranges)

    # Reset binarization vector
    fill!(bin, 0)

    # create unit range for indexing data
    unit_range = range(i; step=d, length=m)

    x_long = view(data, unit_range)

    # compute ordinal pattern based on permutations    
    sortperm!(win, x_long)

    # Convert permutation to lehmer index
    index = perm_to_lehm_idx!(win, idx_used)
    fill!(idx_used, 0)

    # Binarization of ordinal pattern
    bin[index] = 1

    # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
    @. p_vec = lam * bin + (1 - lam) * p_vec
    @. p_count += bin

    # statistic based on smoothed p-estimate
    stat = chart_stat_op(p_vec, chart_choice)

    # Save temporary test statistic
    stats_all[i] = stat
  end

  p_rel = p_count ./ sum(p_count) #length(dindex_ranges)
  return (stats_all, p_rel) # [stats_all, p_rel]

end

