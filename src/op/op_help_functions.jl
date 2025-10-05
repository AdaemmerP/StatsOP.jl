
# --- Function to select abort criterium --- #
# see Equation (6), page 342, Weiss and Testik (2023)
function abort_criterium_op(stat, cl, chart_choice)
  if chart_choice == 1
    # Permutation Entropy-H chart: Equation (3), page 342, Weiss and Testik (2023)
    return stat > cl
  elseif chart_choice == 2
    # Permutation Extropy-hex chart: Equation (3), page 342, Weiss and Testik (2023), Equation (15), page 6 in the paper      
    return stat > cl
  elseif chart_choice == 3
    # Delta3 chart: Equation (3), page 342, Weiss and Testik (2023)
    return stat < cl
  elseif chart_choice == 4
    # beta chart: Equation (4), page 342, Weiss and Testik (2023), Equation (16), page 6 in the paper
    return abs(stat) < cl
  elseif chart_choice == 5
    #  tau chart: Equation (4), page 342, Weiss and Testik (2023), Equation (16), page 6 in the paper
    return abs(stat) < cl
  elseif chart_choice == 6
    # delta chart: Equation (4), page 342, Weiss and Testik (2023)
    return abs(stat) < cl
  else
    println("Wrong number for test statistic.")
  end
end

# Function that returns possible ranks of ordinal patterns
function get_ranks_op(; op_length=3)

  # Verify that the length of ordinal patterns is between 2 and 4
  @assert 2 <= op_length <= 4 "Length of ordinal patterns must be between 2 and 4."

  if op_length == 2

    return [1 2;
      2 1]

  elseif op_length == 3

    # Lexicographical ordering as in Bandt 2019
    return [1 2 3;
      1 3 2;
      2 1 3;
      2 3 1;
      3 1 2;
      3 2 1]

  elseif op_length == 4

    return [1 2 3 4;
      1 2 4 3;
      1 4 2 3;
      4 1 2 3;
      1 3 2 4;
      1 3 4 2;
      1 4 3 2;
      4 1 3 2;
      3 1 2 4;
      3 1 4 2;
      3 4 1 2;
      4 3 1 2;
      2 1 3 4;
      2 1 4 3;
      2 4 1 3;
      4 2 1 3;
      2 3 1 4;
      2 3 4 1;
      2 4 3 1;
      4 2 3 1;
      3 2 1 4;
      3 2 4 1;
      3 4 2 1;
      4 3 2 1]

  else

    permutations(1:op_length) |> collect

  end

end


#--- Lookup function for OPs
function compute_lookup_array_op(; op_length=3)

  # Get ranks
  ranks_op = get_ranks_op(op_length=op_length)

  # Check for op length
  if op_length == 2

    p_ops = @MArray zeros(Int, 2, 2)
    sort_tmp = MVector{2,Int}(undef) # Vector{Int}(undef, 2)

    for (i, j) in enumerate(eachrow(ranks_op))
      sortperm!(sort_tmp, j)
      @views p_ops[i, :] = sort_tmp
    end

    # Construct multi-dimensional lookup array
    lookup_array = @MArray zeros(Int, 2, 2)

    for i in axes(p_ops, 1)
      @views lookup_array[p_ops[i, :][1], p_ops[i, :][2]] = i
    end

  elseif op_length == 3

    # Pre-allocate
    p_ops = @MArray zeros(Int, 6, 3)
    sort_tmp = MVector{3,Int}(undef)  # @MArray Vector{Int}(undef, 3)

    for (i, j) in enumerate(eachrow(ranks_op))
      sortperm!(sort_tmp, j)
      @views p_ops[i, :] = sort_tmp
    end

    # Construct multi-dimensional lookup array 
    lookup_array = @MArray zeros(Int, 3, 3, 3)

    for i in axes(p_ops, 1)
      @views lookup_array[p_ops[i, :][1], p_ops[i, :][2], p_ops[i, :][3]] = i
    end

  elseif op_length == 4

    # Pre-allocate
    ranks_op = get_ranks_op(op_length=op_length)

    p_ops = @MArray zeros(Int, 24, 4)
    sort_tmp = MVector{4,Int}(undef) # Vector{Int}(undef, 4)

    for (i, j) in enumerate(eachrow(ranks_op))
      sortperm!(sort_tmp, j)
      @views p_ops[i, :] = sort_tmp
    end

    # Construct multi-dimensional lookup array
    lookup_array = @MArray zeros(Int, 4, 4, 4, 4)

    for i in axes(p_ops, 1)
      @views lookup_array[p_ops[i, :][1], p_ops[i, :][2], p_ops[i, :][3], p_ops[i, :][4]] = i
    end


  end

  return lookup_array

end


# Function to compute delay indices for ordinal patterns
function compute_dindex_op(data; op_length::Int=3, d::Union{Int,Vector{Int}}=1)

  # compute indices for equidistant step sizes 
  if d isa Int
    #@assert (length(data) < (op_length - 1) * d) "You either chose a too large delay or a too large order of OPs. "
    index_range = [range(x; step=d, length=op_length) for x in 1:(length(data)-(op_length-1)*d)]

    # compute indices for non-equidistant step sizes        
  elseif d isa Vector{Int}
    index_range = Vector{Vector{Int}}(undef, length(last(d):length(data)))
    index_range[1] = d
    for i in 2:length(index_range)
      index_range[i] = index_range[i-1] .+ 1
    end

  end

  return index_range

end
