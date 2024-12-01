function chart_stat_op(p, chart_choice)

    # Verify that chart_choice is between 1 and 6
    @assert 1 <= chart_choice <= 6 "Wrong number for test statistic."
  
    value = 0
  
    if length(p) == 2
      if chart_choice == 4
        # β-chart for op_length of 2 
        return p[2] - p[1]
      end
    end
  
    if chart_choice == 1
      # H-chart: Equation (3), page 342, Weiss and Testik (2023)
      for i in axes(p, 1)
        p[i] > 0 && (value -= p[i] * log(p[i])) # to avoid log(0)          
      end
      return value
    elseif chart_choice == 2
      # Hex-chart: Equation (3), page 342, Weiss and Testik (2023), Equation (15), page 6 in the paper      
      for i in axes(p, 1)
        p[i] < 1 && (value -= (1 - p[i]) * log(1 - p[i])) # to avoid log(0)       
      end
      return value
    elseif chart_choice == 3
      # Δ-chart: Equation (3), page 342, Weiss and Testik (2023)
      op_length = length(p)
      for i in axes(p, 1)
        value += (p[i] - 1 / op_length)^2
      end
      return value
    elseif chart_choice == 4
      # β-chart: Equation (4), page 342, Weiss and Testik (2023), Equation (16), page 6 in the paper
      return p[6] - p[1]
    elseif chart_choice == 5
      # τ-chart: Equation (4), page 342, Weiss and Testik (2023), Equation (16), page 6 in the paper
      return p[6] + p[1] - (1 / 3)
    elseif chart_choice == 6
      # δ-chart: Equation (4), page 342, Weiss and Testik (2023)
      return p[4] + p[5] - p[3] - p[2]
    end
  
  end
  
  
# Function to compute chart statistic
function stat_op(data, lam; chart_choice, op_length::Int=3, d=1)
  #stat_op(data, lam, chart_choice; op_length::Int=3, d=1)
         

    # create vector with unit range for indexing 
    dindex_ranges = compute_dindex_op(data; op_length=op_length, d=d)
  
    # Compute lookup array and number of ops
    lookup_array_op = compute_lookup_array_op(op_length=op_length)
    op_length_fact = factorial(op_length)
  
    p = Vector{Float64}(undef, op_length_fact)
    p_count = zeros(Int, op_length_fact)
    fill!(p, 1 / op_length_fact)
    bin = Vector{Int64}(undef, op_length_fact)
    win = Vector{Int64}(undef, op_length)
    stats_all = Vector{Float64}(undef, length(dindex_ranges))
  
    for (i, j) in enumerate(dindex_ranges)
      x_long = view(data, j) # x_long .= view(data, j) # @views SVector{m}(data[j]) # 
      fill!(bin, 0)
      # compute ordinal pattern based on permutations
      order_vec!(x_long, win)
      # Binarization of ordinal pattern
      if op_length == 2
        bin[lookup_array_op[win[1], win[2]]] = 1
      elseif op_length == 3
        bin[lookup_array_op[win[1], win[2], win[3]]] = 1
      end
      # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
      @. p = lam * bin + (1 - lam) * p
      @. p_count += bin
      # statistic based on smoothed p-estimate
      stat = chart_stat_op(p, chart_choice)
      # Save temporary test statistic
      stats_all[i] = stat
    end
  
    p_rel = p_count ./ length(dindex_ranges)
    return [stats_all, p_rel]
  
  end
  
  