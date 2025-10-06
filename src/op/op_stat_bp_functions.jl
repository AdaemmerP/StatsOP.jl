# Function to compute test statistics for BP-OP and BL-OP tests 
function stat_op_bp(data; chart_choice::InformationMeasure, m::Int=3, w=3, ljung_box::Bool=false)

    @assert chart_choice in 1:7 "chart_choice must be between 1 and 7"

    # Compute lookup array and number of ops
    lookup_array_op = compute_lookup_array_op(m=m)
    m_fact = factorial(m)

    # Pre-allocate
    bin = zeros(Int, m_fact)
    p_rel = zeros(Float64, m_fact)
    win = Vector{Int64}(undef, m)
    bp_stats_all = Vector{Float64}(undef, w)

    for (i, d) in enumerate(1:w)

        for range_start = 1:(length(data)-(m-1)*d) # iterate through time series

            # create unit range for indexing data
            unit_range = range(range_start; step=d, length=m)

            # create view of data based on unit range
            x_long = view(data, unit_range)

            # compute ordinal pattern based on permutations
            sortperm!(win, x_long)

            if m == 2
                bin[lookup_array_op[win[1], win[2]]] += 1
            elseif m == 3
                bin[lookup_array_op[win[1], win[2], win[3]]] += 1
            elseif m == 4
                bin[lookup_array_op[win[1], win[2], win[3], win[4]]] += 1
            elseif m == 5
                bin[lookup_array_op[win[1], win[2], win[3], win[4], win[5]]] += 1
            end

        end # end of range loop

        # Compute relative frequency of types
        p_rel .= bin ./ sum(bin)
        bp_stats_all[i] = OrdinalPatterns.chart_stat_op(p_rel, chart_choice)

        # reset bin for next iteration
        fill!(bin, 0)

    end # end of w loop

    bp_val = 0.0
    log_nr_perm = log(m_fact)

    # ---------------------------------------------------------------------------#
    #                   Sum up the individual test statistics                    # 
    # ---------------------------------------------------------------------------#
    # Weighting based on Ljung-Box?
    if ljung_box
        # Iterator object for BL-weights
        stat_weights = Iterators.map(d -> length(data) - (m - 1) * d, 1:w)
    else
        # Iterator object for BP-weights
        stat_weights = Iterators.repeated(length(data) - m + 1, w)
    end

    # (1) H-chart 
    if chart_choice == 1
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * (log_nr_perm - bp_stats_all[i])
        end
        return 2 / m_fact * bp_val

        # (2) Hex-chart
    elseif chart_choice == 2
        term = (m_fact - 1) * log(m_fact / (m_fact - 1))
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * (term - bp_stats_all[i])
        end
        return (2 * (m_fact - 1) / m_fact) * bp_val

        # (3) Δ-chart  
    elseif chart_choice == 3
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * bp_stats_all[i]
        end
        return bp_val

        # (4) β-chart, (5) τ-chart, (6) γ-chart, (7) δ-chart  
    elseif chart_choice in 4:7
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * bp_stats_all[i]^2
        end
        return bp_val

    end # end of if statement
    # ---------------------------------------------------------------------------#

end # end of function


# Function to compute test statistics for BP-OP and BL-OP tests but mutates the input data  
function stat_op_bp!(
    data,
    bin,
    p_rel,
    win,
    bp_stats_all,
    lookup_array_op;
    chart_choice::InformationMeasure,
    m::Int=3,
    w=3,
    ljung_box::Bool=false,
)

    @assert chart_choice in 1:7 "chart_choice must be between 1 and 7"

    # Compute lookup array and number of ops  
    m_fact = factorial(m)

    for (i, d) in enumerate(1:w)

        for range_start = 1:(length(data)-(m-1)*d) # iterate through time series

            # create unit range for indexing data      
            unit_range = range(range_start; step=d, length=m)

            # create view of data based on unit range
            x_long = view(data, unit_range)

            # compute ordinal pattern based on permutations
            sortperm!(win, x_long)

            if m == 2
                bin[lookup_array_op[win[1], win[2]]] += 1
            elseif m == 3
                bin[lookup_array_op[win[1], win[2], win[3]]] += 1
            elseif m == 4
                bin[lookup_array_op[win[1], win[2], win[3], win[4]]] += 1
            elseif m == 5
                bin[lookup_array_op[win[1], win[2], win[3], win[4], win[5]]] += 1
            end

        end # end of range loop

        # Compute relative frequency of types
        p_rel .= bin ./ sum(bin)
        bp_stats_all[i] = OrdinalPatterns.chart_stat_op(p_rel, chart_choice)

        # reset bin for next iteration
        fill!(bin, 0)
    end # end of w loop

    bp_val = 0.0
    log_nr_perm = log(m_fact)

    # ---------------------------------------------------------------------------#
    #                   Sum up the individual test statistics                    # 
    # ---------------------------------------------------------------------------#
    # Weighting based on Ljung-Box?
    if ljung_box
        # Iterator object for BL-weights
        stat_weights = Iterators.map(d -> length(data) - (m - 1) * d, 1:w)
    else
        # Iterator object for BP-weights
        stat_weights = Iterators.repeated(length(data) - m + 1, w)
    end

    # (1) H-chart 
    if chart_choice == 1
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * (log_nr_perm - bp_stats_all[i])
        end
        return 2 / m_fact * bp_val

        # (2) Hex-chart
    elseif chart_choice == 2
        term = (m_fact - 1) * log(m_fact / (m_fact - 1))
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * (term - bp_stats_all[i])
        end
        return (2 * (m_fact - 1) / m_fact) * bp_val

        # (3) Δ-chart  
    elseif chart_choice == 3
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * bp_stats_all[i]
        end
        return bp_val

        # (4) β-chart, (5) τ-chart, (6) γ-chart, (7) δ-chart  
    elseif chart_choice in 4:7
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * bp_stats_all[i]^2
        end
        return bp_val

    end # end of if statement
    # ---------------------------------------------------------------------------#

end # end of function

# Function to compute critical values -> 
function crit_val_op_bp(; chart_choice::InformationMeasure, w, m, alpha=0.05)

    @assert chart_choice in 1:7 "chart_choice must be between 1 and 7"


    # ---------------------------------------------------------------------------#
    #                               op-length of 2
    # ---------------------------------------------------------------------------#

    if m == 2
        if chart_choice in (1, 3)
            return 1 / 6 * quantile(Chisq(w), 1 - alpha)
        end
    end



    # ---------------------------------------------------------------------------#
    #                               op-length of 3
    # ---------------------------------------------------------------------------#
    if m == 3
        # for H-chart, Hex-chart and Δ-chart
        if chart_choice in 1:3
            if w == 1
                return 1.484224
            elseif w == 2
                return 2.533081
            elseif w == 3
                return 3.345710
            elseif w == 4
                return 4.207398
            elseif w == 5
                return 4.946716
            end
        end

        # critical value for β-chart
        if chart_choice == 4
            return 1 / 3 * quantile(Chisq(w), 1 - alpha)
        end

        # critical value for τ-chart
        if chart_choice == 5
            if w == 1
                return 0.6829163
            elseif w == 2
                return 1.0825457
            elseif w == 3
                return 1.4059123
            elseif w == 4
                return 1.7176861
            elseif w == 5
                return 1.9966817
            end
        end

        # critical value for γ-chart
        if chart_choice == 6
            if w == 1
                return 1.536584
            elseif w == 2
                return 2.413527
            elseif w == 3
                return 3.142336
            elseif w == 4
                return 3.825928
            elseif w == 5
                return 4.456852
            end
        end

        # critical value for δ-chart
        if chart_choice == 7
            if w == 1
                return 2.560972
            elseif w == 2
                return 4.282555
            elseif w == 3
                return 5.467196
            elseif w == 4
                return 6.781190
            elseif w == 5
                return 7.795281
            end
        end

    end
end
