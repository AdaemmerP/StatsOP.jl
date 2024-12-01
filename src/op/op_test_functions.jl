# Function to compute critical values based on Weiss (2022)
function crit_val_op(chart_choice, op_length, n_patterns; alpha=0.05)

    z2 = quantile(Normal(0, 1), 1 - alpha / 2)

    if op_length == 2

        @assert chart_choice ∈ [1, 3, 4] "Wrong number for test statistic. Choose 1, 3 or 4."

        qup2 = quantile(Chisq(1), 1 - alpha) / 6

        if chart_choice == 1
            # H-chart
            return log(2) - qup2 / n_patterns
        elseif chart_choice == 3
            # Δ-chart
            return qup2 / n_patterns
        elseif chart_choice == 4
            # β-chart
            z2 * sqrt(1 / 3 / n_patterns)
        end

    elseif op_length == 3

        @assert 1 <= chart_choice <= 6 "Wrong number for test statistic."
        @assert alpha ∈ [0.01, 0.05, 0.1] "Wrong alpha level. Choose 0.01, 0.05 or 0.1."

        # Compute unit root for quantile function of Q₃ = 1/12(2 + √2) χ₁² + 2/15 χ₁² + 1/10 χ₁² + 1/12(2 - √2) χ₁²
        if alpha == 0.01
            qup3 = 2.267254
        elseif alpha == 0.05
            qup3 = 1.484225
        elseif alpha == 0.1
            qup3 = 1.162639
        end

        if chart_choice == 1
            # H-chart
            return log(6) - 3 * qup3 / n_patterns

        elseif chart_choice == 2
            # Hex-chart       
            return 5 * log(6 / 5) - 3 * qup3 / 5 / n_patterns

        elseif chart_choice == 3
            # Δ-chart     
            return qup3 / n_patterns

        elseif chart_choice == 4
            # β-chart
            return z2 * sqrt(1 / 3 / n_patterns)

        elseif chart_choice == 5
            # τ-chart
            return z2 * sqrt(8 / 45 / n_patterns) # sqrt(2 / 5 / n_patterns)

        elseif chart_choice == 6
            # δ-chart
            return z2 * sqrt(2 / 3 / n_patterns)
        end
    end
end

function test_op(ts; chart_choice, op_length=3, d=1, alpha=0.05)

    # Check that the chart_choice is valid
    if op_length == 3
        @assert 1 <= chart_choice <= 6 "Wrong number for test statistic."
    end

    z2 = quantile(Normal(0, 1), 1 - alpha / 2)

    # Get indices to iterate over        
    n_patterns = length(ts) - (op_length - 1) * d # length(dindex_ranges)

    # Compute p vectors
    p_vec = stat_op(ts, 0.1; chart_choice=chart_choice, op_length=op_length, d=d)[2]
    # stat_op(ts, 0.1, chart_choice; op_length=op_length, d=d)[2]

    # Compute test statistic and critical value
    test_stat = chart_stat_op(p_vec, chart_choice)
    crit_val = crit_val_op(chart_choice, op_length, n_patterns; alpha=alpha)

    # Return tuple with test statistic, critical value and test decision
    if chart_choice == 1
        return (test_stat, crit_val, test_stat < crit_val)
    elseif chart_choice == 2
        return (test_stat, crit_val, test_stat < crit_val)
    elseif chart_choice == 3
        return (test_stat, crit_val, test_stat > crit_val)
    elseif chart_choice == 4
        return (test_stat, crit_val, abs(test_stat) > crit_val)
    elseif chart_choice == 5
        return (test_stat, crit_val, abs(test_stat) > crit_val)
    elseif chart_choice == 6
        return (test_stat, crit_val, abs(test_stat) > crit_val)
    end

end
