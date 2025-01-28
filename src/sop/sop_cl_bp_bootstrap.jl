"""
    function cl_sop_bp(
        data::Array{T,3}, lam, L0, cl_init, w, reps;
        chart_choice=3, jmin=4, jmax=6, verbose=false
    ) where {T<:Real}

Computes the control limit for the bootstrap version of the BP-SOP chart.

    The input parameters are:

- `data::Array{T,3}`: A 3D-array containing the images for the EWMA-chart.
- `lam`: A scalar value for lambda for the EWMA chart.
- `L0`: The desired average run length.
- `cl_init`: The initial value for the control limit.
- `w`: The window size for the BP-statistic.
- `reps`: The number of replications to compute the ARL.
- `chart_choice`: The chart choice for the SOP chart.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose`: A boolean to indicate whether to print the control limit and ARL for each iteration.
"""

function cl_sop_bp(
    data::Array{T,3}, lam, L0, cl_init, w, reps;
    chart_choice=3, jmin=4, jmax=6, verbose=false
) where {T<:Real}

    # Compute a 3D-array, which contains relative frequencies of p-hat values for 
    # each pircture (rows) and d1-d2 combination (third dimension)
    p_array = compute_p_array_bp(data, w; chart_choice=chart_choice)

    L1 = 0.0
    for j in jmin:jmax
        for dh in 1:80
            cl_init = cl_init + (-1)^j * dh / 10^j
            L1 = arl_sop_bp_bootstrap(
                p_array, lam, cl_init, w, reps; chart_choice=chart_choice
            )[1]
            if verbose
                println("cl = ", cl_init, "\t", "ARL = ", L1)
            end
            if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
                break
            end
        end
        cl_init = cl_init
    end

    if L1 < L0
        cl_init = cl_init + 1 / 10^jmax
    end
    return cl_init

end
