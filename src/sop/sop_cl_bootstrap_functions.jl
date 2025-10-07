"""
    cl_sop_bootstrap(
   data::Array{T,3}, lam, L0, cl_init, d1, d2, reps=10_000;
  chart_choice::InformationMeasure=TauTilde(), jmin=3, jmax=7, verbose=false
)

Compute the SOP control limit for the EWMA-chart based on bootstraping. The function returns the control limit for a given average run.

The input parameters are:

- `data::Array{T,3}`: A 3D-array containing the data for the EWMA-chart.
- `lam::Real`: The smoothing parameter for the EWMA-chart.
- `L0::Real`: The average run length for the EWMA-chart.
- `cl_init::Real`: The initial value for the control limit.
- `d1::Int`: The first dimension of the data array.
- `d2::Int`: The second dimension of the data array.
- `reps::Int=10_000`: The number of bootstrap replications.
- `chart_choice::Int`: The choice of the chart for the EWMA-chart. The default is 3.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose::Bool`: A boolean to indicate whether to print the control limit and ARL for each iteration.
"""
function cl_sop_bootstrap(
    data::Array{T,3}, lam, L0, cl_init, d1, d2, reps=10_000;
    chart_choice=TauTilde(), jmin=3, jmax=7, verbose=false
) where {T<:Real}

    p_array = compute_p_array(data, d1, d2; chart_choice=chart_choice)

    L1 = 0.0
    for j in jmin:jmax
        for dh in 1:80
            cl_init = cl_init + (-1)^j * dh / 10^j
            L1 = arl_sop_bootstrap(
                p_array, lam, cl_init, reps, chart_choice=chart_choice
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
