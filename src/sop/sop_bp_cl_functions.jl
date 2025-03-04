"""
    cl_sop_bp(
  sp_dgp::ICSP, lam, L0, cl_init, w, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

Compute the control limit for the EWMA-chart for the BP-statistic.
The function returns the control limit for a given average run. The input parameters are:

- `sp_dgp::ICSTS`: The in-control spatial process (ICSTS).
- `lam::Float64`:  A scalar value for lambda for the EWMA chart.
- `L0::Float64`: The desired average run length.
- `cl_init::Float64`: The initial value for the control limit.
- `w::Int`: The window size for the BP-statistic.
- `reps::Int`: The number of replications to compute the ARL.
- `chart_choice::Int`: The chart choice for the SOP chart.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose::Bool`: A boolean to indicate whether to print the control limit and ARL for each iteration.
"""
function cl_sop_bp(
    sp_dgp::ICSTS, lam, L0, cl_init, w, reps=10_000;
    chart_choice=3, jmin=4, jmax=6, verbose=false
)

    L1 = 0.0
    for j in jmin:jmax
        for dh in 1:80
            cl_init = cl_init + (-1)^j * dh / 10^j
            L1 = arl_sop_bp_ic(
                sp_dgp, lam, cl_init, w, reps; chart_choice=chart_choice
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