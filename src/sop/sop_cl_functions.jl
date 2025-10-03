"""
    cl_sop(
  sop_dgp::ICSTS, lam, L0, cl_init, d1::Int, d2::Int, reps=10_000;
  chart_choice::InformationMeasure=TauTilde(), jmin=4, jmax=6, verbose=false
)

Compute the control limit for a given in-control process. The input parameters are:
  
- `sop_dgp`: The in-control spatial process (ICSTS) to use for the control limit.
- `lam::Float64`:  A scalar value for lambda for the EWMA chart.
- `L0::Float64`: The desired average run length.
- `cl_init::Float64`: The initial value for the control limit.
- `d1::Int`: The first (row) delay for the spatial process.
- `d2::Int`: The second (column) delay for the spatial process.
- `reps::Int`: The number of replications to compute the ARL.
- `chart_choice::Int`: The chart choice for the SOP chart.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose::Bool`: A boolean to indicate whether to print the control limit and ARL for each iteration.
"""
function cl_sop(
    sop_dgp::ICSTS, lam, L0, cl_init, d1::Int, d2::Int, reps=10_000;
    chart_choice::InformationMeasure=TauTilde(), refinement::Int=0, jmin=4, jmax=6, verbose=false
)

    L1 = 0.0

    for j in jmin:jmax
        for dh in 1:80
            cl_init = cl_init + (-1)^j * dh / 10^j
            L1 = arl_sop_ic(
                sop_dgp, lam, cl_init, d1, d2, reps;
                chart_choice=chart_choice, refinement=refinement
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
