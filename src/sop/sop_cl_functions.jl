#========================================================================
Multiple Dispatch for 'cl_sop()':
  1. In-control limits for one d₁-d₂ combination
  2. In-control limits for multiple d₁-d₂ combinations (BP-statistic)
  3. In-control limits using bootstraping for one d₁-d₂ combination
  4. In-control limits using bootstraping for BP-statistic
========================================================================#
"""
    cl_sop(
  lam, L0, sop_dgp::ICSP, cl_init, d1::Int, d2::Int, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

Compute the control limit for a given in-control distribution. The input parameters are:
  
- `lam::Float64`:  A scalar value for lambda for the EWMA chart.
- `L0::Float64`: The desired average run length. 
- `sop_dgp::ICSP`: A struct for the in-control spatial process.
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
  sop_dgp::ICSP, lam, L0, cl_init, d1::Int, d2::Int, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

  L1 = 0.0

  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(sop_dgp, lam, cl_init, d1, d2, reps; chart_choice)[1]
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


#--- Method to compute critical limits based on bootstraping for one d₁-d₂ combination
"""
    cl_sop(
  lam, L0, p_mat::Array{Float64,2}, cl_init, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

Compute the (bootstrap) SOP control limit for the EWMA-chart for one delay (d₁-d₂) combination.
The function returns the control limit for a given average run.

The input parameters are:

- `lam::Float64`:  A scalar value for lambda for the EWMA chart.
- `L0::Float64`: The desired average run length.
- `p_mat::Array{Float64,2}`: The matrix with relative frequencies of the SOPs. These
can be computed using the `compute_p_array()` function.
- `cl_init::Float64`: The initial value for the control limit.
- `reps::Int`: The number of replications to compute the ARL.
- `chart_choice::Int`: The chart choice for the SOP chart.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose::Bool`: A boolean to indicate whether to print the control limit and ARL for each iteration.  
"""
function cl_sop(
  lam, L0, p_mat::Array{Float64,2}, cl_init, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(
        p_mat, lam, cl_init, reps; chart_choice=chart_choice
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

#--- Method to compute critical limits based on bootstraping for BP-statistic
"""
    cl_sop(
  lam, L0, p_array::Array{Float64,3}, cl_init, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

Compute the (bootstrap) SOP control limit for the EWMA-chart for the BP-statistic.
The function returns the control limit for a given average run. The input parameters are:

- `lam::Float64`:  A scalar value for lambda for the EWMA chart.
- `L0::Float64`: The desired average run length.
- `p_array::Array{Float64,3}`: The array with relative frequencies of the SOPs. These
can be computed using the `compute_p_array()` function.
- `cl_init::Float64`: The initial value for the control limit.
- `reps::Int`: The number of replications to compute the ARL.
- `chart_choice::Int`: The chart choice for the SOP chart.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose::Bool`: A boolean to indicate whether to print the control limit and ARL for each iteration.
"""
function cl_sop(
  lam, L0, p_array::Array{Float64,3}, cl_init, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(
        p_array, lam, cl_init, reps; chart_choice=chart_choice
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


"""
    cl_sop(
  lam, L0, sp_dgp, cl_init, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

Compute the control limit for the EWMA-chart for the BP-statistic.
The function returns the control limit for a given average run. The input parameters are:

- `lam::Float64`:  A scalar value for lambda for the EWMA chart.
- `L0::Float64`: The desired average run length.
- `sp_dgp`: The in-control spatial process (ICSP) to use for the control limit.
- `cl_init::Float64`: The initial value for the control limit.
- `d1_vec::Vector{Int}`: The vector of first (row) delays for the spatial process.
- `d2_vec::Vector{Int}`: The vector of second (column) delays for the spatial process.
- `reps::Int`: The number of replications to compute the ARL.
- `chart_choice::Int`: The chart choice for the SOP chart.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose::Bool`: A boolean to indicate whether to print the control limit and ARL for each iteration.
"""
function cl_sop_bp(
  sp_dgp::ICSP, lam, L0, cl_init, w, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop_bp(
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


function cl_sop_bp(
  data::Array{T, 3}, lam, L0, cl_init, w, reps;
  chart_choice=3, jmin=4, jmax=6, verbose=false
) where {T<:Real}

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop_bp(
        data, lam, cl_init, w, reps; chart_choice=chart_choice
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
