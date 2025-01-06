#========================================================================

Multiple Dispatch for 'cl_sop()':
  1. Theoretical distribution
  2. Bootstraping

========================================================================#
"""
    cl_sop(
      lam, L0, sop_dgp::ICSP, cl_init, reps=10_000; 
      chart_choice=3, jmin=4, jmax=6, verbose=false, d=1
    )

Compute the control limit for a given in-control distribution. The input parameters are:
  
- `lam::Float64`: A scalar value for lambda for the EWMA chart.
- `L0::Float64`: A scalar value for the desired average run length.
- `sop_dgp::ICSP`: A struct for the in-control spatial DGP.
- `cl_init::Float64`: A scalar value for the initial control limit. This is used to find the control limit.
- `reps::Int`: An integer value for the number of repetitions.
- `chart_choice::Int`: An integer value for the chart choice. The options are 1-4. The default value is 3.
- `jmin::Int`: An integer value for the minimum value for the control limit.
- `jmax::Int`: An integer value for the maximum value for the control limit.
- `verbose::Bool`: A boolean value whether to print the control limit and the average run length.
- `d::Int` An integer value for the embedding dimension. The default value is 1.


```julia-repl
#-- Example
# Parameters
lam = 0.1
L0 = 370
sop_dgp = ICSP(20, 20, Normal(0, 1))
cl_init = 0.5
reps = 10_000
chart_choice = 2
jmin = 4
jmax = 6
verbose = true
d = 1
cl_sop(lam, L0, sop_dgp, cl_init, reps; chart_choice, jmin, jmax, verbose, d)
```
"""
function cl_sop(
  lam, L0, sop_dgp::ICSP, cl_init, d1::Int, d2::Int, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

  L1 = 0.0

  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(lam, cl_init, sop_dgp, d1, d2, reps; chart_choice)[1]
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
function cl_sop(lam, L0, p_mat::Array{Float64,2}, cl_init, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(
        lam, cl_init, p_mat, reps; chart_choice=chart_choice
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
function cl_sop(lam, L0, p_array::Array{Float64,3}, cl_init, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(
        lam, cl_init, p_array, reps; chart_choice=chart_choice
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


function cl_sop(
  lam, L0, sp_dgp, cl_init, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000;
  chart_choice=3, jmin=4, jmax=6, verbose=false
)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sop(
        lam, cl_init, sp_dgp, d1_vec, d2_vec, reps;
        chart_choice=chart_choice
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


