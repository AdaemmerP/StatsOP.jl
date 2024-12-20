
"""
    cl_sacf(m, n, lam, L0, reps, cl_init, jmin, jmax, verbose, dist_error)

Compute the control limit for the exponentially weighted moving average (EWMA) control chart using the spatial autocorrelation function (SACF) for a lag length of 1. The function returns the control limit for a given average run length (ARL) `L0` and a given number of repetitions `reps`. The input arguments are:
 
- `m::Int`: The number of rows in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `n::Int`: The number of columns in the matrix for the SOP matrix. Note that the original matrix will have dimensions (m + 1) x (n + 1).
- `lam`: The smoothing parameter for the EWMA control chart.
- `L0`: The average run length (ARL) to use for the control limit.
- `reps`: The number of repetitions to compute the ARL.
- `cl_init`: The initial control limit to use for the EWMA control chart. If set to 0, the function will search for the control limit that gives an ARL greater than `L0`.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose`: A boolean to indicate whether to print the control limit and ARL for each iteration.
- `dist_error`: The distribution to use for the error term in the SACF function. This can be any univariate distribution from the `Distributions.jl` package or a custom distribution with a defined method for `rand()` and `rand!()`.

```julia
#--- Example
# Set parameters
m = 10
n = 10
lam = 0.1
L0 = 370
reps = 1000
cl_init = 0.05
jmin = 4
jmax = 7
verbose = true
dist_error = Normal(0, 1)

# Compute control limit
cl = cl_sacf(m, n, lam, L0, reps, cl_init, jmin, jmax, verbose, dist_error)
```
"""
function cl_sacf(lam, L0, sp_dgp::ICSP, cl_init, d1::Int, d2::Int, reps=10_000; jmin=4, jmax=6, verbose=false)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sacf(lam, cl_init, sp_dgp, d1, d2)[1]      
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

# 
function cl_sacf(lam, L0, sp_dgp::ICSP, cl_init, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000; jmin=4, jmax=6, verbose=false)

    L1 = 0.0
    for j in jmin:jmax
      for dh in 1:80
        cl_init = cl_init + (-1)^j * dh / 10^j
        L1 = arl_sacf(lam, cl_init, sp_dgp, d1_vec, d2_vec, reps)[1]              
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
