
"""
    cl_sacf(
      lam, L0, sp_dgp::ICSP, cl_init, d1::Int, d2::Int, reps=10_000; 
      jmin=4, jmax=6, verbose::Bool=false
    )

Compute the control limit for the exponentially weighted moving average (EWMA) 
control chart for one delay (d₁-d₂) combination, using the spatial autocorrelation 
function (SACF) and an in-control spatial process (ICSP). 

The function returns the control limit for a given average run.
 
- `lam`: The smoothing parameter for the EWMA control chart.
- `L0`: The average run length (ARL) to use for the control limit.
- `sp_dgp`: The in-control spatial process (ICSP) to use for the control limit.
- `cl_init`: The initial control limit to use for the control limit.
- `d1`: The first (row) delay for the spatial process.
- `d2`: The second (column) delay for the spatial process.
- `reps`: The number of repetitions to use for the control limit.
- `jmin`: The minimum number of values to change after the decimal point in the control limit.
- `jmax`: The maximum number of values to change after the decimal point in the control limit.
- `verbose`: A boolean to indicate whether to print the control limit and ARL for each iteration.

```julia
#--- Example
# Parameters
lam = 0.1
L0 = 370
sp_dgp = ICSP(20, 20, Normal(0, 1))
cl_init = 0.5
d1 = 1
d2 = 1
reps = 10_000
jmin = 4
jmax = 6
```
"""
function cl_sacf(
  lam, L0, sp_dgp::ICSP, cl_init, d1::Int, d2::Int, reps=10_000;
  jmin=4, jmax=6, verbose=false
)

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

"""
    cl_sacf(
      lam, L0, sp_dgp::ICSP, cl_init, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000; 
      jmin=4, jmax=6, verbose::Bool=false
    )

Compute the control limit for BP-EWMA control chart for multiple delay (d₁-d₂) combinations, 
using the spatial autocorrelation function (SACF) and an in-control spatial process (ICSP). 
  
The function returns the control limit for a given average run.
  
- `lam`: The smoothing parameter for the EWMA control chart.
- `L0`: The average run length (ARL) to use for the control limit.
- `sp_dgp`: The in-control spatial process (ICSP) to use for the control limit.
- `cl_init`: The initial control limit to use for the control limit.
- `d1_vec`: The vector of first (row) delays for the spatial process.
- `d2_vec`: The vector of second (column) delays for the spatial process.
- `reps`: The number of repetitions to use for the control limit. 
"""
function cl_sacf(
  lam, L0, sp_dgp::ICSP, cl_init, d1_vec::Vector{Int}, d2_vec::Vector{Int}, reps=10_000;
  jmin=4, jmax=6, verbose=false
)

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
