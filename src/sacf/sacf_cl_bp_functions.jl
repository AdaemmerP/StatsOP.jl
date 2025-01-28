
"""
    cl_sacf_bp(
      lam, L0, sp_dgp::ICSP, cl_init, w::Int, reps=10_000; 
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
function cl_sacf_bp(
  sp_dgp::ICSP, lam, L0, cl_init, w::Int, reps=10_000;
  jmin=4, jmax=6, verbose=false
)

  L1 = 0.0
  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_sacf_bp(sp_dgp, lam, cl_init, w::Int, reps)[1]
      
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
