export cl_acf


"""
  cl_acf(acf_dgp, lam, L0, cl_init, p_reps=10_000; jmin=4, jmax=6, verbose=false)

Function to compute the control limit for the ACF statistic by XXX.
  
- `lam::Float64`: Smoothing parameter for the EWMA statistic.
- `L0::Float64`: In-control ARL.
- `acf_dgp::Union{IC, AR1, TEAR1}`: DGP.  
- `cl_init::Float64`: Initial guess for the control limit.
- `p_reps::Int64`: Number of replications.
- `jmin::Int64`: Minimum number of iterations.

```julia
# Compute initial values via quantiles

cl_acf(0.1, 3.0, IC(Normal(0, 1)), cl_init, 10000)
```
"""
function cl_acf(
  acf_dgp, lam, L0, cl_init, reps=10_000; acf_version=1, jmin=4, jmax=6, verbose=false
)


  L1 = zeros(2)

  for j in jmin:jmax
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_acf(lam, cl, reps, acf_dgp, acf_version)
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", L1[1])
      end
      if (j % 2 == 1 && L1[1] < L0) || (j % 2 == 0 && L1[1] > L0)
        break
      end
    end
    cl_init = cl_init
  end

  if L1[1] < L0
    cl_init = cl_init + 1 / 10^jmax
  end
  return cl_init
end