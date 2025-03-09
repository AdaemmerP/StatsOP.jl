"""
    cl_op(op_dgp, lam, L0, cl_init, reps=10_000; chart_choice, jmin=4, jmax=6, verbose=false, d=1, ced=false, ad=100)

Function to compute the control limit using in-control processes using ordinal patterns.

- `op_dgp::ICTS.
- `lam::Float64`: Smoothing parameter for the EWMA statistic.
- `L0::Float64`: In-control ARL.
- `cl_init::Float64`: Initial guess for the control limit.
- `reps::Int64`: Number of replications.  
- `chart_choice::Int` 
  1. ``\\widehat{H}^{(d)}=-\\sum_{k=1}^{m!} \\hat{p}_k{ }^{(d)} \\ln \\hat{p}_k{ }^{(d)}``
  2. ``\\widehat{H}_{\\mathrm{ex}}^{(d)}=-\\sum_{k=1}^{m!}\\left(1-\\hat{p}_k{ }^{(d)}\\right) \\ln \\left(1-\\hat{p}_k{ }^{(d)}\\right)``
  3. ``\\widehat{\\Delta}^{(d)}=\\sum_{k=1}^{m!}\\left(\\hat{p}_k^{(d)}-1 / m!\\right)^2``
  4. ``\\hat{\\beta}^{(d)}=\\hat{p}_6^{(d)}-\\hat{p}_1^{(d)}``
  5. ``\\hat{\\tau}^{(d)}=\\hat{p}_6^{(d)}+\\hat{p}_1^{(d)}-\\frac{1}{3}``
  6. ``\\hat{\\delta}^{(d)}=\\hat{p}_4^{(d)}+\\hat{p}_5^{(d)}-\\hat{p}_3^{(d)}-\\hat{p}_2^{(d)}``

  The patterns are categorized as follows:

  ``
  \\qquad p_1 = (3,2,1);  \\quad p_2=(3,1,2);  \\quad p_3 = (2,3,1); 
  ``

  ``
  \\qquad p_4 = (1,3,2);  \\quad p_5 = (2,1,3);  \\quad p_ 6 = (1,2,3)
  ``
- `jmin::Int` Minimum number of decimals for final control limit to optimize.
- `jmax::Int` Maximum number of decimals for final control limit to optimize.
- `verbose::Bool=false` Print intermediate results?
- `d::Union{Int,Vector{Int}}=1`: Delay value or vector. 
- `ced::Bool=false`: Use conditional expected delay? Default is false.
- `ad::Int=100`: Number of iterations for ced.
"""
function cl_op(
  op_dgp, lam, L0, cl_init, reps=10_000; 
  chart_choice, jmin=4, jmax=6, verbose=false, d=1, ced=false, ad=100
  )
         
  L1 = zeros(2)
  
  for j in jmin:jmax
    for dh in 1:80

      if (chart_choice == 1 || chart_choice == 2)
        cl_init = cl_init - (-1)^j * dh / 10^j
      else
        cl_init = cl_init + (-1)^j * dh / 10^j
      end
      L1 = arl_op(lam, cl_init, op_dgp, reps; chart_choice=chart_choice, d=d, ced=ced, ad=ad) 

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

