

# --- Function to compute control limit for OPs --- #
function cl_kappa(
  dgp, lam, L0, cl_init, reps=10_000; chart_choice, jmin, jmax, verbose=false
)
  L1 = zeros(3)

  for j in jmin:jmax
    #println("Iteration j: $j")
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_kappa_ic(
        dgp, lam, cl_init, reps; chart_choice=chart_choice
      )
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", round(L1[1], digits=2))
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