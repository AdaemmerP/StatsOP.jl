export cl_gop


# --- Function to compute control limit for OPs --- #
function cl_gop(
  gop_dgp, lam, L0, cl_init, reps=10_000; chart_choice, jmin, jmax, verbose=false, d=1
)
  L1 = zeros(3)

  for j in jmin:jmax
    #println("Iteration j: $j")
    for dh in 1:80
      cl_init = cl_init + (-1)^j * dh / 10^j
      L1 = arl_gop_ic(
        gop_dgp, lam, cl, reps; chart_choice=chart_choice, d=d
      )
      if verbose
        println("cl = ", cl_init, "\t", "ARL = ", round(L1[1], digits=2), " ARM = ", L1[3])
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


# # --- Function to compute control limit for OPs --- #
# function cl_gop(
#   lam, L0, reps, gop_dgp, chart_choice, cl_init, jmin, jmax, verbose; d=1
# )
#   L1 = zeros(2)

#   # Copy starting value so we can compare at the end
#   cl_current = cl_init

#   for j in jmin:jmax
#     println("Iteration j: $j")

#     direction = (-1)^j  # alternate search direction
#     step = 1.0 / 10.0^j

#     for dh in 1:80
#       # compute candidate (do NOT overwrite cl_current yet)
#       cl_candidate = cl_current + direction * dh * step

#       # compute ARL for candidate limit
#       L1 = arl_gop(lam, cl_candidate, reps, gop_dgp, chart_choice; d=d)
#       ARL = L1[1]

#       if verbose
#         println("cl = ", cl_candidate, "\t", "ARL = ", ARL)
#       end

#       # if direction=+1 and ARL > L0 -> break (limit too loose)
#       # if direction=-1 and ARL < L0 -> break (limit too tight)
#       if (direction > 0 && ARL > L0) || (direction < 0 && ARL < L0)
#         # accept previous step as current
#         cl_current = cl_candidate
#         break
#       end
#     end
#   end

#   # final fine adjustment
#   if L1[1] < L0
#     cl_current += 1.0 / 10.0^jmax
#   end

#   return cl_current
# end

