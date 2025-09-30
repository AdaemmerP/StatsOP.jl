# Packages to use
# Change to current directory
cd(@__DIR__)
using Pkg
Pkg.activate("../../.")
using Random
using LinearAlgebra
using Distributed
using OrdinalPatterns

addprocs(5)
@everywhere using OrdinalPatterns

# Function to compute boolean of rejections for SAR (1,1), SINAR (1,1), SQMA (1,1), SQINMA (1,1), BSQMA(1,1)
@everywhere function compute_reject_sop(
  crit_sop_1,
  crit_sop_2,
  crit_sop_3,
  refinement,
  dgp::Union{SAR11,SINAR11,SQMA11,SQINMA11,BSQMA11},
  data,
  mat,
  mat_ao,
  mat_ma
)

  # extract m and n
  m = dgp.M_rows - 1
  n = dgp.N_cols - 1
  dist_error = dgp.dist
  dist_ao = dgp.dist_ao

  d1 = d2 = 1

  fill!(mat, 0)

  if dgp isa SAR11 || dgp isa SINAR11
    init_mat!(dgp, dist_error, mat)
  end

  data .= fill_mat_dgp_sop!(dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

  # check whether to add noise
  if dist_error isa DiscreteUnivariateDistribution
    for j in axes(data, 2)
      for i in axes(data, 1)
        data[i, j] = data[i, j] + rand()
      end
    end
  end

  # compute test statistic  
  sop_stat_1 = stat_sop(data, d1, d2; chart_choice=5, refinement=refinement, add_noise=false)
  sop_stat_2 = stat_sop(data, d1, d2; chart_choice=6, refinement=refinement, add_noise=false)
  sop_stat_3 = stat_sop(data, d1, d2; chart_choice=7, refinement=refinement, add_noise=false)

  # for sops
  check_crit_sop_1 = (sop_stat_1 > crit_sop_1)
  check_crit_sop_2 = (sop_stat_2 > crit_sop_2)
  check_crit_sop_3 = (sop_stat_3 > crit_sop_3)
  return (check_crit_sop_1, check_crit_sop_2, check_crit_sop_3)

end


#-------------------------------------------#
# Set parameters
#-------------------------------------------#
alpha = 0.05
reps = 10_000
prerun = 100
margin = 20
d1 = d2 = 1
MN_vec = [(11, 11); (16, 16); (21, 21); (41, 26)]

# Loop to compute values
#-------------------------------------------#
#             Table B.2
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
results_mat = zeros(4, 12, length(params_dgp))
refinements = 0#[0, 1, 2, 3]
chart_choices = 1:4#5:7

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.2")
println("#-------------------------------------------#")
for (i, dgp_params) in enumerate(params_dgp)
  for (j, MN) in enumerate(MN_vec)
    col_index = 1
    for (k, refinement) in enumerate(refinements)      
      for chart_choice in chart_choices

        # Get m and n
        M = MN[1]
        N = MN[2]
        m = M - 1
        n = N - 1

        # create spatial dgp
        dgp_sar11 = SAR11(dgp_params, M, N, dist_error, dist_ao, prerun)

        # pre-allocate SACF
        data = zeros(M, N)
        # X_centered = similar(data)

        # pre-allocate SOPs
        mat = zeros(M + prerun, N + prerun)
        mat_ma = similar(mat)
        mat_ao = similar(mat)

        # Compute critical values
        crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=chart_choice, refinement=refinement, approximate=false)
        crit_sop_2 = crit_sop_1
        crit_sop_3 = crit_sop_1

        get_checks = pmap(1:reps) do i
          compute_reject_sop(
            crit_sop_1,
            crit_sop_2,
            crit_sop_3,
            refinement,
            dgp_sar11,
            data,
            mat,
            mat_ao,
            mat_ma)
        end

        # fill matrix
        results_mat[j, col_index, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
        results_mat[j, col_index, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
        results_mat[j, col_index, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)

        println("MN: ", MN, " | refinement: ", refinement, " | dgp_params: ", dgp_params)
        col_index += 1
      end
    end
  end
end