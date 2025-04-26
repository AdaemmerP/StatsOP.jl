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
@everywhere using Random
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

# Method to compute boolean of rejections for in-control processes
@everywhere function compute_reject_sop(
  crit_sop_1,
  crit_sop_2,
  crit_sop_3,
  crit_sop_4,
  crit_sacf,
  dist_error::UnivariateDistribution,
  data,
  X_centered,
  d1,
  d2
)

  # extract m and n
  m = size(data, 1) - 1
  n = size(data, 2) - 1

  rand!(dist_error, data)

  # compute sacf statistic
  X_centered .= data .- mean(data)
  sacf_stat = sacf(X_centered, d1, d2)
  check_crit_sacf = (abs(sacf_stat) > crit_sacf)

  # check whether to add noise
  if dist_error isa DiscreteUnivariateDistribution
    for j in axes(data, 2)
      for i in axes(data, 1)
        data[i, j] = data[i, j] + rand()
      end
    end
  end

  # compute test statistic  
  sop_stat_1 = stat_sop(data, d1, d2; chart_choice=1, add_noise=false)
  sop_stat_2 = stat_sop(data, d1, d2; chart_choice=2, add_noise=false)
  sop_stat_3 = stat_sop(data, d1, d2; chart_choice=3, add_noise=false)
  sop_stat_4 = stat_sop(data, d1, d2; chart_choice=4, add_noise=false)

  # for sops
  check_crit_sop_1 = (abs(sop_stat_1) > crit_sop_1)
  check_crit_sop_2 = (abs(sop_stat_2) > crit_sop_2)
  check_crit_sop_3 = (abs(sop_stat_3) > crit_sop_3)
  check_crit_sop_4 = (abs(sop_stat_4) > crit_sop_4)

  return (check_crit_sop_1, check_crit_sop_2, check_crit_sop_3, check_crit_sop_4, check_crit_sacf)

end

# Function to compute boolean of rejections for SAR (1,1), SINAR (1,1), SQMA (1,1), SQINMA (1,1), BSQMA(1,1)
@everywhere function compute_reject_sop(
  crit_sop_1,
  crit_sop_2,
  crit_sop_3,
  crit_sop_4,
  crit_sacf,
  dgp::Union{SAR11,SINAR11,SQMA11,SQINMA11, BSQMA11},
  data,
  mat,
  mat_ao,
  mat_ma,
  X_centered
)

  # extract m and n
  m = dgp.M_rows - 1
  n = dgp.N_cols - 1
  dist_error = dgp.dist
  dist_ao = dgp.dist_ao

  fill!(mat, 0)

  if dgp isa SAR11 || dgp isa SINAR11
    init_mat!(dgp, dist_error, mat)
  end

  data .= fill_mat_dgp_sop!(dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)

  # compute sacf statistic
  X_centered .= data .- mean(data)
  sacf_stat = sacf(X_centered, d1, d2)
  check_crit_sacf = (abs(sacf_stat) > crit_sacf)

  # check whether to add noise
  if dist_error isa DiscreteUnivariateDistribution
    for j in axes(data, 2)
      for i in axes(data, 1)
        data[i, j] = data[i, j] + rand()
      end
    end
  end

  # compute test statistic  
  sop_stat_1 = stat_sop(data, d1, d2; chart_choice=1, add_noise=false)
  sop_stat_2 = stat_sop(data, d1, d2; chart_choice=2, add_noise=false)
  sop_stat_3 = stat_sop(data, d1, d2; chart_choice=3, add_noise=false)
  sop_stat_4 = stat_sop(data, d1, d2; chart_choice=4, add_noise=false)

  # for sops
  check_crit_sop_1 = (abs(sop_stat_1) > crit_sop_1)
  check_crit_sop_2 = (abs(sop_stat_2) > crit_sop_2)
  check_crit_sop_3 = (abs(sop_stat_3) > crit_sop_3)
  check_crit_sop_4 = (abs(sop_stat_4) > crit_sop_4)

  return (check_crit_sop_1, check_crit_sop_2, check_crit_sop_3, check_crit_sop_4, check_crit_sacf)

end

# Function to compute boolean of rejections for SAR(1)
@everywhere function compute_reject_sop(
  crit_sop_1,
  crit_sop_2,
  crit_sop_3,
  crit_sop_4,
  crit_sacf,
  dgp::SAR1,
  dist_error,
  dist_ao,
  data,
  mat,
  mat_ao,
  vec_ar,
  vec_ar2,
  mat2,
  X_centered)

  # extract m and n
  m = dgp.M_rows - 1
  n = dgp.N_cols - 1

  data .= fill_mat_dgp_sop!(dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2, mat2)

  # compute test statistic
  sop_stat_1 = stat_sop(data, d1, d2; chart_choice=1, add_noise=false)
  sop_stat_2 = stat_sop(data, d1, d2; chart_choice=2, add_noise=false)
  sop_stat_3 = stat_sop(data, d1, d2; chart_choice=3, add_noise=false)
  sop_stat_4 = stat_sop(data, d1, d2; chart_choice=4, add_noise=false)

  # for sops
  check_crit_sop_1 = (abs(sop_stat_1) > crit_sop_1)
  check_crit_sop_2 = (abs(sop_stat_2) > crit_sop_2)
  check_crit_sop_3 = (abs(sop_stat_3) > crit_sop_3)
  check_crit_sop_4 = (abs(sop_stat_4) > crit_sop_4)

  # for sacf
  X_centered .= data .- mean(data)
  sacf_stat = sacf(X_centered, d1, d2)
  check_crit_sacf = (abs(sacf_stat) > crit_sacf)

  return (check_crit_sop_1, check_crit_sop_2, check_crit_sop_3, check_crit_sop_4, check_crit_sacf)

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

#-------------------------------------------#
#             Table B.1
#-------------------------------------------#
dist_errors = [Normal(0, 1), Poisson(0.5), Poisson(1), Poisson(2), Poisson(5), Poisson(10)]
results_mat = zeros(4, 5, length(dist_errors))

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.1")
println("#-------------------------------------------#")
for (i, dist_error) in enumerate(dist_errors)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]

    # pre-allocate SACF
    data = zeros(M, N)
    cdata = similar(data)

    # Compute critical values    
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dist_error,
        data,
        cdata,
        d1,
        d2
      )
    end

    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    println("m: ", M - 1, " ; n: ", N - 1, " ; dist_error: ", dist_error)
  end
end

# Print results (3rd dimension is for DGP)
results_mat

#-------------------------------------------#
#             Table B.2
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
results_mat = zeros(4, 5, length(params_dgp))

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.2")
println("#-------------------------------------------#")
for (i, dgp_params) in enumerate(params_dgp)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # create spatial dgp
    dgp_sar11 = SAR11(dgp_params, M, N, dist_error, dist_ao, prerun)

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = zeros(M + prerun, N + prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp_sar11,
        data,
        mat,
        mat_ao,
        mat_ma,
        X_centered)
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_params)
  end
end

# Print results
results_mat


#-------------------------------------------#
#             Table B.3
#-------------------------------------------#
dist_ao = nothing

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
lambdas_vec = [0.5; 1; 2; 5; 10]
results_mat = zeros(4, 5, length(params_dgp), length(lambdas_vec))

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.3")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_params) in enumerate(params_dgp)
  for (j, lambda) in enumerate(lambdas_vec)

    dist_error = Poisson(lambda)

    for (k, MN) in enumerate(MN_vec)

      # Get m and n
      M = MN[1]
      N = MN[2]
      m = M - 1
      n = N - 1

      # pre-allocate SACF
      data = zeros(M, N)
      X_centered = similar(data)

      # pre-allocate SOPs
      mat = zeros(M + prerun, N + prerun)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
      crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
      crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
      crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
      crit_sacf = crit_val_sacf(M, N, alpha)

      # create spatial dgp
      dgp_sinar11 = SINAR11(dgp_params, M, N, dist_error, dist_ao, prerun)

      get_checks = pmap(1:reps) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp_sinar11,
          data,
          mat,
          mat_ao,
          mat_ma,
          X_centered)
      end

      # fill matrix
      results_mat[k, 1, i, j] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
      results_mat[k, 2, i, j] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
      results_mat[k, 3, i, j] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
      results_mat[k, 4, i, j] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
      results_mat[k, 5, i, j] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

      # print progress
      println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_params, " ; lambda: ", lambda)

    end
  end
end

# Show results
results_mat[:, :, 1, :]
results_mat[:, :, 2, :]
results_mat[:, :, 3, :]
results_mat[:, :, 4, :]

#-------------------------------------------#
# Table B.4.1 -> Choose 5 or 10 for BinomialC()
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialC(0.1, 10)
params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
results_mat = zeros(4, 5, length(params_dgp))

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.4.1")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_params) in enumerate(params_dgp)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = zeros(M + prerun, N + prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    # create spatial dgp
    dgp_sar11_outl = SAR11(dgp_params, M, N, dist_error, dist_ao, prerun)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp_sar11_outl,
        data,
        mat,
        mat_ao,
        mat_ma,
        X_centered)
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_params)

  end
end

#-------------------------------------------#
#             Table B.4.2
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialCvec(0.1, [-10; 10])
params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
results_mat = zeros(4, 5, length(params_dgp))

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.4.2")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_params) in enumerate(params_dgp)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = zeros(M + prerun, N + prerun)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    # create spatial dgp
    dgp_sar11_outl = SAR11(dgp_params, M, N, dist_error, dist_ao, prerun)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp_sar11_outl,
        data,
        mat,
        mat_ao,
        mat_ma,
        X_centered)
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_params)

  end
end

#---------------------------------------------------------------#
#  Table B.5 + B.6 -> Set multiplication factor to either 5 or 10
#---------------------------------------------------------------#
lambdas_vec = [0.5; 1; 2; 5; 10]
params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
results_mat = zeros(4, 5, length(params_dgp), length(lambdas_vec))

# Loop to compute values
println("#-------------------------------------------#")
println("#   Table B.5 -> PoiBin(0.1, 5 * lambda)")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_params) in enumerate(params_dgp)
  for (j, lambda) in enumerate(lambdas_vec)

    dist_error = Poisson(lambda)
    dist_ao = PoiBin(0.1, 5 * lambda)

    for (k, MN) in enumerate(MN_vec)

      # Get m and n
      M = MN[1]
      N = MN[2]
      m = M - 1
      n = N - 1

      # pre-allocate SACF
      data = zeros(M, N)
      X_centered = similar(data)

      # pre-allocate SOPs
      mat = zeros(M + prerun, N + prerun)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
      crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
      crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
      crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
      crit_sacf = crit_val_sacf(M, N, alpha)

      # create spatial dgp
      dgp_sinar11_outl = SINAR11(dgp_params, M, N, dist_error, dist_ao, prerun)

      get_checks = pmap(1:reps) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp_sinar11_outl,
          data,
          mat,
          mat_ao,
          mat_ma,
          X_centered)
      end

      # fill matrix
      results_mat[k, 1, i, j] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
      results_mat[k, 2, i, j] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
      results_mat[k, 3, i, j] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
      results_mat[k, 4, i, j] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
      results_mat[k, 5, i, j] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

      # print progress
      println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_params, " ; lambda: ", lambda)

    end
  end
end

# Show results
results_mat[:, :, 1, :]
results_mat[:, :, 2, :]
results_mat[:, :, 3, :]
results_mat[:, :, 4, :]


#-------------------------------------------#
#             Table B.7
#-------------------------------------------#
dist_ao = nothing
params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
mn_vec = [(10, 10); (15, 15); (20, 20); (40, 25)]
lambdas_vec = [0.5; 1; 2; 5; 10]
results_mat = zeros(4, 5, length(params_dgp), length(lambdas_vec))

println("#-------------------------------------------#")
println("#              Table B.7")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_params) in enumerate(params_dgp)
  for (j, lambda) in enumerate(lambdas_vec)

    dist_error = ZIP(lambda, 0.9)

    for (k, MN) in enumerate(MN_vec)

      # Get m and n
      M = MN[1]
      N = MN[2]
      m = M - 1
      n = N - 1

      # pre-allocate SACF
      data = zeros(M, N)
      X_centered = similar(data)

      # pre-allocate SOPs
      mat = zeros(M + prerun, N + prerun)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
      crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
      crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
      crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
      crit_sacf = crit_val_sacf(M, N, alpha)

      # create spatial dgp
      dgp_sinar11_outl = SINAR11(dgp_params, M, N, dist_error, dist_ao, prerun)

      get_checks = pmap(1:reps) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp_sinar11_outl,
          data,
          mat,
          mat_ao,
          mat_ma,
          X_centered)
      end

      # fill matrix
      results_mat[k, 1, i, j] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
      results_mat[k, 2, i, j] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
      results_mat[k, 3, i, j] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
      results_mat[k, 4, i, j] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
      results_mat[k, 5, i, j] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

      # print progress
      println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_params, " ; lambda: ", lambda)

    end
  end
end

# Show results
results_mat[:, :, 1, :]
results_mat[:, :, 2, :]
results_mat[:, :, 3, :]
results_mat[:, :, 4, :]

#-------------------------------------------#
#             Table B.8
#-------------------------------------------#
dist_ao = nothing
dist_error = Normal(0, 1)
eps_params = [(2, 2, 2); (1, 1, 2); (2, 1, 2); (2, 1, 1)]
results_mat = zeros(4, 5, length(eps_params))

println("#-------------------------------------------#")
println("#   Table B.8 -> PoiBin(0.1, 5 * lambda)")
println("#-------------------------------------------#")
# Loop to compute values
for (i, eps_param) in enumerate(eps_params)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # create spatial dgp
    dgp_sqma11 = SQMA11((0.8, 0.8, 0.8), eps_param, M, N, dist_error, dist_ao)

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = zeros(M + 1, N + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp_sqma11,
        data,
        mat,
        mat_ao,
        mat_ma,
        X_centered)
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    # print progress
    println("m: ", M - 1, " ; n: ", N - 1, " ; eps_param: ", eps_param)

  end
end

#-------------------------------------------#
#             Table B.9
#-------------------------------------------#
dist_ao = nothing
lambdas_vec = [0.5; 1; 2; 5; 10]
eps_params = [(2, 2, 2); (1, 1, 2); (2, 1, 2); (2, 1, 1)]
results_mat = zeros(4, 5, length(eps_params), length(lambdas_vec))

println("#-------------------------------------------#")
println("#   Table B.9 -> PoiBin(0.1, 5 * lambda)")
println("#-------------------------------------------#")
# Loop to compute values
for (i, eps_param) in enumerate(eps_params)
  for (j, lambda) in enumerate(lambdas_vec)

    dist_error = Poisson(lambda)

    for (k, MN) in enumerate(MN_vec)

      # Get m and n
      M = MN[1]
      N = MN[2]
      m = M - 1
      n = N - 1

      # create spatial dgp
      dgp_sqma11 = SQINMA11((0.8, 0.8, 0.8), eps_param, M, N, dist_error, dist_ao)

      # pre-allocate SACF
      data = zeros(M, N)
      X_centered = similar(data)

      # pre-allocate SOPs
      mat = zeros(M + 1, N + 1)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
      crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
      crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
      crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
      crit_sacf = crit_val_sacf(M, N, alpha)

      get_checks = pmap(1:reps) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp_sqma11,
          data,
          mat,
          mat_ao,
          mat_ma,
          X_centered)
      end

      # fill matrix
      results_mat[k, 1, i, j] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
      results_mat[k, 2, i, j] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
      results_mat[k, 3, i, j] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
      results_mat[k, 4, i, j] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
      results_mat[k, 5, i, j] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

      # print progress
      println("m: ", M - 1, " ; n: ", N - 1, " ; eps_param: ", eps_param, " ; lambda: ", lambda)

    end

  end
end

# Show results
results_mat[:, :, 1, :]
results_mat[:, :, 2, :]
results_mat[:, :, 3, :]
results_mat[:, :, 4, :]


#-------------------------------------------#
#             Table B.10.1
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing
dgp_params = [(0.1, 0.1, 0.1, 0.1); (0.05, 0.05, 0.15, 0.15); (0.05, 0.15, 0.05, 0.15)]
margin = 20
results_mat = zeros(4, 5, length(dgp_params))

println("#-------------------------------------------#")
println("#   Table B.10.1")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_param) in enumerate(dgp_params)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # create spatial dgp
    dgp = SAR1(dgp_param, M, N, dist_error, dist_ao, margin)

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = build_sar1_matrix(dgp) # will be done only once
    mat_ao = zeros((M + 2 * margin), (N + 2 * margin))
    vec_ar = zeros((M + 2 * margin) * (N + 2 * margin))
    vec_ar2 = similar(vec_ar)
    mat2 = similar(mat_ao)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        vec_ar,
        vec_ar2,
        mat2,
        X_centered)
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    # print progress
    println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_param)

  end
end

#-------------------------------------------#
#             Table B.10.2
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialC(0.1, 5)
dgp_params = [(0.1, 0.1, 0.1, 0.1); (0.05, 0.05, 0.15, 0.15); (0.05, 0.15, 0.05, 0.15)]
margin = 20
results_mat = zeros(4, 5, length(dgp_params))

println("#-------------------------------------------#")
println("#   Table B.10.2")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_param) in enumerate(dgp_params)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # create spatial dgp
    dgp = SAR1(dgp_param, M, N, dist_error, dist_ao, margin)

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = build_sar1_matrix(dgp) # will be done only once
    mat_ao = zeros((M + 2 * margin), (N + 2 * margin))
    vec_ar = zeros((M + 2 * margin) * (N + 2 * margin))
    vec_ar2 = similar(vec_ar)
    mat2 = similar(mat_ao)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        vec_ar,
        vec_ar2,
        mat2,
        X_centered)
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    # print progress
    println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_param)

  end
end

# Show results
results_mat

#-------------------------------------------#
#             Table B.10.3
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialCvec(0.1, [-5; 5])
dgp_params = [(0.1, 0.1, 0.1, 0.1); (0.05, 0.05, 0.15, 0.15); (0.05, 0.15, 0.05, 0.15)]
results_mat = zeros(4, 5, length(dgp_params))

println("#-------------------------------------------#")
println("#   Table B.10.3")
println("#-------------------------------------------#")
# Loop to compute values
for (i, dgp_param) in enumerate(dgp_params)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # create spatial dgp
    dgp = SAR1(dgp_param, M, N, dist_error, dist_ao, margin)

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = build_sar1_matrix(dgp) # will be done only once
    mat_ao = zeros((M + 2 * margin), (N + 2 * margin))
    vec_ar = zeros((M + 2 * margin) * (N + 2 * margin))
    vec_ar2 = similar(vec_ar)
    mat2 = similar(mat_ao)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        vec_ar,
        vec_ar2,
        mat2,
        X_centered)
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    # print progress
    println("m: ", M - 1, " ; n: ", N - 1, " ; params: ", dgp_param)

  end
end

# Show results
results_mat


#-------------------------------------------#
#             Table B.11
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing
eps_params = [(2, 2, 2, 2); (2, 1, 2, 1); (2, 2, 1, 1)]
results_mat = zeros(4, 5, length(eps_params))

println("#-------------------------------------------#")
println("#              Table B.11")
println("#-------------------------------------------#")
# Loop to compute values
for (i, eps_param) in enumerate(eps_params)
  for (j, MN) in enumerate(MN_vec)

    # Get m and n
    M = MN[1]
    N = MN[2]
    m = M - 1
    n = N - 1

    # pre-allocate SACF
    data = zeros(M, N)
    X_centered = similar(data)

    # pre-allocate SOPs
    mat = zeros(M + 1, N + 1)
    mat_ma = zeros(M + 2, N + 2) # one extra row and column for "forward looking"
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=1, approximate=false)
    crit_sop_2 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=2, approximate=false)
    crit_sop_3 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=3, approximate=false)
    crit_sop_4 = crit_val_sop(M, N, alpha, d1, d2; chart_choice=4, approximate=false)
    crit_sacf = crit_val_sacf(M, N, alpha)

    # create spatial dgp
    dgp_bsqma11 = BSQMA11((0.8, 0.8, 0.8, 0.8), eps_param, M, N, dist_error, dist_ao)

    get_checks = pmap(1:reps) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp_bsqma11,
        data,
        mat,
        mat_ao,
        mat_ma,
        X_centered
      )
    end

    # fill matrix
    results_mat[j, 1, i] = round(sum(getindex.(get_checks, 1) / reps); digits=4)
    results_mat[j, 2, i] = round(sum(getindex.(get_checks, 2) / reps); digits=4)
    results_mat[j, 3, i] = round(sum(getindex.(get_checks, 3) / reps); digits=4)
    results_mat[j, 4, i] = round(sum(getindex.(get_checks, 4) / reps); digits=4)
    results_mat[j, 5, i] = round(sum(getindex.(get_checks, 5) / reps); digits=4)

    # print progress
    println("m: ", M - 1, " ; n: ", N - 1, " ; eps_param: ", eps_param)
    
  end
end

# Show results
results_mat