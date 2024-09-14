# Packages to use
using Pkg
Pkg.activate()
using BenchmarkTools

Pkg.activate(".")
using Random
using LinearAlgebra
using Statistics
#using Combinatorics
using Distributions
using Distributed
using OrdinalPatterns

addprocs(10)
@everywhere using OrdinalPatterns
@everywhere using Distributions
@everywhere using Random
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

# Function to compute boolean of rejections for in-control processes
@everywhere function compute_reject_sop(
  crit_sop_1,
  crit_sop_2,
  crit_sop_3,
  crit_sop_4,
  crit_sacf,
  dist_error::UnivariateDistribution,
  data,
  cdata,
  cx_t_cx_t1,
  cdata_sq
)

  # extract m and n
  m = size(data, 1) - 1
  n = size(data, 2) - 1

  rand!(dist_error, data)

  # compute sacf statistic
  sacf_stat = sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)
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
  sop_stat_1 = stat_sop(data, 1)
  sop_stat_2 = stat_sop(data, 2)
  sop_stat_3 = stat_sop(data, 3)
  sop_stat_4 = stat_sop(data, 4)

  # for sops
  check_crit_sop_1 = (sqrt(m * n) * abs(sop_stat_1) > crit_sop_1)
  check_crit_sop_2 = (sqrt(m * n) * abs(sop_stat_2) > crit_sop_2)
  check_crit_sop_3 = (sqrt(m * n) * abs(sop_stat_3) > crit_sop_3)
  check_crit_sop_4 = (sqrt(m * n) * abs(sop_stat_4) > crit_sop_4)

  return (check_crit_sop_1, check_crit_sop_2, check_crit_sop_3, check_crit_sop_4, check_crit_sacf)

end

# Function to compute boolean of rejections for SAR (1,1), SINAR (1,1), SQMA (1,1), SQINMA (1,1)
@everywhere function compute_reject_sop(
  crit_sop_1,
  crit_sop_2,
  crit_sop_3,
  crit_sop_4,
  crit_sacf,
  dgp,
  dgp_params,
  dist_error,
  dist_ao,
  data,
  mat,
  mat_ao,
  mat_ma,
  cdata,
  cx_t_cx_t1,
  cdata_sq)

  # extract m and n
  m = dgp.m
  n = dgp.n

  fill!(mat, 0)
  init_mat!(dgp, dist_error, dgp_params, mat)
  data .= (fill_mat_dgp_sop!(dgp, dist_error, dist_ao, mat, mat_ao, mat_ma))


  # compute sacf statistic
  sacf_stat = sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)
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
  sop_stat_1 = stat_sop(data, 1)
  sop_stat_2 = stat_sop(data, 2)
  sop_stat_3 = stat_sop(data, 3)
  sop_stat_4 = stat_sop(data, 4)

  # for sops
  check_crit_sop_1 = (sqrt(m * n) * abs(sop_stat_1) > crit_sop_1)
  check_crit_sop_2 = (sqrt(m * n) * abs(sop_stat_2) > crit_sop_2)
  check_crit_sop_3 = (sqrt(m * n) * abs(sop_stat_3) > crit_sop_3)
  check_crit_sop_4 = (sqrt(m * n) * abs(sop_stat_4) > crit_sop_4)

  return (check_crit_sop_1, check_crit_sop_2, check_crit_sop_3, check_crit_sop_4, check_crit_sacf)

end

# Function to compute boolean of rejections for SAR (1)
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
  cdata,
  cx_t_cx_t1,
  cdata_sq)

  # extract m and n
  m = dgp.m
  n = dgp.n

  data .= fill_mat_dgp_sop!(dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2)

  # compute test statistic
  sop_stat_1 = stat_sop(data, 1)
  sop_stat_2 = stat_sop(data, 2)
  sop_stat_3 = stat_sop(data, 3)
  sop_stat_4 = stat_sop(data, 4)

  # for sops
  check_crit_sop_1 = (sqrt(m * n) * abs(sop_stat_1) > crit_sop_1)
  check_crit_sop_2 = (sqrt(m * n) * abs(sop_stat_2) > crit_sop_2)
  check_crit_sop_3 = (sqrt(m * n) * abs(sop_stat_3) > crit_sop_3)
  check_crit_sop_4 = (sqrt(m * n) * abs(sop_stat_4) > crit_sop_4)

  # for sacf
  sacf_stat = sacf_11(data, cdata, cx_t_cx_t1, cdata_sq)
  check_crit_sacf = (abs(sacf_stat) > crit_sacf)

  return (check_crit_sop_1, check_crit_sop_2, check_crit_sop_3, check_crit_sop_4, check_crit_sacf)

end

#-------------------------------------------#
# Set parameters
#-------------------------------------------#
alpha = 0.05
N = 10_00
prerun = 100
margin = 20
mn_vec = [(10, 10); (15, 15); (20, 20); (40, 25)]

#-------------------------------------------#
#             Table B.1
#-------------------------------------------#
dist_errors = [Normal(0, 1), Poisson(0.5), Poisson(1), Poisson(2), Poisson(5), Poisson(10)]

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.1")
println("#-------------------------------------------#")
for dist_error in dist_errors
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1, false)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2, false)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3, false)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4, false)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    get_checks = pmap(1:N) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dist_error,
        data,
        cdata,
        cx_t_cx_t1,
        cdata_sq
      )
    end
    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("dist_error ", dist_error)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end

#-------------------------------------------#
#             Table B.2
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.2")
println("#-------------------------------------------#")
for dgp_params in params_dgp
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = zeros(m + prerun + 1, n + prerun + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    # create spatial dgp
    dgp = SAR11(dgp_params, m, n, 100)

    get_checks = pmap(1:N) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dgp_params,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        mat_ma,
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("params: ", dgp_params)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end


#-------------------------------------------#
#             Table B.3
#-------------------------------------------#
dist_ao = nothing

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
lambdas_vec = [0.5; 1; 2; 5; 10]

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.3")
println("#-------------------------------------------#")

# Loop to compute values
for dgp_params in params_dgp
  for lambda in lambdas_vec

    dist_error = Poisson(lambda)

    for mn in mn_vec

      # Get m and n
      m = mn[1]
      n = mn[2]

      # pre-allocate SACF
      data = zeros(m + 1, n + 1)
      cdata = similar(data)
      cdata_sq = similar(data)
      cx_t_cx_t1 = zeros(m, n)

      # pre-allocate SOPs
      mat = zeros(m + prerun + 1, n + prerun + 1)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(m, n, alpha, 1)
      crit_sop_2 = crit_val_sop(m, n, alpha, 2)
      crit_sop_3 = crit_val_sop(m, n, alpha, 3)
      crit_sop_4 = crit_val_sop(m, n, alpha, 4)
      crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

      # create spatial dgp
      dgp = SINAR11(dgp_params, m, n, 100)

      get_checks = pmap(1:N) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp,
          dgp_params,
          dist_error,
          dist_ao,
          data,
          mat,
          mat_ao,
          mat_ma,
          cdata,
          cx_t_cx_t1,
          cdata_sq)
      end

      println("#--------------------------------------------------------#")
      println("m: ", m)
      println("n: ", n)
      println("lambda: ", lambda)
      println("params: ", dgp_params)
      round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
    end
  end
end

#-------------------------------------------#
# Table B.4.1 -> Choose 5 or 10 for BinomialC()
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialC(0.1, 10)

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.4.1")
println("#-------------------------------------------#")
# Loop to compute values
for dgp_params in params_dgp
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = zeros(m + prerun + 1, n + prerun + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    # create spatial dgp
    dgp = SAR11(dgp_params, m, n, 100)

    # Make chunks for separate tasks (based on number of threads)        
    #chunks = Iterators.partition(1:N, div(N, Threads.nthreads())) 

    get_checks = pmap(1:N) do i
      #Threads.@spawn 
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dgp_params,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        mat_ma,
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("params: ", dgp_params)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end

#-------------------------------------------#
#             Table B.4.2
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialCvec(0.1, [-10; 10])

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]

# Loop to compute values
println("#-------------------------------------------#")
println("#             Table B.4.2")
println("#-------------------------------------------#")
# Loop to compute values
for dgp_params in params_dgp
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = zeros(m + prerun + 1, n + prerun + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    # create spatial dgp
    dgp = SAR11(dgp_params, m, n, 100)

    # Make chunks for separate tasks (based on number of threads)        
    #chunks = Iterators.partition(1:N, div(N, Threads.nthreads())) 

    get_checks = pmap(1:N) do i
      #Threads.@spawn 
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dgp_params,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        mat_ma,
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("params: ", dgp_params)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end

#---------------------------------------------------------------#
#  Table B.5 + 6 -> Set multiplication factor to either 5 or 10
#---------------------------------------------------------------#
lambdas_vec = [0.5; 1; 2; 5; 10]
params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]

# Loop to compute values
println("#-------------------------------------------#")
println("#   Table B.5 -> PoiBin(0.1, 5 * lambda)")
println("#-------------------------------------------#")
# Loop to compute values
for dgp_params in params_dgp
  for lambda in lambdas_vec

    dist_error = Poisson(lambda)
    dist_ao = PoiBin(0.1, 10 * lambda)

    for mn in mn_vec

      # Get m and n
      m = mn[1]
      n = mn[2]

      # pre-allocate SACF
      data = zeros(m + 1, n + 1)
      cdata = similar(data)
      cdata_sq = similar(data)
      cx_t_cx_t1 = zeros(m, n)

      # pre-allocate SOPs
      mat = zeros(m + prerun + 1, n + prerun + 1)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(m, n, alpha, 1)
      crit_sop_2 = crit_val_sop(m, n, alpha, 2)
      crit_sop_3 = crit_val_sop(m, n, alpha, 3)
      crit_sop_4 = crit_val_sop(m, n, alpha, 4)
      crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

      # create spatial dgp
      dgp = SINAR11(dgp_params, m, n, 100)

      get_checks = pmap(1:N) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp,
          dgp_params,
          dist_error,
          dist_ao,
          data,
          mat,
          mat_ao,
          mat_ma,
          cdata,
          cx_t_cx_t1,
          cdata_sq)
      end

      println("#--------------------------------------------------------#")
      println("m: ", m)
      println("n: ", n)
      println("lambda: ", lambda)
      println("params: ", dgp_params)
      round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
    end
  end
end

#-------------------------------------------#
#             Table B.7
#-------------------------------------------#
dist_ao = nothing

params_dgp = [(0.1, 0.1, 0.1); (0.2, 0.2, 0.2); (0.2, 0.2, 0.5); (0.4, 0.3, 0.1)]
mn_vec = [(10, 10); (15, 15); (20, 20); (40, 25)]
lambdas_vec = [0.5; 1; 2; 5; 10]

println("#-------------------------------------------#")
println("#              Table B.7")
println("#-------------------------------------------#")
# Loop to compute values
for dgp_params in params_dgp
  for lambda in lambdas_vec

    # create dist error
    dist_error = ZIP(lambda, 0.9)

    for mn in mn_vec

      # Get m and n
      m = mn[1]
      n = mn[2]

      # pre-allocate SACF
      data = zeros(m + 1, n + 1)
      cdata = similar(data)
      cdata_sq = similar(data)
      cx_t_cx_t1 = zeros(m, n)

      # pre-allocate SOPs
      mat = zeros(m + prerun + 1, n + prerun + 1)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(m, n, alpha, 1)
      crit_sop_2 = crit_val_sop(m, n, alpha, 2)
      crit_sop_3 = crit_val_sop(m, n, alpha, 3)
      crit_sop_4 = crit_val_sop(m, n, alpha, 4)
      crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

      # create spatial dgp
      dgp = SINAR11(dgp_params, m, n, 100)

      get_checks = pmap(1:N) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp,
          dgp_params,
          dist_error,
          dist_ao,
          data,
          mat,
          mat_ao,
          mat_ma,
          cdata,
          cx_t_cx_t1,
          cdata_sq)
      end

      println("#--------------------------------------------------------#")
      println("m: ", m)
      println("n: ", n)
      println("lambda: ", lambda)
      println("params: ", dgp_params)
      round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
    end
  end
end


#-------------------------------------------#
#             Table B.8
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing

eps_params = [(2, 2, 2); (1, 1, 2); (2, 1, 2); (2, 1, 1)]

println("#-------------------------------------------#")
println("#   Table B.8 -> PoiBin(0.1, 5 * lambda)")
println("#-------------------------------------------#")
# Loop to compute values
for eps_param in eps_params
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = zeros(m + prerun + 1, n + prerun + 1)
    mat_ma = similar(mat)
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    # create spatial dgp
    dgp = SQMA11((0.8, 0.8, 0.8), eps_param, m, n, 100)
    dgp_params = (0.1, 0.1, 0.1)

    get_checks = pmap(1:N) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dgp_params,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        mat_ma,
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("eps params: ", eps_param)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end

#-------------------------------------------#
#             Table B.9
#-------------------------------------------#
lambdas_vec = [0.5; 1; 2; 5; 10]
dist_ao = nothing
eps_params = [(2, 2, 2); (1, 1, 2); (2, 1, 2); (2, 1, 1)]

println("#-------------------------------------------#")
println("#   Table B.9 -> PoiBin(0.1, 5 * lambda)")
println("#-------------------------------------------#")
# Loop to compute values
for eps_param in eps_params
  for lambda in lambdas_vec

    dist_error = Poisson(lambda)
    for mn in mn_vec

      # Get m and n
      m = mn[1]
      n = mn[2]

      # pre-allocate SACF
      data = zeros(m + 1, n + 1)
      cdata = similar(data)
      cdata_sq = similar(data)
      cx_t_cx_t1 = zeros(m, n)

      # pre-allocate SOPs
      mat = zeros(m + prerun + 1, n + prerun + 1)
      mat_ma = similar(mat)
      mat_ao = similar(mat)

      # Compute critical values
      crit_sop_1 = crit_val_sop(m, n, alpha, 1)
      crit_sop_2 = crit_val_sop(m, n, alpha, 2)
      crit_sop_3 = crit_val_sop(m, n, alpha, 3)
      crit_sop_4 = crit_val_sop(m, n, alpha, 4)
      crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

      # create spatial dgp
      dgp = SQINMA11((0.8, 0.8, 0.8), eps_param, m, n, 100)
      dgp_params = (0.1, 0.1, 0.1)

      get_checks = pmap(1:N) do i
        compute_reject_sop(
          crit_sop_1,
          crit_sop_2,
          crit_sop_3,
          crit_sop_4,
          crit_sacf,
          dgp,
          dgp_params,
          dist_error,
          dist_ao,
          data,
          mat,
          mat_ao,
          mat_ma,
          cdata,
          cx_t_cx_t1,
          cdata_sq)
      end

      println("#--------------------------------------------------------#")
      println("m: ", m)
      println("n: ", n)
      println("lambda: ", lambda)
      println("eps params: ", eps_param)
      round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
      round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
    end
  end
end

#-------------------------------------------#
#             Table B.10.1
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing

dgp_params = [(0.1, 0.1, 0.1, 0.1); (0.05, 0.05, 0.15, 0.15); (0.05, 0.15, 0.05, 0.15)]

println("#-------------------------------------------#")
println("#   Table B.10.1")
println("#-------------------------------------------#")
# Loop to compute values
for dgp_param in dgp_params
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # create spatial dgp
    dgp = SAR1(dgp_param, m, n, margin)

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = build_sar1_matrix(dgp) # will be done only once
    mat_ao = zeros((m + 1 + 2 * margin), (n + 1 + 2 * margin))
    vec_ar = zeros((m + 1 + 2 * margin) * (n + 1 + 2 * margin))
    vec_ar2 = similar(vec_ar)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    get_checks = pmap(1:N) do i
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
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("params: ", dgp_param)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end

#-------------------------------------------#
#             Table B.10.2
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialC(0.1, 5)

dgp_params = [(0.1, 0.1, 0.1, 0.1); (0.05, 0.05, 0.15, 0.15); (0.05, 0.15, 0.05, 0.15)]

println("#-------------------------------------------#")
println("#   Table B.10.2")
println("#-------------------------------------------#")
# Loop to compute values
for dgp_param in dgp_params
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # create spatial dgp
    dgp = SAR1(dgp_param, m, n, margin)

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = build_sar1_matrix(dgp) # will be done only once
    mat_ao = zeros((m + 1 + 2 * margin), (n + 1 + 2 * margin))
    vec_ar = zeros((m + 1 + 2 * margin) * (n + 1 + 2 * margin))
    vec_ar2 = similar(vec_ar)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    get_checks = pmap(1:N) do i
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
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("params: ", dgp_param)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end


#-------------------------------------------#
#             Table B.10.3
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialCvec(0.1, [-5; 5])

dgp_params = [(0.1, 0.1, 0.1, 0.1); (0.05, 0.05, 0.15, 0.15); (0.05, 0.15, 0.05, 0.15)]

println("#-------------------------------------------#")
println("#   Table B.10.3")
println("#-------------------------------------------#")
# Loop to compute values
for dgp_param in dgp_params
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # create spatial dgp
    dgp = SAR1(dgp_param, m, n, margin)

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = build_sar1_matrix(dgp) # will be done only once
    mat_ao = zeros((m + 1 + 2 * margin), (n + 1 + 2 * margin))
    vec_ar = zeros((m + 1 + 2 * margin) * (n + 1 + 2 * margin))
    vec_ar2 = similar(vec_ar)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    get_checks = pmap(1:N) do i
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
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("params: ", dgp_param)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end


#-------------------------------------------#
#             Table B.11
#-------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = nothing

eps_params = [(2, 2, 2, 2); (2, 1, 2, 1); (2, 2, 1, 1)]

println("#-------------------------------------------#")
println("#              Table B.11")
println("#-------------------------------------------#")
# Loop to compute values
for eps_param in eps_params
  for mn in mn_vec

    # Get m and n
    m = mn[1]
    n = mn[2]

    # pre-allocate SACF
    data = zeros(m + 1, n + 1)
    cdata = similar(data)
    cdata_sq = similar(data)
    cx_t_cx_t1 = zeros(m, n)

    # pre-allocate SOPs
    mat = zeros(m + prerun + 1, n + prerun + 1)
    mat_ma = zeros(m + prerun + 1 + 1, n + prerun + 1 + 1) # add one more row and column for "forward looking" BSQMA11
    mat_ao = similar(mat)

    # Compute critical values
    crit_sop_1 = crit_val_sop(m, n, alpha, 1)
    crit_sop_2 = crit_val_sop(m, n, alpha, 2)
    crit_sop_3 = crit_val_sop(m, n, alpha, 3)
    crit_sop_4 = crit_val_sop(m, n, alpha, 4)
    crit_sacf = crit_val_sacf((m + 1), (n + 1), alpha)

    # create spatial dgp
    dgp = BSQMA11((0.8, 0.8, 0.8, 0.8), eps_param, m, n, 100)
    dgp_params = (0.8, 0.8, 0.8, 0.8)

    get_checks = pmap(1:N) do i
      compute_reject_sop(
        crit_sop_1,
        crit_sop_2,
        crit_sop_3,
        crit_sop_4,
        crit_sacf,
        dgp,
        dgp_params,
        dist_error,
        dist_ao,
        data,
        mat,
        mat_ao,
        mat_ma,
        cdata,
        cx_t_cx_t1,
        cdata_sq)
    end

    println("#--------------------------------------------------------#")
    println("m: ", m)
    println("n: ", n)
    println("eps params: ", eps_param)
    round(sum(getindex.(get_checks, 1) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 2) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 3) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 4) / N); digits=4) |> x -> print(x, " ; ")
    round(sum(getindex.(get_checks, 5) / N); digits=4) |> x -> println("SACF: ", x)
  end
end