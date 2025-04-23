# Change to current directory and activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("../../../.")

using Distributed
# set the number of workers for parallel computing
addprocs(10)

@everywhere include("../../../src/OrdinalPatterns.jl")
@everywhere using .OrdinalPatterns
@everywhere using LinearAlgebra
@everywhere using JLD2
@everywhere BLAS.set_num_threads(1)

# Vector and delay combinations
reps = 10^3 # Increase the number of replications to 10^5 for reproduction of the results in the paper
lam = 0.1
MN_vec = [(11, 11), (16, 16), (26, 26), (41, 26)]
d1d2_vec = [(1, 1), (2, 2), (3, 3)]
w_max = 3

# Load critical values
cl_sacf_mat = load_object("../climits/cl_sacf_delays.jld2")
cl_sop_mat = load_object("../climits/cl_sop_delays.jld2")
cl_sacf_bp_mat = load_object("../climits/cl_sacf_bp.jld2")
cl_sop_bp_mat = load_object("../climits/cl_sop_bp.jld2")

# Pre-allocate matrices
dist = [Normal(0, 1), TDist(2), Exponential(1)] 

# SACF 
arl_ic_sacf_mat = zeros(length(MN_vec), length(d1d2_vec), length(dist))
arlse_ic_sacf_mat = similar(arl_ic_sacf_mat)
arl_ic_sacf_bp_mat = zeros(length(MN_vec), w_max, length(dist))
arlse_ic_sacf_bp_mat = similar(arl_ic_sacf_bp_mat)

# SOP
arl_ic_sop_mat = similar(arl_ic_sacf_mat)
arlse_ic_sop_mat = similar(arlse_ic_sacf_mat)
arl_ic_sop_bp_mat = similar(arl_ic_sacf_bp_mat)
arlse_ic_sop_bp_mat = similar(arlse_ic_sacf_bp_mat)

# Loop for Table A.11 (left panel)
for (k, dist) in enumerate(dist)
  for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSTS(M, N, dist)

    for (j, d1d2) in enumerate(d1d2_vec)
      d1 = d1d2[1]
      d2 = d1d2[2]

      # Compute SACF for one d1-d2 pair
      sacf_arl = arl_sacf_ic(sp_dgp, lam, cl_sacf_mat[i, j], d1, d2, reps)
      arl_ic_sacf_mat[i, j, k] = sacf_arl[1]
      arlse_ic_sacf_mat[i, j, k] = sacf_arl[2]

      # Compute SACF-BP for one d1-d2 pair
      sop_arl_sd = arl_sop_ic(sp_dgp, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
      arl_ic_sop_mat[i, j, k] = sop_arl_sd[1]
      arlse_ic_sop_mat[i, j, k] = sop_arl_sd[2]
      println("Progress -> k: $k, i: $i, j: $j")
    end
  end
end

# Loop for Table A.11 (right panel)
for (k, dist) in enumerate(dist)
  for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSTS(M, N, dist)

    for w in 1:w_max
      # Compute SACF-BP for one d1-d2 pair
      sacf_bp_arl = arl_sacf_bp_ic(sp_dgp, lam, cl_sacf_bp_mat[i, w], w, reps)
      arl_ic_sacf_bp_mat[i, w, k] = sacf_bp_arl[1]
      arlse_ic_sacf_bp_mat[i, w, k] = sacf_bp_arl[2]

      # Compute SOP-BP for one d1-d2 pair
      sop_bp_arl = arl_sop_bp_ic(sp_dgp, lam, cl_sop_bp_mat[i, w], w, reps; chart_choice=3)
      arl_ic_sop_bp_mat[i, w, k] = sop_bp_arl[1]
      arlse_ic_sop_bp_mat[i, w, k] = sop_bp_arl[2]
      println("Progress -> k: $k, i: $i, w: $w")     
    end
  end
end


# compare results with Table A.11

# ARL results
hcat(reshape(arl_ic_sacf_mat, 4, 9)[:,4:8], reshape(arl_ic_sacf_bp_mat, 4, 9)[:,4:8])

# Maximum ARL standard error
findmax(
  hcat(reshape(arlse_ic_sacf_mat, 4, 9)[:,4:8],
  reshape(arlse_ic_sacf_bp_mat, 4, 9)[:,4:8]
))

# Remove workers
rmprocs(workers())