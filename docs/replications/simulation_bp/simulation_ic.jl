# Change to current directory
cd(@__DIR__)
using Pkg
Pkg.activate("../../.")
using OrdinalPatterns
using Distributed
using JLD2
using LinearAlgebra
using SlurmClusterManager

# Add number of cores
addprocs(5)
# addprocs(SlurmManager())
@everywhere using OrdinalPatterns
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
BLAS.set_num_threads(1)


#--- Compute critical limits for SACF
# Build Matrix to store all critical limits
cl_sacf_mat = zeros(length(MN_vec), length(d1d2_vec))
cl_sacf_bp_mat = zeros(length(MN_vec), 1)

cl_sop_mat = similar(cl_sacf_mat)
cl_sop_mat_bp = similar(cl_sacf_bp_mat)

# Vector and delay combinations
reps = 10^6
lam = 0.1
MN_vec = [(11, 11), (16, 16), (26, 26), (41, 26)]
d1d2_vec = [(1, 1), (2, 2), (3, 3)]
w = 3

# Load critical values
cl_sacf_mat = load_object("cl_sacf_delays.jld2")
cl_sop_mat = load_object("cl_sop_delays.jld2")
cl_sacf_bp_mat = load_object("cl_sacf_bp.jld2")
cl_sop_bp_mat = load_object("cl_sop_bp.jld2")


# Pre-allocate matrices
dist = [TDist(2), Exponential(1)]
# SACF
arl_ic_sacf_mat = zeros(length(MN_vec), length(d1d2_vec), length(dist))
sd_ic_sacf_mat = similar(arl_ic_sacf_mat)
arl_ic_sacf_bp_mat = zeros(length(MN_vec), w, length(dist))
sd_ic_sacf_bp_mat = similar(arl_ic_sacf_bp_mat)

# SOP
arl_ic_sop_mat = similar(arl_ic_sacf_mat)
sd_ic_sop_mat = similar(sd_ic_sacf_mat)
arl_ic_sop_bp_mat = similar(arl_ic_sacf_bp_mat)
sd_ic_sop_bp_mat = similar(sd_ic_sacf_bp_mat)

# Loop for individual d1-d2 pair
for (k, dist) in enumerate(dist)
  for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSP(M, N, dist)

    for (j, d1d2) in enumerate(d1d2_vec)
      d1 = d1d2[1]
      d2 = d1d2[2]

      # Compute SACF for one d1-d2 pair
      sacf_arl = arl_sacf_ic(sp_dgp, lam, cl_sacf_mat[i, j], d1, d2, reps)
      arl_ic_sacf_mat[i, j, k] = sacf_arl[1]
      sd_ic_sacf_mat[i, j, k] = sacf_arl[2]

      # Compute SACF-BP for one d1-d2 pair
      sop_arl_sd = arl_sop_ic(sp_dgp, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
      arl_ic_sop_mat[i, j, k] = sop_arl_sd[1]
      sd_ic_sop_mat[i, j, k] = sop_arl_sd[2]
      println("Progress -> k: $k, i: $i, j: $j")
    end
  end
end

# --- Save matrices to JLD2 file
jldsave("arl_sac_delays.jld2"; arl_sac_mat)
jldsave("sd_sac_delays.jld2"; sd_sac_mat)
jldsave("arl_sop_delays.jld2"; arl_ic_sop_mat)
jldsave("sd_sop_delays.jld2"; sd_ic_sop_mat)

# Loop for BP statistics
for (k, dist) in enumerate(dist)
  for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSP(M, N, dist)

    for w in 1:w
      # Compute SACF-BP for one d1-d2 pair
      sacf_bp_arl = arl_sacf_bp_ic(sp_dgp, lam, cl_sacf_bp_mat[i], w, reps)
      arl_ic_sacf_bp_mat[i, k] = sacf_bp_arl[1]
      sd_ic_sacf_bp_mat[i, k] = sacf_bp_arl[2]

      # Compute SOP-BP for one d1-d2 pair
      sop_bp_arl = arl_sop_bp_ic(sp_dgp, lam, cl_sop_bp_mat[i], w, reps; chart_choice=3)
      arl_ic_sop_bp_mat[i, k] = sop_bp_arl[1]
      sd_ic_sop_bp_mat[i, k] = sop_bp_arl[2]
      println("Progress -> k: $k, i: $i, w: $w")     
    end
  end
end