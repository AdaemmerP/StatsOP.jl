# Change to current directory and activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("../../../../.")

using Distributed
# set the number of workers for parallel computing
addprocs(10)

@everywhere using OrdinalPatterns
@everywhere using LinearAlgebra
@everywhere using JLD2
@everywhere BLAS.set_num_threads(1)

# Vector and delay combinations
MN_vec = [(2, 2), (11, 11), (16, 16), (26, 26), (41, 26)]

# ----------------------------------------------------------------------#
# -------- Computation of critical limits for SACF and SOPs ------------#
# ----------------------------------------------------------------------#
reps = 10^3 # Increase the number of replications to 10^6 for reproduction of the results in the paper
L0 = 370
jmin = 3
jmax = 7
d1 = 1
d2 = 1
lam_vec = [0.05, 0.1, 0.25]
chart_vec = [1, 2, 3, 4]

# Build Matrix to store all critical limits, ARLs and ARLSEs
cl_sop_mat = zeros(length(MN_vec), 3, 4)
arl_sop_mat = similar(cl_sop_mat)
arlse_sop_mat = similar(cl_sop_mat)

cl_sacf_mat = zeros(length(MN_vec), 3, 2)
arl_sacf_mat = similar(cl_sacf_mat)
arlse_sacf_mat = similar(cl_sacf_mat)

# Compute SOP statistics
for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSTS(M, N, Normal(0, 1))
    for j in 1:length(lam_vec)
        lam = lam_vec[j]
        for k in 1:length(chart_vec)
            println("M = $M, N = $N, lambda = $lam", " chart = $k")

            # Limits for SOPs
            cl_init_sop = map(i -> stat_sop(randn(M, N, 370), lam, d1, d2; chart_choice=k) |> last, 1:1_000) |> x -> quantile(x, 0.99)
            cl = cl_sop(sp_dgp, lam, L0, cl_init_sop, d1, d2, reps; chart_choice=k, jmin=jmin, jmax=jmax, verbose=true)
            cl_sop_mat[i, j, k] = cl
            # ARL and ARLSE for SOPs
            arl = arl_sop_ic(sp_dgp, lam, cl, d1, d2, reps; chart_choice=k)
            arl_sop_mat[i, j, k] = arl[1]
            arlse_sop_mat[i, j, k] = arl[2]
        end
    end
end

dist = [Normal(0, 1), Poisson(0.5)]

# Compute SACF statistics
for (k, dist) in enumerate(dist)
    for (i, MN) in enumerate(MN_vec)
        M = MN[1]
        N = MN[2]
        sp_dgp = ICSTS(M, N, dist)
        for j in 1:length(lam_vec)
            lam = lam_vec[j]
            println("M = $M, N = $N, lambda = $lam, distribution = $k")
            # Limits for SACF
            # automatic start value computation based on quatiles not working 
            # for Poisson for lambda = 0.25 and M,N=(2, 2)
            if k == 2 && i == 1 && j == 3 
                cl_init_sacf = 0.5 
                else
                cl_init_sacf = map(i -> stat_sacf(randn(M, N, 370), lam, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
            end
            cl = cl_sacf(sp_dgp, lam, L0, cl_init_sacf, d1, d2, reps; jmin=jmin, jmax=jmax, verbose=true)
            cl_sacf_mat[i, j, k] = cl
            # ARL and ARLSE for SACF
            arl = arl_sacf_ic(sp_dgp, lam, cl, d1, d2, reps;)
            arl_sacf_mat[i, j, k] = arl[1]
            arlse_sacf_mat[i, j, k] = arl[2]
        end
    end
end

# compare results with Table 1

# Control limit results
display(round.(cl_sop_mat, digits=5))
display(round.(cl_sacf_mat, digits=5))

# ARL results
display(round.(arl_sop_mat, digits=1))
display(round.(arl_sacf_mat, digits=1))

# Maximum ARL standard error
findmax(round.(arlse_sop_mat, digits=2))
findmax(round.(arlse_sacf_mat, digits=2))