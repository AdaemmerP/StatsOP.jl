# Change to current directory and activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("../../../.")

using Distributed
# set the number of workers for parallel computing
addprocs(10)

@everywhere using OrdinalPatterns
@everywhere using JLD2
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

# Vector and delay combinations
lam = 0.1
MN_vec = [(11, 11), (16, 16), (26, 26), (41, 26)]
d1 = 1
d2 = 1
reps = 10^3 # Increase the number of replications to 10^5 for reproduction of the results in the paper

# Load critical values
cl_sacf_mat = load_object("../higher_order/climits/cl_sacf_delays.jld2")

# Pre-allocate matrices
dist = [TDist(2), PoiBin(0.2, 5), Weibull(1, 1.5), Exponential(1), Poisson(0.5),
    Laplace(0, 1), SkewNormal(0, 1, 10), Uniform(0, 1), BinNorm(-9, 9, 1, 1), Bernoulli(0.5)]

# SACF 
arl_ic_sacf_mat = zeros(length(MN_vec), length(dist))
arlse_ic_sacf_mat = similar(arl_ic_sacf_mat)

# Loop for individual d1-d2 pair
for (k, dist) in enumerate(dist)
    for (i, MN) in enumerate(MN_vec)
        M = MN[1]
        N = MN[2]
        sp_dgp = ICSTS(M, N, dist)

        # Compute SACF
        sacf_arl = arl_sacf_ic(sp_dgp, lam, cl_sacf_mat[i, 1], d1, d2, reps)
        arl_ic_sacf_mat[i, k] = sacf_arl[1]
        arlse_ic_sacf_mat[i, k] = sacf_arl[2]

        println("Progress -> distribution k: $k, grid i: $i")
    end
end

# compare results with Table 2

# ARL results
display(round.(arl_ic_sacf_mat, digits=2))

# Maximum ARL standard error
findmax(round.(arlse_ic_sacf_mat, digits=2))

# Remove workers
rmprocs(workers())