# Change to current directory and activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("../../../.")

using Distributed
# set the number of workers for parallel computing
addprocs(6)

@everywhere include("../../../src/OrdinalPatterns.jl")
@everywhere using .OrdinalPatterns
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

MN_vec = [(11, 11), (16, 16), (26, 26), (41, 26)];
d1 = 1
d2 = 1
lam = 0.1
reps = 10^3 # Increase the number of replications to 10^5 for the reproduction of the results in the paper

# Load computed in-control limits SOPs
ic_sop = [
    0.03049 0.05426 0.03174 0.05209;
    0.02036 0.03626 0.02122 0.03476;
    0.01223 0.0218 0.01276 0.02087;
    0.00967 0.01724 0.01009 0.01650
]

# Load computed in-control limits SACF for continuous data
ic_sacf_cont = [0.05313, 0.03701, 0.02310, 0.01847]

# --------------------------------------------- #
# SAR(1,1) with additive outliers, single value #
# --------------------------------------------- #

arl_mat = zeros(4 * 4, 5)
arlse_mat = similar(arl_mat)

dgp_vals = [
    (0.1, 0.1, 0.1);
    (0.2, 0.2, 0.2);
    (0.2, 0.2, 0.5);
    (0.4, 0.3, 0.1)
]

c = 10
dist_ao = BinomialC(0.1, c)
i = 1
for alphas in dgp_vals
    for (j, mn_tup) in enumerate(MN_vec)
        M = mn_tup[1]
        N = mn_tup[2]

        # Set up DGP SAR(1, 1) with additive outliers
        sop_dgp = SAR11(alphas, M, N, Normal(0, 1), dist_ao, 100)

        for chart in 1:4
            # Compute ARL for SOP for SAR(1, 1)
            sop_results = arl_sop_oc(sop_dgp, lam, ic_sop[j, chart], d1, d2, reps; chart_choice=chart)
            arl_mat[i, chart] = sop_results[1]
            arlse_mat[i, chart] = sop_results[2]
        end
        # Compute ARL for SACF for SAR(1, 1)
        sacf_results = arl_sacf_oc(sop_dgp, lam, ic_sacf_cont[j], d1, d2, reps)            
        arl_mat[i, 5] = sacf_results[1]
        arlse_mat[i, 5] = sacf_results[2]
        println("Progress -> SAR(1, 1): grid: $j, alphas: $alphas")
        global i = i + 1
    end
end

# compare results with Table A.3 (top half)

# ARL results
display(round.(arl_mat, digits=2))

# Maximum ARL standard error
findmax(round.(arlse_mat, digits=2))

# --------------------------------------- #
# SAR(1,1) with additive outliers, vector #
# --------------------------------------- #

arl_mat = zeros(4 * 4, 5)
arlse_mat = similar(arl_mat)

dgp_vals = [
    (0.1, 0.1, 0.1);
    (0.2, 0.2, 0.2);
    (0.2, 0.2, 0.5);
    (0.4, 0.3, 0.1)
]

c = [-10, 10]
dist_ao = BinomialCvec(0.1, c)
i = 1
for alphas in dgp_vals
    for (j, mn_tup) in enumerate(MN_vec)
        M = mn_tup[1]
        N = mn_tup[2]

        # Set up DGP SAR(1, 1) with additive outliers
        sop_dgp = SAR11(alphas, M, N, Normal(0, 1), dist_ao, 100)

        for chart in 1:4
            # Compute ARL for SOP for SAR(1, 1)
            sop_results = arl_sop_oc(sop_dgp, lam, ic_sop[j, chart], d1, d2, reps; chart_choice=chart)
            arl_mat[i, chart] = sop_results[1]
            arlse_mat[i, chart] = sop_results[2]
        end
        # Compute ARL for SACF for SAR(1, 1)
        sacf_results = arl_sacf_oc(sop_dgp, lam, ic_sacf_cont[j], d1, d2, reps)            
        arl_mat[i, 5] = sacf_results[1]
        arlse_mat[i, 5] = sacf_results[2]
        println("Progress -> SAR(1, 1): grid: $j, alphas: $alphas")
        i = i + 1
    end
end

# compare results with Table A.3 (bottom half)

# ARL results
display(round.(arl_mat, digits=2))

# Maximum ARL standard error
findmax(round.(arlse_mat, digits=2))

# Remove workers
rmprocs(workers())