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

# Load computed in-control limits SACF for count data
ic_sacf_count = [0.05305, 0.03698, 0.02309, 0.01847]

arl_mat = zeros(4 * 4, 5)
arlse_mat = similar(arl_mat)

# Initial values SQINMA(1, 1)
eps_vals = [
    (2, 2, 2);
    (2, 1, 2);
    (1, 1, 2);
    (2, 1, 1)
]

i = 1
for eps in eps_vals
    for (j, mn_tup) in enumerate(MN_vec)
        M = mn_tup[1]
        N = mn_tup[2]

        # Setup DGP SQINMA(1, 1)
        sop_dgp = SQINMA11((0.8, 0.8, 0.8), eps, M, N, Poisson(5), nothing)
        
        # Compute ARL for SOP for SQINMA(1, 1)
        for chart in 1:4
            sop_results = arl_sop_oc(sop_dgp, lam, ic_sop[j, chart], d1, d2, reps; chart_choice=chart)
            arl_mat[i, chart] = sop_results[1]
            arlse_mat[i, chart] = sop_results[2]
        end

        # Compute ARL for SACF for SQINMA(1, 1)
        sacf_results = arl_sacf_oc(sop_dgp, lam, ic_sacf_count[j], d1, d2, reps)
        arl_mat[i, 5] = sacf_results[1]
        arlse_mat[i, 5] = sacf_results[2]
        println("Progress -> SQINMA(1, 1): i: $i, j: $j")
        global i = i + 1
    end
end
# compare results with Table A.7

# ARL results
display(round.(arl_mat, digits=2))

# Maximum ARL standard error
findmax(round.(arlse_mat, digits=2))

# Remove workers
rmprocs(workers())
