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


# Vector and delay combinations
MN_vec = [(11, 11), (16, 16), (26, 26), (41, 26)]
d1d2_vec = [(1, 1), (2, 2), (3, 3)]
w = 3

# ----------------------------------------------------------------------#
# -------- Computation of critical limits for SACF and SOPs ------------#
# ----------------------------------------------------------------------#
reps = 1_00
lam = 0.1
L0 = 370
jmin = 4
jmax = 7

#--- Compute critical limits for SACF
# Build Matrix to store all critical limits
cl_sacf_mat = zeros(length(MN_vec), length(d1d2_vec))
cl_sacf_bp_mat = zeros(length(MN_vec), w)

cl_sop_mat = similar(cl_sacf_mat)
cl_sop_mat_bp = similar(cl_sacf_bp_mat)

# Compute SACF and SOP statistics for d1-d2-combinations
for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSP(M, N, Normal(0, 1))
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        println("M = $M, N = $N, d1 = $d1, d2 = $d2")

        # Limits for SACF
        cl_init_sacf = map(i -> stat_sacf(randn(M, N, 370), 0.1, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
        cl = cl_sacf(sp_dgp, lam, L0, cl_init_sacf, d1, d2, reps; jmin=4, jmax=7, verbose=true)
        cl_sacf_mat[i, j] = cl

        # Limits for SOPs
        cl_init_sop = map(i -> stat_sop(randn(M, N, 370), 0.1, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
        cl = cl_sop(sp_dgp, lam, L0, cl_init_sop, d1, d2, reps; jmin=jmin, jmax=jmax, verbose=true)
        cl_sop_mat[i, j] = cl
    end
end

# Save 
jldsave("cl_sacf_delays.jld2"; cl_sacf_mat)
jldsave("cl_sop_delays.jld2"; cl_sop_mat)

# ----------------------------------------------------------------------#
# --- Computation of critical limits for BP-SACF and BP-SOPs -----------#
# ----------------------------------------------------------------------#
@time for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSP(M, N, Normal(0, 1))

    for w in 1:w
        println("M = $M, N = $N, w = $w")

        # Limits for SACF-BP
        cl_init_sacf = map(i -> stat_sacf_bp(randn(M, N, 370), lam, w) |> last, 1:1_000) |>
                       x -> quantile(x, 0.99)
        cl = cl_sacf_bp(sp_dgp, lam, L0, cl_init_sacf, w::Int, reps;
            jmin=jmin, jmax=jmax, verbose=true
        )
        cl_sacf_bp_mat[i, w] = cl

        # Limits for SOP-BP
        cl_init_sop = map(i -> stat_sop_bp(randn(M, N, 370), 0.1, w) |> last, 1:1_000) |> x -> quantile(x, 0.99)
        cl = cl_sop_bp(sp_dgp, lam, L0, cl_init_sop, w, reps; jmin=jmin, jmax=jmax, verbose=true)
        cl_sop_mat_bp[i, w] = cl
    end

end
# --- Save matrix to JLD2 file
jldsave("cl_sacf_bp.jld2"; cl_sacf_bp_mat)
jldsave("cl_sop_bp.jld2"; cl_sop_mat_bp)
