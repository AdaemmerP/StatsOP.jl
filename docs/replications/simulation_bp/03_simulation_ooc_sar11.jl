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
addprocs(20)
# addprocs(SlurmManager())
@everywhere using OrdinalPatterns
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
BLAS.set_num_threads(1)


# Vector and delay combinations
reps = 10^5
lam = 0.1
MN_vec = [(11, 11), (16, 16), (26, 26), (41, 26)]
d1d2_vec = [(1, 1), (2, 2), (3, 3)]
w_max = 3

# Load critical values
cl_sacf_mat = load_object("climits/cl_sacf_delays.jld2")
cl_sop_mat = load_object("climits/cl_sop_delays.jld2")
cl_sacf_bp_mat = load_object("climits/cl_sacf_bp.jld2")
cl_sop_bp_mat = load_object("climits/cl_sop_bp.jld2")


# ---------------------------------------------------------------- #
# ----                      SACF and SOP                -----------#
# ---------------------------------------------------------------- #

# --- Results for SAR(1, 1) for SACF and SOP
arl_sacf_sar11_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sar11_mat = similar(arl_sacf_sar11_mat)
arl_sop_sar11_mat = similar(arl_sacf_sar11_mat)
sd_sop_sar11_mat = similar(arl_sacf_sar11_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sar11 = SAR11((0.4, 0.3, 0.1), M, N, Normal(0, 1), nothing, 100)

        # Compute ARL for SAR(1, 1)
        sacf_results = arl_sacf_oc(sar11, lam, cl_sacf_mat[i, j], d1, d2, reps)        
        arl_sacf_sar11_mat[i, j] = sacf_results[1]
        sd_sacf_sar11_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SAR(1, 1)
        sop_results = arl_sop_oc(sar11, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
        arl_sop_sar11_mat[i, j] = sop_results[1]
        sd_sop_sar11_mat[i, j] = sop_results[2]
        println("Progress -> SAR(1, 1): i: $i, j: $j")
    end
end
# Save matrices to JLD2 file
jldsave("arl_sacf_sar11.jld2"; arl_sacf_sar11_mat)
jldsave("sd_sacf_sar11.jld2"; sd_sacf_sar11_mat)
jldsave("arl_sop_sar11.jld2"; arl_sop_sar11_mat)
jldsave("sd_sop_sar11.jld2"; sd_sop_sar11_mat)

# --- Results for SAR(1, 1) with outliers
arl_sacf_sar11_outl_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sar11_outl_mat = similar(arl_sacf_sar11_outl_mat)
arl_sop_sar11_outl_mat = similar(arl_sacf_sar11_outl_mat)
sd_sop_sar11_outl_mat = similar(arl_sacf_sar11_outl_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sar11_outl = SAR11((0.4, 0.3, 0.1), M, N, Normal(0, 1), BinomialC(0.1, 10), 100)

        # Compute ARL for SAR(1, 1) with outliers
        sacf_results = arl_sacf_oc(sar11_outl, lam, cl_sacf_mat[i, j], d1, d2, reps)
        arl_sacf_sar11_outl_mat[i, j] = sacf_results[1]
        sd_sacf_sar11_outl_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SAR(1, 1) with outliers
        sop_results = arl_sop_oc(sar11_outl, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
        arl_sop_sar11_outl_mat[i, j] = sop_results[1]
        sd_sop_sar11_outl_mat[i, j] = sop_results[2]
        println("Progress -> SAR(1, 1) with outliers: i: $i, j: $j")
    end
end

# Save matrices to JLD2 file
jldsave("arl_sacf_sar11_outl.jld2"; arl_sacf_sar11_outl_mat)
jldsave("sd_sacf_sar11_outl.jld2"; sd_sacf_sar11_outl_mat)
jldsave("arl_sop_sar11_outl.jld2"; arl_sop_sar11_outl_mat)
jldsave("sd_sop_sar11_outl.jld2"; sd_sop_sar11_outl_mat)

# ---------------------------------------------------------------- #
# ----                  BP-SACF and BP-SOP              -----------#
# ---------------------------------------------------------------- #

# --- Results for SAR(1, 1) for BP-SACF and BP-SOP without outliers
arl_sacf_bp_sar11_mat = zeros(length(MN_vec), w_max)
sd_sacf_bp_sar11_mat = similar(arl_sacf_bp_sar11_mat)
arl_sop_bp_sar11_mat = similar(arl_sacf_bp_sar11_mat)
sd_sop_bp_sar11_mat = similar(arl_sacf_bp_sar11_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for w in 1:w_max
        sar11 = SAR11((0.4, 0.3, 0.1), M, N, Normal(0, 1), nothing, 100)

        # Compute ARL for BP-SACF for SAR(1, 1)
        sacf_results = arl_sacf_bp_oc(sar11, lam, cl_sacf_bp_mat[i, w], w, reps)
        arl_sacf_bp_sar11_mat[i, w] = sacf_results[1]
        sd_sacf_bp_sar11_mat[i, w] = sacf_results[2]

        # Compute ARL for BP-SOP for SAR(1, 1)
        sop_results = arl_sop_bp_oc(sar11, lam, cl_sop_bp_mat[i, w], w, reps; chart_choice=3)
        arl_sop_bp_sar11_mat[i, w] = sop_results[1]
        sd_sop_bp_sar11_mat[i, w] = sop_results[2]
        println("Progress -> SAR(1, 1): i: $i, w: $w")
    end
end

# Safe
jldsave("arl_sacf_bp_sar11.jld2"; arl_sacf_bp_sar11_mat)
jldsave("sd_sacf_bp_sar11.jld2"; sd_sacf_bp_sar11_mat)
jldsave("arl_sop_bp_sar11.jld2"; arl_sop_bp_sar11_mat)
jldsave("sd_sop_bp_sar11.jld2"; sd_sop_bp_sar11_mat)

# --- Results for SAR(1, 1) for BP-SACF and BP-SOP with outliers
arl_sacf_bp_sar11_outl_mat = zeros(length(MN_vec), w_max)
sd_sacf_bp_sar11_outl_mat = similar(arl_sacf_bp_sar11_outl_mat)
arl_sop_bp_sar11_outl_mat = similar(arl_sacf_bp_sar11_outl_mat)
sd_sop_bp_sar11_outl_mat = similar(arl_sacf_bp_sar11_outl_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for w in 1:w_max
        sar11_outl = SAR11((0.4, 0.3, 0.1), M, N, Normal(0, 1), BinomialC(0.1, 10), 100)

        # Compute ARL for BP-SACF for SAR(1, 1) with outliers
        sacf_results = arl_sacf_bp_oc(sar11_outl, lam, cl_sacf_bp_mat[i, w], w, reps)
        arl_sacf_bp_sar11_outl_mat[i, w] = sacf_results[1]
        sd_sacf_bp_sar11_outl_mat[i, w] = sacf_results[2]

        # Compute ARL for BP-SOP for SAR(1, 1) with outliers
        sop_results = arl_sop_bp_oc(sar11_outl, lam, cl_sop_bp_mat[i, w], w, reps; chart_choice=3)
        arl_sop_bp_sar11_outl_mat[i, w] = sop_results[1]
        sd_sop_bp_sar11_outl_mat[i, w] = sop_results[2]
        println("Progress -> SAR(1, 1) with outliers: i: $i, w: $w")
    end
end

# Safe
jldsave("arl_sacf_bp_sar11_outl.jld2"; arl_sacf_bp_sar11_outl_mat)
jldsave("sd_sacf_bp_sar11_outl.jld2"; sd_sacf_bp_sar11_outl_mat)
jldsave("arl_sop_bp_sar11_outl.jld2"; arl_sop_bp_sar11_outl_mat)
jldsave("sd_sop_bp_sar11_outl.jld2"; sd_sop_bp_sar11_outl_mat)
