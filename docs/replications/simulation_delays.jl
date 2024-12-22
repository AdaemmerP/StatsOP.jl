using OrdinalPatterns
using Distributions
using Distributed
using JLD2

# Add number of cores
addprocs(10)
@everywhere using OrdinalPatterns
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

# Change to current directory
cd(@__DIR__)

# Vector and delay combinations
MN_vec = [(11, 11), (16, 16), (26, 26), (41, 26)]
d1d2_vec = [(1, 1), (2, 2), (3, 3), ([1, 2, 3], [1, 2, 3])]

# ----------------------------------------------------------------------#
# -------- Computation of critical limits for SACF and SOPs ------------#
# ----------------------------------------------------------------------#
reps = 10_000
lam = 0.1
L0 = 370
jmin = 4
jmax = 7

#--- Compute critical limits for SACF
# Build Matrix to store all critical limits
cl_sacf_mat = zeros(length(MN_vec), length(d1d2_vec))

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSP(M, N, Normal(0, 1))

    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        if d1 isa Int
            crit_init = map(i -> stat_sacf(0.1, randn(M, N, 370), d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
        else
            # Compute 370 pictures, compute the ARL and save the value at the 370th picture. From these values, compute the 1% quantile
            crit_init = map(i -> stat_sacf(0.1, randn(M, N, 370), d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.01)
        end
        println("M = $M, N = $N, d1 = $d1, d2 = $d2")
        #println("Initial limit: $crit_init")
        cl = cl_sacf(lam, L0, sp_dgp, crit_init, d1, d2, reps; jmin=jmin, jmax=jmax, verbose=true)
        cl_sacf_mat[i, j] = cl

    end
end

# --- Save matrix to JLD2 file
jldsave("cl_sacf_delays.jld2"; cl_sacf_mat)
# load_object("cl_sacf.jld2")

#--- Compute critical limits for SOPs
# Build Matrix to store all critical limits
cl_sop_mat = zeros(length(MN_vec), length(d1d2_vec))

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    sp_dgp = ICSP(M, N, Normal(0, 1))
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        if d1 isa Int
            crit_init = map(i -> stat_sop(0.1, randn(M, N, 370), d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.999)
        else
            # Compute 370 pictures, compute the ARL and save the value at the 370th picture. From these values, compute the 1% quantile
            crit_init = map(i -> stat_sop(0.1, randn(M, N, 370), d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.01)
        end
        println("M = $M, N = $N, d1 = $d1, d2 = $d2")
        #println("Initial limit: $crit_init")
        cl = cl_sop(lam, L0, sp_dgp, crit_init, d1, d2, reps; jmin=jmin, jmax=jmax, verbose=true)
        cl_sop_mat[i, j] = cl
    end
end

# --- Save matrix to JLD2 file
jldsave("cl_sop_delays.jld2"; cl_sop_mat)


# ----------------------------------------------------------------------#
# --------    Computation of ARLs for IC processes          ------------#
# ----------------------------------------------------------------------#
dist = [TDist(2), Exponential(1)]
arl_sac_mat = zeros(length(MN_vec), length(d1d2_vec), length(dist))
arl_sop_mat = similar(arl_sac_mat)
sd_sac_mat = similar(arl_sac_mat)
sd_sop_mat = similar(arl_sac_mat)

for (k, dist) in enumerate(dist)
    for (i, MN) in enumerate(MN_vec)
        M = MN[1]
        N = MN[2]
        sp_dgp = ICSP(M, N, dist)

        for (j, d1d2) in enumerate(d1d2_vec)
            d1 = d1d2[1]
            d2 = d1d2[2]
            # Compute and save for SACF
            sacf_arl_sd = arl_sacf(lam, cl_sacf_mat[i, j], sp_dgp, d1, d2, reps)
            arl_sac_mat[i, j, k] = sacf_arl_sd[1]
            sd_sac_mat[i, j, k] = sacf_arl_sd[2]
            # Compute and save for SOP
            sop_arl_sd = arl_sop(lam, cl_sop_mat[i, j], sp_dgp, d1, d2, reps; chart_choice=3)
            arl_sop_mat[i, j, k] = sop_arl_sd[1]
            sd_sop_mat[i, j, k] = sop_arl_sd[2]
            println("Progress -> k: $k, i: $i, j: $j")
        end
    end
end

# --- Save matrices to JLD2 file
jldsave("arl_sac_delays.jld2"; arl_sac_mat)
jldsave("sd_sac_delays.jld2"; sd_sac_mat)
jldsave("arl_sop_delays.jld2"; arl_sop_mat)
jldsave("sd_sop_delays.jld2"; sd_sop_mat)

# ----------------------------------------------------------------------#
# --------    Computation of ARLs for OOC processes          -----------#
# ----------------------------------------------------------------------#

cl_sacf_mat = load_object("cl_sacf.jld2")
cl_sop_mat  = load_object("cl_sop_delays.jld2")

# --- Results for SAR(1, 1)
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
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sar11, d1, d2, reps)
        arl_sacf_sar11_mat[i, j] = sacf_results[1]
        sd_sacf_sar11_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sar11, d1, d2, reps; chart_choice=3)
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
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sar11_outl, d1, d2, reps)
        arl_sacf_sar11_outl_mat[i, j] = sacf_results[1]
        sd_sacf_sar11_outl_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sar11_outl, d1, d2, reps; chart_choice=3)
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

# --- Results for SAR(2, 2)
arl_sacf_sar22_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sar22_mat = similar(arl_sacf_sar22_mat)
arl_sop_sar22_mat = similar(arl_sacf_sar22_mat)
sd_sop_sar22_mat = similar(arl_sacf_sar22_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sar22 = SAR22((0.0, 0.0, 0.0, 0.4, 0.3, 0.0, 0.0, 0.1), M, N, Normal(0, 1), nothing, 100)
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sar22, d1, d2, reps)
        arl_sacf_sar22_mat[i, j] = sacf_results[1]
        sd_sacf_sar22_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sar22, d1, d2, reps; chart_choice=3)
        arl_sop_sar22_mat[i, j] = sop_results[1]
        sd_sop_sar22_mat[i, j] = sop_results[2]
        println("Progress -> SAR(2, 2): i: $i, j: $j")
    end
end

# Save matrices to JLD2 file
jldsave("arl_sacf_sar22.jld2"; arl_sacf_sar22_mat)
jldsave("sd_sacf_sar22.jld2"; sd_sacf_sar22_mat)
jldsave("arl_sop_sar22.jld2"; arl_sop_sar22_mat)
jldsave("sd_sop_sar22.jld2"; sd_sop_sar22_mat)

# --- Results for SAR(2, 2) with outliers
arl_sacf_sar22_outl_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sar22_outl_mat = similar(arl_sacf_sar22_outl_mat)
arl_sop_sar22_outl_mat = similar(arl_sacf_sar22_outl_mat)
sd_sop_sar22_outl_mat = similar(arl_sacf_sar22_outl_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sar22_outl = SAR22((0.0, 0.0, 0.0, 0.4, 0.3, 0.0, 0.0, 0.1), M, N, Normal(0, 1), BinomialC(0.1, 10), 100)
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sar22_outl, d1, d2, reps)
        arl_sacf_sar22_outl_mat[i, j] = sacf_results[1]
        sd_sacf_sar22_outl_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sar22_outl, d1, d2, reps; chart_choice=3)
        arl_sop_sar22_outl_mat[i, j] = sop_results[1]
        sd_sop_sar22_outl_mat[i, j] = sop_results[2]
        println("Progress -> SAR(2, 2) with outliers: i: $i, j: $j")
    end
end

# Save matrices to JLD2 file
jldsave("arl_sacf_sar22_outl.jld2"; arl_sacf_sar22_outl_mat)
jldsave("sd_sacf_sar22_outl.jld2"; sd_sacf_sar22_outl_mat)
jldsave("arl_sop_sar22_outl.jld2"; arl_sop_sar22_outl_mat)
jldsave("sd_sop_sar22_outl.jld2"; sd_sop_sar22_outl_mat)

# --- Results for SQMA(1, 1)
arl_sacf_sqma11_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sqma11_mat = similar(arl_sacf_sqma11_mat)
arl_sop_sqma11_mat = similar(arl_sacf_sqma11_mat)
sd_sop_sqma11_mat = similar(arl_sacf_sqma11_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sqma11 = SQMA11((0.8, 0.8, 0.8), (2, 2, 2), M, N, Normal(0, 1), nothing, 1)
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sqma11, d1, d2, reps)
        arl_sacf_sqma11_mat[i, j] = sacf_results[1]
        sd_sacf_sqma11_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sqma11, d1, d2, reps; chart_choice=3)
        arl_sop_sqma11_mat[i, j] = sop_results[1]
        sd_sop_sqma11_mat[i, j] = sop_results[2]
        println("Progress -> SQMA(1, 1): i: $i, j: $j")
    end
end

# Save matrices to JLD2 file
jldsave("arl_sacf_sqma11.jld2"; arl_sacf_sqma11_mat)
jldsave("sd_sacf_sqma11.jld2"; sd_sacf_sqma11_mat)
jldsave("arl_sop_sqma11.jld2"; arl_sop_sqma11_mat)
jldsave("sd_sop_sqma11.jld2"; sd_sop_sqma11_mat)

# --- Results for SQMA(2, 2)
arl_sacf_sqma22_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sqma22_mat = similar(arl_sacf_sqma22_mat)
arl_sop_sqma22_mat = similar(arl_sacf_sqma22_mat)
sd_sop_sqma22_mat = similar(arl_sacf_sqma22_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sqma22 = SQMA22(
            (0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.8), 
            (0.0, 0.0, 0.0, 2, 2, 0.0, 0.0, 2), 
             M, N, Normal(0, 1), nothing, 1
        )
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sqma22, d1, d2, reps)
        arl_sacf_sqma22_mat[i, j] = sacf_results[1]
        sd_sacf_sqma22_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sqma22, d1, d2, reps; chart_choice=3)
        arl_sop_sqma22_mat[i, j] = sop_results[1]
        sd_sop_sqma22_mat[i, j] = sop_results[2]
        println("Progress -> SQMA(2, 2): i: $i, j: $j")
    end
end

# Save matrices to JLD2 file
jldsave("arl_sacf_sqma22.jld2"; arl_sacf_sqma22_mat)
jldsave("sd_sacf_sqma22.jld2"; sd_sacf_sqma22_mat)
jldsave("arl_sop_sqma22.jld2"; arl_sop_sqma22_mat)
jldsave("sd_sop_sqma22.jld2"; sd_sop_sqma22_mat)


# --- SAR(1)
#sar1 = SAR1((0.1, 0.1, 0.1, 0.1), 10, 10, Normal(0, 1), nothing, 20)
#sar1_outl = SAR1((0.1, 0.1, 0.1, 0.1), 10, 10, Normal(0, 1), BinomialCvec(0.1, [-5; 5]), 20)

# Results for SAR(1)
arl_sacf_sar1_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sar1_mat = similar(arl_sacf_sar1_mat)
arl_sop_sar1_mat = similar(arl_sacf_sar1_mat)
sd_sop_sar1_mat = similar(arl_sacf_sar1_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sar1 = SAR1((0.1, 0.1, 0.1, 0.1), M, N, Normal(0, 1), nothing, 20)
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sar1, d1, d2, reps)
        arl_sacf_sar1_mat[i, j] = sacf_results[1]
        sd_sacf_sar1_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sar1, d1, d2, reps; chart_choice=3)
        arl_sop_sar1_mat[i, j] = sop_results[1]
        sd_sop_sar1_mat[i, j] = sop_results[2]
        println("Progress -> SAR(1): i: $i, j: $j")
    end
end

# Save matrices to JLD2 file
jldsave("arl_sacf_sar1.jld2"; arl_sacf_sar1_mat)
jldsave("sd_sacf_sar1.jld2"; sd_sacf_sar1_mat)
jldsave("arl_sop_sar1.jld2"; arl_sop_sar1_mat)
jldsave("sd_sop_sar1.jld2"; sd_sop_sar1_mat)

# --- SAR(1) with outliers
arl_sacf_sar1_outl_mat = zeros(length(MN_vec), length(d1d2_vec))
sd_sacf_sar1_outl_mat = similar(arl_sacf_sar1_outl_mat)
arl_sop_sar1_outl_mat = similar(arl_sacf_sar1_outl_mat)
sd_sop_sar1_outl_mat = similar(arl_sacf_sar1_outl_mat)

for (i, MN) in enumerate(MN_vec)
    M = MN[1]
    N = MN[2]
    for (j, d1d2) in enumerate(d1d2_vec)
        d1 = d1d2[1]
        d2 = d1d2[2]
        sar1_outl = SAR1((0.1, 0.1, 0.1, 0.1), M, N, Normal(0, 1), BinomialC(0.1, 10), 20)
        sacf_results = arl_sacf(lam, cl_sacf_mat[i, j], sar1_outl, d1, d2, reps)
        arl_sacf_sar1_outl_mat[i, j] = sacf_results[1]
        sd_sacf_sar1_outl_mat[i, j] = sacf_results[2]
        sop_results = arl_sop(lam, cl_sop_mat[i, j], sar1_outl, d1, d2, reps; chart_choice=3)
        arl_sop_sar1_outl_mat[i, j] = sop_results[1]
        sd_sop_sar1_outl_mat[i, j] = sop_results[2]
        println("Progress -> SAR(1) with outliers: i: $i, j: $j")
    end
end

# Save matrices to JLD2 file
jldsave("arl_sacf_sar1_outl.jld2"; arl_sacf_sar1_outl_mat)
jldsave("sd_sacf_sar1_outl.jld2"; sd_sacf_sar1_outl_mat)
jldsave("arl_sop_sar1_outl.jld2"; arl_sop_sar1_outl_mat)
jldsave("sd_sop_sar1_outl.jld2"; sd_sop_sar1_outl_mat)




# Testings
# @btime arl_sop(lam, cl_sop_mat[1, 1], sar1, 1, 1, 100; chart_choice=3)
sar1 = SAR1((0.1, 0.1, 0.1, 0.1), 11, 11, Normal(0, 1), nothing, 20)
@btime arl_sop(0.1, cl_sop_mat[1, 1], sar1, 1, 1, 100; chart_choice=3)

# mat = 

function test(dgp::SAR1, dist_error, dist_ao::Nothing, mat, mat_ao::Matrix{Float64}, vec_ar::Vector{Float64}, vec_ar2::Vector{Float64})

    # draw MA-errors  
    margin = dgp.margin
    M_rows = dgp.M_rows
    N_cols = dgp.N_cols
    M = M_rows + 2 * margin
    N = N_cols + 2 * margin
    #m = dgp.m_rows
    #n = dgp.n_cols
    #M = m + 1 + 2 * margin
    #N = n + 1 + 2 * margin
  
    rand!(dist_error, vec_ar)
    mul!(vec_ar2, mat, vec_ar)
    mat2 = reshape(vec_ar2, M, N)
  
    @views mat2[(margin+1):(margin+M_rows), (margin+1):(margin+N_cols)]
  
  end

  M = N = 10
  mat = build_sar1_matrix(sar1)
  mat_ao = zeros((M + 2 * 20), (N + 2 * 20))
  vec_ar = zeros((M + 2 *20) * (N + 2 * 20))
  vec_ar2 = similar(vec_ar)


  @btime test($sar1, Normal(0, 1), nothing, $mat, $mat_ao, $vec_ar, $vec_ar2)
  @code_warntype test(sar1, Normal(0, 1), nothing, mat, mat_ao, vec_ar, vec_ar2)

#   @code_warntype build_sar1_matrix(sar1)


#   rand!( Normal(0, 1), vec_ar)
#   mul!(vec_ar2, mat, vec_ar)