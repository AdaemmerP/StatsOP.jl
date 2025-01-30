
# ----------------------------------------------------------------------#
# --------    Computation of ARLs for OOC processes          -----------#
# ----------------------------------------------------------------------#

cl_sacf_mat = load_object("cl_sacf_delays.jld2")
cl_sop_mat = load_object("cl_sop_delays.jld2")

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

        # Compute ARL for SACF for SAR(2, 2)
        sacf_results = arl_sacf_oc(sar22, lam, cl_sacf_mat[i, j],d1, d2, reps)
        arl_sacf_sar22_mat[i, j] = sacf_results[1]
        sd_sacf_sar22_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SAR(2, 2)
        sop_results = arl_sop_oc(sar22, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
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

        # Compute ARL for SACF for SAR(2, 2) with outliers
        sacf_results = arl_sacf_oc(sar22_outl, lam, cl_sacf_mat[i, j], d1, d2, reps)
        arl_sacf_sar22_outl_mat[i, j] = sacf_results[1]
        sd_sacf_sar22_outl_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SAR(2, 2) with outliers
        sop_results = arl_sop_oc(sar22_outl, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
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
        sqma11 = SQMA11((0.8, 0.8, 0.8), (2, 2, 2), M, N, Normal(0, 1), nothing)

        # Compute ARL for SACF for SQMA(1, 1)
        sacf_results = arl_sacf_oc(sqma11, lam, cl_sacf_mat[i, j], d1, d2, reps)
        arl_sacf_sqma11_mat[i, j] = sacf_results[1]
        sd_sacf_sqma11_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SQMA(1, 1)
        sop_results = arl_sop_oc(sqma11, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
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
            M, N, Normal(0, 1), nothing
        )

        # Compute ARL for SACF for SQMA(2, 2)
        sacf_results = arl_sacf_oc(sqma22, lam, cl_sacf_mat[i, j], d1, d2, reps)
        arl_sacf_sqma22_mat[i, j] = sacf_results[1]
        sd_sacf_sqma22_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SQMA(2, 2)
        sop_results = arl_sop_oc(sqma22, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
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

        # Compute ARL for SACF for SAR(1)
        sacf_results = arl_sacf_oc(sar1, lam, cl_sacf_mat[i, j], d1, d2, reps)
        arl_sacf_sar1_mat[i, j] = sacf_results[1]
        sd_sacf_sar1_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SAR(1)
        sop_results = arl_sop_oc(sar1, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
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

        # Compute ARL for SACF for SAR(1) with outliers
        sacf_results = arl_sacf_oc(sar1_outl, lam, cl_sacf_mat[i, j], d1, d2, reps)
        arl_sacf_sar1_outl_mat[i, j] = sacf_results[1]
        sd_sacf_sar1_outl_mat[i, j] = sacf_results[2]

        # Compute ARL for SOP for SAR(1) with outliers
        sop_results = arl_sop_oc(sar1_outl, lam, cl_sop_mat[i, j], d1, d2, reps; chart_choice=3)
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

