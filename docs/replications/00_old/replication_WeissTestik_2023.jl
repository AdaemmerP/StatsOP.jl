using Pkg
Pkg.activate(".")
using Random
using LinearAlgebra
using Statistics
using Combinatorics
using Distributions
using Distributed
using BenchmarkTools
using Makie
using CairoMakie
using ComplexityMeasures
using StaticArrays
using TimerOutputs

include("op_dgp_structs.jl")
include("op_dgp_functions.jl")
include("op_help_functions.jl")
include("op_stat_functions.jl")
include("op_arl_functions.jl")
include("op_acf_functions.jl")

# --------------------------------------------------- #
# ---   Replication for Weiss and Testik (2023)   --- #
# --------------------------------------------------- #

#@time begin

# --------------------------------------------------- #
# --- Compute control limits for Table 1, p. 344  --- #
# --------------------------------------------------- #
op_dgp = ICST(Normal(0, 1))
L0 = 370.0
cl_init = 0.1
reps = 10_000
jmin = 1
jmax = 6
verbose = true
lam = [0.25, 0.1, 0.05]
chart = 5
#cl = cl_op(lam[1], L0, reps, op_dgp, chart, cl_init, jmin, jmax, verbose; ced=false, ad=100)
# 
matcl_op = zeros(3, 6)
for i in 1:3 # lambda loop
    for j in 1:6 # chart loop
        println("lambda i: $i, chart j: $j")
        data = randn(10_000)
        if j == 1 || j == 2
            cl_init = quantile(stat_op(data, lam[i], chart_choice=j)[1], 0.01)             
        else
            cl_init = quantile(stat_op(data, lam[i], chart_choice=j)[1], 0.99)
        end
        matcl_op[i, j] = cl_op(lam[i], L0, op_dgp, cl_init, reps; chart_choice=j, jmin=4, jmax=6, verbose=verbose, d=1, ced=false, ad=100)
                     
    end
    println(i)
end

matcl_op

# --------------------------------------------------- #
# --- Table 1 page 344, Weiss and Testik (2023)   --- #
# --------------------------------------------------- #
matcl_op = Array{Float64}(undef, 3, 6)
matcl_op[1, :] = [1.014, 0.6621, 0.3338, 0.6437, 0.4253, 0.7656]
matcl_op[2, :] = [1.4601, 0.8405, 0.1115, 0.3638, 0.2529, 0.4876]
matcl_op[3, :] = [1.6356, 0.88017, 0.05125, 0.233, 0.16775, 0.3246]
lam = [0.25, 0.1, 0.05]
# Make struct for in-control
op_dgp = IC(Normal(0, 1))
matarl_op = zeros(3, 6)
matarlse_op = zeros(3, 6)
 
for i in 1:3 # lambda loop
    for j in 1:6 # chart loop
        println("lambda i: $i, chart j: $j")
        res_op = arl_op(lam[i], matcl_op[i, j], op_dgp, reps; chart_choice=j)
        matarl_op[i, j] = res_op[1]
        matarlse_op[i, j] = res_op[2]
    end
end

matarl_op
matarlse_op

# --------------------------------------------------- #
# ---  Table 2 page 345 Weiss and Testik (2023)   --- #
# --------------------------------------------------- #
#matarl_op_tmp = Vector{Float64}
matarl_op = zeros(1, 5)

for i in 1:3 # lambda
    for alpha in [0.20, 0.40, 0.60, 0.80, -0.20, -0.40, -0.60, -0.80]
        matarl_op_tmp = ["λ = $(lam[i]); α = $alpha"]
        for j in [1, 2, 3, 5] # chart loop

            # Make struct for out-of-control and compute ARL 
            op_dgp = AR1(alpha, Normal(0, 1))
            res_op = arl_op(lam[i], matcl_op[i, j], op_dgp, reps; chart_choice=j)

            # Save temporary results
            matarl_op_tmp = hcat(matarl_op_tmp, res_op[1])
        end
        # Merge temporary results
        matarl_op = vcat(matarl_op, matarl_op_tmp)
    end
    # Add separator
    matarl_op = vcat(matarl_op, repeat(["---"], 1, 5))
    println(i)
end
matarl_op[2:end-1, :]

# --------------------------------------------------- #
# --- Table 3 page 345 Weiss and Testik (2023)    --- #
# --------------------------------------------------- #
matarl_op = zeros(1, 7)

for i in 1:3 # lambda loop
    for alpha in 0.1:0.1:0.6
        matarl_op_tmp = ["λ = $(lam[i]); α = $alpha"]
        for j in 1:6 # chart loop      

            # Make struct for out-of-control and compute ARL
            op_dgp = TEAR1(alpha, Exponential(1))
            res_op = arl_op(lam[i], matcl_op[i, j], op_dgp, reps; chart_choice=j)

            # Save temporary results
            matarl_op_tmp = hcat(matarl_op_tmp, res_op[1])
        end
        # Merge temporary results
        matarl_op = vcat(matarl_op, matarl_op_tmp)
    end
    # Add separator
    matarl_op = vcat(matarl_op, repeat(["---"], 1, 7))
    println(i)
end
matarl_op[2:end-1, :]

# --------------------------------------------------- #
# ---  Table 4 page 345 Weiss and Testik (2023)   --- #
# --------------------------------------------------- #
matarl_op = zeros(1, 6)

for i in 1:3 # lambda loop
    for alpha in [0.2, 0.4, 0.6, 0.8]
        matarl_op_tmp = ["λ = $(lam[i]); α = $alpha"]
        for j in 1:5 # chart loop

            # Make struct for out-of-control and compute ARL
            op_dgp = AAR1(alpha, Normal(0, 1))
            res_op = arl_op(lam[i], matcl_op[i, j], op_dgp, reps; chart_choice=j)

            # Save temporary results
            matarl_op_tmp = hcat(matarl_op_tmp, res_op[1])
        end
        # Merge temporary results
        matarl_op = vcat(matarl_op, matarl_op_tmp)
    end
    # Add separator
    matarl_op = vcat(matarl_op, repeat(["---"], 1, 6))
    println(i)
end
matarl_op[2:end-1, :]

# --------------------------------------------------- #
# ---  Table 5 page 345 Weiss and Testik (2023)   --- #
# --------------------------------------------------- #
matarl_op = zeros(1, 6)

for i in 1:3 # lambda loop
    for alpha in [0.15, 0.2, 0.25, 0.3]
        matarl_op_tmp = ["λ = $(lam[i]); α = $alpha"]
        for j in 1:5 # chart loop

            # Make struct for out-of-control and compute ARL
            op_dgp = QAR1(alpha, Normal(0, 1))
            res_op = arl_op(lam[i], matcl_op[i, j], op_dgp, reps; chart_choice=j)

            # Save temporary results
            matarl_op_tmp = hcat(matarl_op_tmp, res_op[1])
        end
        # Merge temporary results
        matarl_op = vcat(matarl_op, matarl_op_tmp)
    end
    # Add separator
    matarl_op = vcat(matarl_op, repeat(["---"], 1, 6))
    println(i)
end
matarl_op[2:end-1, :]

# --------------------------------------------------- #
# ---   Table 6 page 346 Weiss and Testik (2023)  --- #
# --------------------------------------------------- #
lam = 0.1
cl_select = [1.4601, 0.8405, 0.1115, 0.2529]
matarl_op = zeros(1, 5)

# Upper panel
for alpha in [0.20, 0.40, 0.60, 0.80, -0.20, -0.40, -0.60, -0.80]
    mat_arl_op_tmp = ["α = $alpha"]
    for (j, chart) in enumerate([1, 2, 3, 5]) # chart loop

        # Make struct for out-of-control and compute ARL
        op_dgp = AR1(alpha, Normal(0, 1))
        res_op = arl_op(lam, cl_select[j], op_dgp, reps; chart_choice=j)

        # Save temporary results
        mat_arl_op_tmp = hcat(mat_arl_op_tmp, res_op[1])
    end
    # Merge temporary results
    matarl_op = vcat(matarl_op, mat_arl_op_tmp)
    println(alpha)
end
matarl_op[2:end, :]

# Lower panel
matarl_op = zeros(1, 7)
cl_select = [1.4601, 0.8405, 0.1115, 0.3638, 0.2529, 0.4876]

for alpha in 0.1:0.1:0.6
    mat_arl_op_tmp = ["α = $alpha"]
    for j in 1:6 # chart loop

        # Make struct for out-of-control and compute ARL
        op_dgp = TEAR1(alpha, Exponential(1))
        res_op = arl_op(lam, cl_select[j], op_dgp, reps; chart_choice=j)

        # Save temporary results
        mat_arl_op_tmp = hcat(mat_arl_op_tmp, res_op[1])
    end
    # Merge temporary results
    matarl_op = vcat(matarl_op, mat_arl_op_tmp)
    println(alpha)
end
matarl_op[2:end, :]

# ------------------------------------------------------------- #
# --- Table 7 left, AR(1), page 346 Weiss and Testik (2023) --- #
# ------------------------------------------------------------- #

# acf_dgp = IC(Normal(0, 1))
# L0 = 370.0
# cl_init = 0.1
# jmin = 1
# reps = 10_000
# jmax = 6
# verbose = true
# cl = zeros(2)
# cl[1] = cl_acf(lam[1], L0, reps, acf_dgp, cl_init, jmin, jmax, verbose)
# cl[2] = cl_acf(lam[2], L0, reps, acf_dgp, cl_init, jmin, jmax, verbose)
# cl[1] = 1.142
# cl[2] = 0.6

# Left part using AR(1)
lam = [0.25, 0.1]
matarl_op = zeros(1, 3)
cl = [1.142, 0.6]

for alpha in [0.20, 0.40, 0.60, 0.80, -0.20, -0.40, -0.60, -0.80]
    mat_arl_op_tmp = ["α = $alpha"]
    for i in 1:2 # lambda loop

        # Make struct for out-of-control and compute ARL
        acf_dgp = AR1(alpha, Normal(0, 1))
        res_op = arl_acf(lam[i], cl[i], acf_dgp, reps)

        # Save temporary results
        mat_arl_op_tmp = hcat(mat_arl_op_tmp, res_op[1])
    end
    # Merge temporary results
    matarl_op = vcat(matarl_op, mat_arl_op_tmp)
    println(mat_arl_op_tmp)
end
matarl_op[2:end, :]

# Right part using TEAR(1)
lam = [0.25, 0.1]
matarl_op = zeros(1, 3)
cl = [1.142, 0.6]

for alpha in 0.1:0.1:0.6
    mat_arl_op_tmp = ["α = $alpha"]
    for i in 1:2 # lambda loop

        # Make struct for out-of-control and compute ARL
        acf_dgp = TEAR1(alpha, Exponential(1))
        res_op = arl_acf(lam[i], cl[i], acf_dgp, reps)

        # Save temporary results
        mat_arl_op_tmp = hcat(mat_arl_op_tmp, res_op[1])
    end
    # Merge temporary results
    matarl_op = vcat(matarl_op, mat_arl_op_tmp)
    println(alpha)
end
matarl_op[2:end, :]

# ------------------------------------------------------------- #
# --- Figures 1 and 2, page 347 Weiss and Testik (2023) --- #
# ------------------------------------------------------------- #
## Chemical process data from Box and Jenkins (1970) 
data = [47, 64, 23, 71, 38, 64, 55, 41, 59, 48, 71, 35, 57, 40, 58, 44, 80, 55, 37, 74, 51, 57, 50, 60, 45, 57, 50, 45,
    25, 59, 50, 71, 56, 74, 50, 58, 45, 54, 36, 54, 48, 55, 45, 57, 50, 62, 44, 64, 43, 52, 38, 59, 55, 41, 53, 49, 34, 35, 54,
    45, 68, 38, 50, 60, 39, 59, 40, 57, 54, 23]

# Figure 1, page 347 in Weiss and Testik (2023)
let
    fig = CairoMakie.Figure(size=(700, 300))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Batch", ylabel="Yields xₜ", yticks=[30, 40, 50, 60, 70, 80], xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false)
    CairoMakie.lines!(ax1, 1:70, data)
    CairoMakie.hist(fig[1, 2], data, bins=20:10:80, color=:grey, strokewidth=1, strokecolor=:black)
    CairoMakie.save("Figure_1_Weiss_Testik(2023).pdf", fig)
end


# Figure 2, page 347 in Weiss and Testik (2023)

stats_all = Matrix{Float64}(undef, 68, 8)

# H charts
stats_all[:, 1] = stat_op(data, 0.1, 1)
stats_all[:, 2] = stat_op(data, 0.05, 1)

# Hex charts
stats_all[:, 3] = stat_op(data, 0.1, 2)
stats_all[:, 4] = stat_op(data, 0.05, 2)

# Delta charts
stats_all[:, 5] = stat_op(data, 0.1, 3)
stats_all[:, 6] = stat_op(data, 0.05, 3)
# tau charts
stats_all[:, 7] = stat_op(data, 0.1, 5)
stats_all[:, 8] = stat_op(data, 0.05, 5)

let
    fig = CairoMakie.Figure(size=(700, 1000))
    # Hex chart lambda = 0.1
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="", ylabel="H chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[1.3, 1.4, 1.5, 1.6, 1.7, 1.8])

    CairoMakie.lines!(ax1, 3:70, stats_all[:, 1])
    CairoMakie.scatter!(ax1, 3:70, stats_all[:, 1])
    CairoMakie.hlines!(ax1, 1.4601, color=:"red", label="Control limits")
    CairoMakie.hlines!(ax1, log(6), color=:"red", label="Control limits")

    # H chart lambda = 0.05
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel="", ylabel="H chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[1.5, 1.6, 1.7, 1.8])

    CairoMakie.lines!(ax2, 3:70, stats_all[:, 2])
    CairoMakie.scatter!(ax2, 3:70, stats_all[:, 2])
    CairoMakie.hlines!(ax2, 1.6356, color=:"red", label="Control limits")
    CairoMakie.hlines!(ax2, log(6), color=:"red", label="Control limits")

    # Hex chart lambda = 0.1
    ax3 = CairoMakie.Axis(fig[2, 1], xlabel="", ylabel="Hex chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[0.82, 0.84, 0.86, 0.88, 0.90, 0.92])

    CairoMakie.lines!(ax3, 3:70, stats_all[:, 3])
    CairoMakie.scatter!(ax3, 3:70, stats_all[:, 3])
    CairoMakie.hlines!(ax3, 0.8405, color=:"red", label="Control limits")
    CairoMakie.hlines!(ax3, 5 * log(6 / 5), color=:"red", label="Control limits")

    # Hex chart lambda = 0.05
    ax4 = CairoMakie.Axis(fig[2, 2], xlabel="", ylabel="Hex chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[0.86, 0.88, 0.90, 0.92])

    CairoMakie.lines!(ax4, 3:70, stats_all[:, 4])
    CairoMakie.scatter!(ax4, 3:70, stats_all[:, 4])
    CairoMakie.hlines!(ax4, 0.88017, color=:"red", label="Control limits")
    CairoMakie.hlines!(ax4, 5 * log(6 / 5), color=:"red", label="Control limits")
    # Delta chart lambda = 0.1
    ax5 = CairoMakie.Axis(fig[3, 1], xlabel="", ylabel="Δ chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[0, 0.05, 0.1, 0.15])

    CairoMakie.lines!(ax5, 3:70, stats_all[:, 5])
    CairoMakie.scatter!(ax5, 3:70, stats_all[:, 5])
    CairoMakie.hlines!(ax5, 0.1115, color=:"red", label="Control limits")
    # Delta chart lambda = 0.05
    ax6 = CairoMakie.Axis(fig[3, 2], xlabel="", ylabel="Δ chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1])
    CairoMakie.lines!(ax6, 3:70, stats_all[:, 6])
    CairoMakie.scatter!(ax6, 3:70, stats_all[:, 6])
    CairoMakie.hlines!(ax6, 0.05125, color=:"red", label="Control limits")
    CairoMakie.hlines!(ax6, 0, color=:"red", label="Center line")
    # tau chart lambda = 0.1
    ax7 = CairoMakie.Axis(fig[4, 1], xlabel="t", ylabel="τ chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    CairoMakie.lines!(ax7, 3:70, stats_all[:, 7])
    CairoMakie.scatter!(ax7, 3:70, stats_all[:, 7])
    CairoMakie.hlines!(ax7, [-0.2529, 0.2529], color=:"red", label="Control limits")
    CairoMakie.hlines!(ax7, 0, color=:"red", label="Center line")
    # tau chart lambda = 0.05
    ax8 = CairoMakie.Axis(fig[4, 2], xlabel="t", ylabel="τ chart", xticks=[0, 10, 20, 30, 40, 50, 60, 70],
        xgridvisible=false, ygridvisible=false, yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    CairoMakie.lines!(ax8, 3:70, stats_all[:, 8])
    CairoMakie.scatter!(ax8, 3:70, stats_all[:, 8])
    CairoMakie.hlines!(ax8, [-0.16775, 0.16775], color=:"red", label="Control limits")
    CairoMakie.hlines!(ax8, 0, color=:"red", label="Center line", :dashed)
    CairoMakie.save("Figure_2_Weiss_Testik(2023).pdf", fig)
end

#end


# --------------------------------------------------- #
# ---       Extra: Run Simulation with MA1        --- #
# --------------------------------------------------- #

# Simulate MA1 process
#matarl_op_tmp = Vector{Float64}
matarl_op = zeros(1, 5)
lam = [0.25]

for i in 1 # lambda
    for alpha in [0.5, 0.6, 0.8]
        matarl_op_tmp = ["λ = $(lam[i]); α = $alpha"]
        for j in [1, 2, 3, 5] # chart loop

            # Make struct for out-of-control and compute ARL 
            op_dgp = MA1(alpha, Normal(0, 1))
            res_op = arl_op(lam, cl_select[j], op_dgp, reps; chart_choice=j)

            # Save temporary results
            matarl_op_tmp = hcat(matarl_op_tmp, res_op[1])
        end
        # Merge temporary results
        matarl_op = vcat(matarl_op, matarl_op_tmp)
    end
    # Add separator
    matarl_op = vcat(matarl_op, repeat(["---"], 1, 5))
    println(i)
end
matarl_op[2:end-1, :]



# --------------------------------------------------- #
# ---       Extra: Run Simulation with MA2        --- #
# --------------------------------------------------- #
matarl_op = zeros(1, 5)
lam = [0.25]
reps = 10_000

for i in 1 # lambda
    for alpha in [0.5, 0.6, 0.8]
        matarl_op_tmp = ["λ = $(lam[i]); α = $alpha"]
        for j in [1, 2, 3, 5] # chart loop

            # Make struct for out-of-control and compute ARL 
            op_dgp = MA2(0, alpha, Normal(0, 1))
            res_op = arl_op(lam[i], matcl_op[i, j], reps, op_dgp, j)

            # Save temporary results
            matarl_op_tmp = hcat(matarl_op_tmp, res_op[1])
        end
        # Merge temporary results
        matarl_op = vcat(matarl_op, matarl_op_tmp)
    end
    # Add separator
    matarl_op = vcat(matarl_op, repeat(["---"], 1, 5))
    println(i)
end
matarl_op[2:end-1, :]