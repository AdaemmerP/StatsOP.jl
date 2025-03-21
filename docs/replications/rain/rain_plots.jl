# Load packages
using OrdinalPatterns
using CodecZlib
using JLD2
using Random
using CairoMakie


# load data
mat_all = load("docs/replications/rain/rain_data_2007_08_self_extracted.jld2")["large_array"]

# check for missing values in array all_mats
missing_values = sum(ismissing.(mat_all))
# replace missing values with 0
mat_all = coalesce.(mat_all, 0.0)

#M = 27
#N = 12
#d1 = 1
#d2 = 1
#reps = 1_000
#lam = 1
#L0 = 370
#sp_dgp = ICSTS(M, N, Normal(0, 1))
#crit_init = map(i -> stat_sop(randn(M, N, 370), lam, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
#reps = 100_000
#cl = cl_sop(
#    sp_dgp, lam, L0, crit_init, d1, d2, reps; jmin=4, jmax=7, verbose=true
#)

# chart parameter setup 
d1 = 1
d2 = 1
ic_start = 1
ic_end = 168

# ---------------------------------------------------------#
#                    SOP statistics                        #
# ---------------------------------------------------------#
d1_d2_vec = Iterators.product(1:1, 1:1) |> collect

lam = [0.1, 1]
d1_d2_crit_ewma = [0.018819]
d1_d2_shewart = [0.092075]

let
    for d1_d2 in d1_d2_vec
        fig = Figure()
        d1 = d1_d2[1]
        d2 = d1_d2[2]
        fig_title = ["EWMA chart (d1 = $d1, d2 = $d2, λ = 0.1)", "EWMA chart (d1 = $d1, d2 = $d2, λ = 1)"]

        for i in 1:2
            # Compute the statistic 1000 times
            Random.seed!(123)  # Random.seed!(4321) # 10
            results_all = map(x -> stat_sop(mat_all, lam[i], d1, d2; chart_choice=3, add_noise=true)', 1:1_000)

            # Convert to matrix
            mapooc_stat_sops = vcat(results_all...)

            # Compute mean vector
            mean_vec = vec(mean(mapooc_stat_sops, dims=1))

            # Get L1 distance
            dist_vec = map(x -> sum(abs.(mean_vec - x)), eachrow(mapooc_stat_sops))
            ooc_vec = mapooc_stat_sops[sortperm(dist_vec)[1], :]

            # Makie figure to save
            fontsize_theme = Theme(fontsize=14)
            set_theme!(fontsize_theme)

            ax = Axis(
                fig[1, i],
                ylabel="τ̃",
                xlabel="Hour",
                title=fig_title[i],
                titlefont=:regular,
                xaxisposition=:bottom,
                yreversed=false,
                width=350,
                height=250,
                xticks=collect(0:24:168),
            )

            if lam[i] == 0.1
                cl = d1_d2_crit_ewma[d1, d2] #cl_sop_crit[i]
            elseif lam[i] == 1
                cl = d1_d2_shewart[d1, d2]
            end
            for i in 1:1000
                lines!(ax, ic_start:ic_end, mapooc_stat_sops[i, ic_start:ic_end], color=:grey, alpha=0.05, label="Single runs")
            end
            lines!(ax, ooc_vec, color=:black, label="Typical run")
            lines!(ax, mean_vec, color=:blue, label="Mean of runs")
            hlines!([-cl, cl], color=:"red", label="Control limits")

            # Add legend
            if i == 2
                Legend(fig[2, 1:2], ax, labelsize=14, framecolor=:white, orientation=:horizontal, merge=true, unique=true)
            end

            # summarize alarm point results
            println(map(x -> sum((x .>= cl .|| x .<= -cl)), ooc_vec[136:140]))

        end

        # Save figure
        resize_to_layout!(fig)
        save("Figure3_$(d1)_$(d2).pdf", fig)

    end
end

# ---------------------------------------------------------#
#   BP statistics with w = 3 and 5, and nthreads = 28      #
# ---------------------------------------------------------#

#M = 27
#N = 12
#w = 5
#reps = 1_000
#lam = 1
#L0 = 370
#sp_dgp = ICSTS(M, N, Normal(0, 1))
#crit_init = map(i -> stat_sop_bp(randn(M, N, 370), lam, w) |> last, 1:1_000) |> x -> quantile(x, 0.99)
#reps = 100_000
#cl = cl_sop_bp(
#    sp_dgp, lam, L0, crit_init, w, reps; jmin=4, jmax=7, verbose=true
#)

lam = [0.1, 1]
clbp = [
    0.0013443 0.0287518; # w = 3, lam = 0.1, 1
    0.0031717 0.0668780  # w = 5, lam = 0.1, 1
]

fig_title = [
    "(a) BP-EWMA chart (w = 3, λ = 0.1)" "(b) BP-EWMA chart (w = 3, λ = 1)";
    "(c) BP-EWMA chart (w = 5, λ = 0.1)" "(d) BP-EWMA chart (w = 5, λ = 1)"
]


w = [3, 5]
lam = [0.1, 1]
YLS = [(0, 0.01), (0, 0.02), (0, 0.15), (0, 0.25)]

let
    fig = Figure()
    for (i, w) in enumerate(w)
        for j in 1:2

            # Compute the statistic 1000 times
            Random.seed!(123) #Random.seed!(4321) #
            results_all = map(x -> stat_sop_bp(
                    mat_all,
                    lam[j],
                    w,
                    chart_choice=3,
                    add_noise=true
                )', 1:1_000)

            # Convert to matrix
            mapooc_stat_sops = vcat(results_all...)

            # Compute mean vector
            mean_vec = vec(mean(mapooc_stat_sops, dims=1))

            # Get L1 distance
            dist_vec = map(x -> sum(abs.(mean_vec - x)), eachrow(mapooc_stat_sops))
            ooc_vec = mapooc_stat_sops[sortperm(dist_vec)[1], :]

            # Makie figure to save
            fontsize_theme = Theme(fontsize=14)
            set_theme!(fontsize_theme)

            ax = Axis(
                fig[i, j],
                ylabel="τ̃",
                xlabel="Hour",
                title=fig_title[i, j],
                titlefont=:regular,
                xaxisposition=:bottom,
                yreversed=false,
                width=350,
                height=250,
                xticks=collect(0:24:168),
            )
            for i in 1:1000
                lines!(ax, ic_start:ic_end, mapooc_stat_sops[i, ic_start:ic_end], color=:grey, alpha=0.05, label="Single runs")
            end
            lines!(ax, ooc_vec, color=:black, label="Typical run")
            lines!(ax, mean_vec, color=:blue, label="Mean of runs")

            hlines!([clbp[i, j]], color=:"red", label="Control limits")

            # Add legend
            if i == 2 && j == 2
                Legend(fig[3, 1:2], ax, labelsize=14, framecolor=:white, orientation=:horizontal, merge=true, unique=true)
            end

            # Individual plot limits
            if i == 1 && j == 1
                ylims!(ax, YLS[i+2*(j-1)])
            end
            if i == 1 && j == 2
                ylims!(ax, YLS[i+2*(j-1)])
            end
            if i == 2 && j == 1
                ylims!(ax, YLS[i+2*(j-1)])
            end
            if i == 2 && j == 2
                ylims!(ax, YLS[i+2*(j-1)])
            end

            # summarize alarm point results

            println(map(x -> sum(x .>= clbp[i, j]), mean_vec[136:140]))

        end

    end

    resize_to_layout!(fig)
    fig

    save("Figure3_v5_BP_rain.pdf", fig)
end

# -----------------------------------------------------------------------------
#                           Makie heatmaps
# -----------------------------------------------------------------------------

# -------------------------------------------- #
# -- Plot selected matrices of hourly -------- #
# -----precipitation sums in August 2007 ----- #
# ------------ Figure 2b) ---------- --------- #
# -------------------------------------------- #

# compute range of values in all_mats
min_val = minimum(mat_all)
max_val = maximum(mat_all)

Random.seed!(1234)
v = [1, 138, 139, 140, 141, 142]
n_rows = 2
n_cols = 3
maps = mat_all[:, :, v]

# get global extrema
extremas = map(extrema, maps)
global_min = minimum(t -> first(t), extremas)
global_max = maximum(t -> last(t), extremas)
# these limits have to be shared by the maps and the colorbar
clims = (global_min, global_max)
borders_x = collect(0:size(mat_all, 2)-1)
borders_y = collect(size(mat_all, 1)-1:-1:0)
cm = CairoMakie.cgrad([:white, :deepskyblue3, :navyblue])
grid_M = 26
grid_N = 11


let
    fig = CairoMakie.Figure()
    k = 3
    for i in 1:2
        for j in 1:k
            kk = k * (i - 1) + j
            ax = CairoMakie.Axis(
                fig[i, j],
                aspect=CairoMakie.AxisAspect(1),
                xaxisposition=:top,
                xticks=0:2:(grid_N-1),
                yticks=0:5:(grid_M-1),
                title="Hour $(v[kk])"
            )
            ax.yreversed = true
            #ax = Axis(fig[i, j], xaxisposition=:top, yreversed=false, title="Hour $(v[kk])")
            CairoMakie.heatmap!(
                fig[i, j],
                borders_x,
                borders_y,
                maps[grid_M:-1:1, :, kk]';
                colorrange=clims,
                colormap=cm
            )
        end
    end
    #resize_to_layout!(gb)
    cb = CairoMakie.Colorbar(fig[:, n_cols+1]; limits=clims, colormap=cm)
    # Add labels

    CairoMakie.resize_to_layout!(fig)
    fig
    save("rain_data_spatial_plot_single.pdf", fig)
end
