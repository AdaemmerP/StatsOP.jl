cd(@__DIR__)

using Pkg
Pkg.activate("../../../.")

using OrdinalPatterns
using CodecZlib
using JLD2
using Random
using CairoMakie


# load data
mat_all = load("rainfall_data_2007_08_one_week.jld2")["large_array"]

# check for missing values in array all_mats
missing_values = sum(ismissing.(mat_all))
# replace missing values with 0
mat_all = coalesce.(mat_all, 0.0)

# chart parameter setup 
d1 = 1
d2 = 1
ic_start = 1
ic_end = 168

# ---------------------------------------------------------#
#                    SOP statistics                        #
# ---------------------------------------------------------#
d1_d2_vec = [(1, 1)]

lam = [0.1, 1]
d1_d2_crit_ewma = 0.0188163 # see load("../applications/limits/cl_sop_rain_01.jld2")
d1_d2_shewhart = 0.0920738 # see load("../applications/limits/cl_sop_rain_1.jld2")

for d1_d2 in d1_d2_vec
    d1 = d1_d2[1]
    d2 = d1_d2[2]

    for i in 1:2
        # Start figure
        fig = Figure()

        # Compute the statistic 1000 times
        Random.seed!(123)  # Random.seed!(4321) # 10
        results_all = map(x -> stat_sop(mat_all, lam[i], d1, d2; chart_choice=3, add_noise=true, noise_dist=Uniform(0, 0.1))', 1:1_000)

        # Convert to matrix
        mapooc_stat_sops = vcat(results_all...)

        # Compute mean vector
        mean_vec = vec(mean(mapooc_stat_sops, dims=1))

        # Get L1 distance
        dist_vec = map(x -> sum(abs.(mean_vec .- x)), eachrow(mapooc_stat_sops))
        ooc_vec = mapooc_stat_sops[sortperm(dist_vec)[1], :] # Get that sequence with the smallest distance

        # Makie figure to save
        fontsize_theme = Theme(fontsize=14)
        set_theme!(fontsize_theme)

        ax = Axis(
            fig[1, 1],
            ylabel=L"\tilde{\tau}",
            xlabel="Hour",
            xaxisposition=:bottom,
            yreversed=false,
            width=350,
            height=250,
            xticks=collect(0:24:168),
        )

        if lam[i] == 0.1
            cl = d1_d2_crit_ewma[d1, d2]
        elseif lam[i] == 1
            cl = d1_d2_shewhart[d1, d2]
        end

        for i in 2:1000
            lines!(ax, ic_start:ic_end, mapooc_stat_sops[i, ic_start:ic_end], color=(:grey, 0.05), label="Single runs")
        end
        li = lines!(ax, ic_start:ic_end, mapooc_stat_sops[1, ic_start:ic_end], color=(:grey, 0.05), label="Single runs")
        li1 = lines!(ax, ooc_vec, color=:black, label="Typical run")
        li2 = lines!(ax, mean_vec, color=:blue, label="Mean of runs")
        li3 = hlines!([-cl, cl], color=:"red", label="Control limits")

        # Add legend
        if i == 1
            axislegend(
                ax,
                [
                    li => (; linewidth=1.5, color=(:grey, 0.5)),
                    li1 => (; linewidth=1.5, color=(:black)),
                    li2 => (; linewidth=1.5, color=(:blue)),
                    li3 => (; linewidth=1.5, color=(:red)),
                ],
                ["Single runs", "Typical run", "Mean of runs", "Control limits"],
                merge=true, unique=true, position=:lb, labelsize=10
            )
        end

        # Save figure
        resize_to_layout!(fig)
        display(fig)

    end

end

# -----------------------------------------------------------------------------
#                           Makie heatmaps
# -----------------------------------------------------------------------------

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
    fig = Figure()
    k = 3
    for i in 1:2
        for j in 1:k
            kk = k * (i - 1) + j
            ax = Axis(
                fig[i, j],
                aspect=AxisAspect(1),
                xaxisposition=:top,
                xticks=0:2:(grid_N-1),
                yticks=0:5:(grid_M-1),
                title="Hour $(v[kk])"
            )
            ax.yreversed = true
            heatmap!(
                fig[i, j],
                borders_x,
                borders_y,
                maps[grid_M:-1:1, :, kk]';
                colorrange=clims,
                colormap=cm
            )
        end
    end
    # Add colorbar
    cb = Colorbar(fig[:, n_cols+1]; limits=clims, colormap=cm)

    resize_to_layout!(fig)
    display(fig)
end

# --------------------------- #
#   BP statistics with w = 3  #
# --------------------------- #

w = 3
lam = 0.1
YLS = [0, 0.01]
cl_bp = [
    0.00134464 # w = 3, lam = 0.1; see load("../applications/limits/cl_bp_rain_01.jld2")
]

let
    fig1 = Figure()
    fig2 = Figure()
    Random.seed!(123)
    # Compute the statistic 1000 times
    results_all = map(x -> stat_sop_bp(
            mat_all,
            lam,
            w,
            chart_choice=3,
            add_noise=true,
            noise_dist=Uniform(0, 0.1)
        )', 1:1_000)

    mapooc_stat_sops = vcat(results_all...)

    # Compute mean vector
    mean_vec = vec(mean(mapooc_stat_sops, dims=1))

    # Get L1 distance
    dist_vec = map(x -> sum(abs.(mean_vec - x)), eachrow(mapooc_stat_sops))
    ooc_vec = mapooc_stat_sops[sortperm(dist_vec)[1], :]

    fontsize_theme = Theme(fontsize=14)
    set_theme!(fontsize_theme)

    ## plot 1 (full y-axis)
    ax1 = Axis(
        fig1[1, 1],
        ylabel=L"\tilde{\tau}",
        xlabel="Hour",
        titlefont=:regular,
        xaxisposition=:bottom,
        yreversed=false,
        width=350,
        height=250,
        xticks=collect(0:24:168),
    )

    li = lines!(ax1, ic_start:ic_end, mapooc_stat_sops[1, ic_start:ic_end], color=(:grey, 0.05), label="Single runs")
    for i in 2:1000
        lines!(ax1, ic_start:ic_end, mapooc_stat_sops[i, ic_start:ic_end], color=(:grey, 0.05), label="Single runs")
    end
    li1 = lines!(ax1, ooc_vec, color=:black, label="Typical run")
    li2 = lines!(ax1, mean_vec, color=:blue, label="Mean of runs")
    li3 = hlines!(ax1, [cl_bp[1, 1]], color=:"red", label="Control limits")

    # Add legend
    axislegend(
        ax1,
        [
            li => (; linestyle=:solid, linewidth=1.5, color=(:grey, 0.5)),
            li1 => (; linestyle=:solid, linewidth=1.5, color=(:black)),
            li2 => (; linestyle=:solid, linewidth=1.5, color=(:blue)),
            li3 => (; linestyle=:solid, linewidth=1.5, color=(:red)),
        ],
        ["Single runs", "Typical run", "Mean of runs", "Control limits"],
        merge=true, unique=true, position=:lt, labelsize=10
    )

    ## plot 2 (zoomed y-axis)
    ax2 = Axis(
        fig2[1, 1],
        ylabel=L"\tilde{\tau}",
        xlabel="Hour",
        titlefont=:regular,
        xaxisposition=:bottom,
        yreversed=false,
        width=350,
        height=250,
        xticks=collect(0:24:168),
    )

    li = lines!(ax2, ic_start:ic_end, mapooc_stat_sops[1, ic_start:ic_end], color=(:grey, 0.05), label="Single runs")
    for i in 2:1000
        lines!(ax2, ic_start:ic_end, mapooc_stat_sops[i, ic_start:ic_end], color=(:grey, 0.05), label="Single runs")
    end
    li1 = lines!(ax2, ooc_vec, color=:black, label="Typical run")
    li2 = lines!(ax2, mean_vec, color=:blue, label="Mean of runs")
    li3 = hlines!(ax2, [cl_bp[1, 1]], color=:"red", label="Control limits")
    ylims!(ax2, [0 0.01])

    # summarize alarm point results
    #println(transpose(collect(136:140)))
    #println(map(x -> sum(x .>= cl_bp[1, 1]), mean_vec[136:140]))

    resize_to_layout!(fig1)
    resize_to_layout!(fig2)

    display(fig1)
    display(fig2)
end
