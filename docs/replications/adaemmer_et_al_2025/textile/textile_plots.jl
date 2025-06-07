cd(@__DIR__)

using Pkg
Pkg.activate("../../.")

# Load module OrdinalPatterns only if not loaded already
if !isdefined(Main, :OrdinalPatterns)
  include("../../src/OrdinalPatterns.jl")
  using .OrdinalPatterns
end
using CodecZlib
using JLD2
using Random
using CairoMakie

# load data extracted from the R Packages "textile" and "spc4sts"
images_ic = load("icImgs.jld2")["large_array"]
images_oc = load("ocImgs.jld2")["large_array"]
mat_all = cat(images_ic, images_oc, dims=3)

fontsize_theme = Theme(fontsize=10)
set_theme!(fontsize_theme)
fig = Figure(size=(800, 600))

# -----------------------------------------------------------------------------
#                               SOP chart λ=0.1 N(0,1)
# -----------------------------------------------------------------------------
lam = 0.1
chart_choice = 3
cl_sop = 0.0012839 # see load("../applications/limits/cl_sop_textile_01.jld2")
stats = stat_sop(images_ic, lam, 1, 1; chart_choice=chart_choice, add_noise=false)
stat0 = chart_stat_sop([1 / 3, 1 / 3, 1 / 3], chart_choice)
lcl = stat0 - cl_sop
ucl = stat0 + cl_sop

ax1 = Axis(fig[1, 1], ylabel=L"\tilde{\tau}", xlabel="Image number",
  title="", xaxisposition=:bottom, xticks=[1, 10, 20, 30, 40, 50])
li = hlines!(stat0, color=:green, label="CL", linestyle=:dash)
li1 = lines!(ax1, 1:50, stats[1:50], color=:blue, label="Chart statistic")
li2 = hlines!([lcl, ucl], color=:red, label="Control limits")

axislegend(
  ax1,
  [
    li => (; linewidth=1.5, color=(:green)),
    li1 => (; linewidth=1.5, color=(:blue)),
    li2 => (; linewidth=1.5, color=(:red)),
  ],
  ["Center line", "Chart statistic", "Control limits"],
  merge=true, unique=true, position=:rc, labelsize=10
)

display(fig)

# -----------------------------------------------------------------------------
#                               SOP chart λ=0.1
# -----------------------------------------------------------------------------

lam = 0.1
L0 = 20
clinit = 0.001
reps = 10^6
cl = cl_sop_bootstrap(
  images_ic, lam, L0, clinit, 1, 1, reps;
  chart_choice=chart_choice, jmin=3, jmax=7, verbose=true
)
p_array = compute_p_array(images_ic, 1, 1; chart_choice=chart_choice)

p_array_mean = mean(p_array, dims=1)
p_array_mean = permutedims(p_array_mean, (2, 1))

stats_sop = stat_sop(
  images_oc,
  lam,
  1,
  1;
  type_freq_init=p_array_mean
)

stat0 = chart_stat_sop(p_array_mean, chart_choice)
lcl = chart_stat_sop(p_array_mean, chart_choice) - cl
ucl = chart_stat_sop(p_array_mean, chart_choice) + cl

# Use Makie to plots stats. Draw in red the critical limits  
ax2 = Axis(
  fig[1, 2],
  ylabel=L"\tilde{\tau}",
  xlabel="Image number",
  width=150,
  height=150,
  xticks=(1:6, string.(collect(95:100)))
)
lines!(ax2, stats_sop, color=:blue)
hlines!(ax2, stat0, color=:green, linestyle=:dash)
hlines!(ax2, [lcl, ucl], color=:red)

display(fig)
# -----------------------------------------------------------------------------
#                               SOP chart λ=1
# -----------------------------------------------------------------------------

lam = 1
cl = cl_sop_bootstrap(
  images_ic, lam, L0, clinit, 1, 1, reps;
  chart_choice=chart_choice, jmin=3, jmax=7, verbose=true
)
p_array = compute_p_array(images_ic, 1, 1; chart_choice=chart_choice)

p_array_mean = mean(p_array, dims=1)
p_array_mean = permutedims(p_array_mean, (2, 1))

stats_sop = stat_sop(
  images_oc,
  lam,
  1,
  1;
  type_freq_init=p_array_mean
)

stat0 = chart_stat_sop(p_array_mean, chart_choice)
lcl = chart_stat_sop(p_array_mean, chart_choice) - cl
ucl = chart_stat_sop(p_array_mean, chart_choice) + cl

ax3 = Axis(
  fig[1, 3],
  ylabel=L"\tilde{\tau}",
  xlabel="Image number",
  width=150,
  height=150,
  xticks=(1:6, string.(collect(95:100)))
)
lines!(ax3, stats_sop, color=:blue)
hlines!(ax3, stat0, color=:green, linestyle=:dash)
hlines!(ax3, [lcl, ucl], color=:red)

# combined plots

resize_to_layout!(fig)
colsize!(fig.layout, 1, Auto(1))
display(fig)

#save("textile_data_combined_plot.pdf", fig)

# -----------------------------------------------------------------------------
#                spatial textile data plot 
# -----------------------------------------------------------------------------
all_mats = cat(images_ic, images_oc, dims=3)

v = [1, 96, 97, 98]

n_rows = 1
n_cols = 4
maps = mat_all[:, :, v]
grid_M = 250
grid_N = 250

# get global extrema
extremas = map(extrema, maps)
global_min = minimum(t -> first(t), extremas)
global_max = maximum(t -> last(t), extremas)

# these limits have to be shared by the maps and the colorbar
clims = (global_min, global_max)
borders_x = collect(0:size(mat_all, 2)-1)
borders_y = collect(size(mat_all, 1)-1:-1:0)
cm = cgrad([:black, :grey, :white])

let
  fig = Figure()
  fontsize_theme = Theme(fontsize=10)
  set_theme!(fontsize_theme)
  k = 4
  for i in 1:1
    for j in 1:k
      kk = k * (i - 1) + j
      ax = Axis(
        fig[i, j],
        aspect=AxisAspect(1),
        xaxisposition=:top,
        xticks=0:50:(grid_N-1),
        yticks=0:50:(grid_M-1),
        title="Patch $(v[kk])"
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

  cb = Colorbar(fig[:, n_cols+1]; limits=clims, colormap=cm)
  rowsize!(fig.layout, 1, Aspect(1, 1))
  resize_to_layout!(fig)
  display(fig)
end