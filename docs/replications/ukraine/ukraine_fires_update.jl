
using DataFrames
using DataFramesMeta
using OrdinalPatterns
using Dates
using CSV
using CategoricalArrays
using Makie
using CairoMakie

# Load the data
cd(@__DIR__)
ukraine_fires = CSV.read("ukraine_war_fires_25_01_25.csv", DataFrame)
ukraine_regions = CSV.read("ukr_regions.csv", DataFrame, decimal=',')

# Rename column names
rename!(ukraine_fires, :LATITUDE => :latitude, :LONGITUDE => :longitude)

# Select specific columns
select!(ukraine_fires, :date, :year, :latitude, :longitude, r"fire")

# Add 'week_year' column 
@chain ukraine_fires begin
  @rtransform!(:week = string(week(:date)))
  @rtransform!(:week = ifelse(length(:week) == 1, string("0", :week), :week))
  @rtransform!(:year_week = string(:year, "_", :week))
  select!(:date, :year_week, All())
end

chart_choice = 3
M = 41    # grid size for M (latitude)
N = 26    # grid size for N (longitude)
N_charts = Int(1e3) # Number of charts
latitude_lower = 45.75651 
latitude_upper = 50.43185
longitude_lower = 31.50439
longitude_upper = 40.15952

# Prepare data and compute number of fires in each bin
df_prepared = @chain ukraine_fires begin
  # Subset data frame based on long-and latitude for east ukraine  
  @subset(:latitude .>= latitude_lower,
    :latitude .<= latitude_upper,
    :longitude .>= longitude_lower,
    :longitude .<= longitude_upper)
  # Make bins for longitude and latitude        
  @transform(:lat_bin = cut(:latitude, collect(range(latitude_lower, latitude_upper, length=M + 1)), labels=1:M))
  @transform(:long_bin = cut(:longitude, collect(range(longitude_lower, longitude_upper, length=N + 1)), labels=1:N))
  @transform(:lat_bin = :lat_bin.refs) # Use integers
  @transform(:long_bin = :long_bin.refs) # Use integers
  # Compute sum of fires in each bin
  groupby([:year_week, :lat_bin, :long_bin])
  combine(:war_fire => sum => :sum_fire) # :war_fire -> determines whether this specific fire is assessed as war-related (0 or 1)
  # Add 'year' and 'week' columns as integers
  @rtransform(:year = parse(Int, :year_week[1:4]))
  @rtransform(:week = parse(Int, :year_week[6:7]))
  # Subset 
  @rsubset!(:year >= 2023, :year <= 2024)
end

# Count number of war-related fires in eastern Ukraine between 2023 and 2024
sum(df_prepared.sum_fire)

# Create a vector with all possible week-year combinations
week_vec = String[]
for i in sort(unique(df_prepared.week)) # 1:52
  if i < 10
    push!(week_vec, string("0", i))
  else
    push!(week_vec, string(i))
  end
end
dates_sort = reduce(vcat, [[string("2023_", x), string("2024_", x)] for x in week_vec]) |> sort

# Create grid-matrices for all possible lat and long bins. 
# Fill fire column :sum_fire with 0
add_df = DataFrame(lat_bin=Int[], long_bin=Int[], sum_fire=Int[])
foreach(x -> push!(add_df, x), Iterators.product(1:M, 1:N, 0))
tuple_add_df = Tuple.(eachrow(add_df[:, 1:2]))

# Create matrices for each week based on the prepared data frame
all_mats = map(dates_sort) do i
  # Subset data frame based on year-week
  df_tmp = df_prepared[df_prepared.year_week.==i, [:lat_bin, :long_bin, :sum_fire]]
  # Convert data frames to tuples to find which rows from 'add_df' to keep 
  tuple_df_tmp = Tuple.(eachrow(df_tmp[:, 1:2]))
  # Find rows from 'add_df' to keep
  index = map(x -> x ∉ tuple_df_tmp, tuple_add_df)
  # Add rows from 'add_df' to 'df_tmp' that are not already in 'df_tmp'
  df_tmp = vcat(df_tmp, add_df[index, :])
  # Make data frame wider, sort the rows based on 'lat_bin' and remove column 'lat_bin' for conversion  
  df_tmp_wide = sort(unstack(df_tmp, :lat_bin, :long_bin, :sum_fire), :lat_bin)[:, 2:end]
  # Get column names and sort them ascendingly  
  col_names = names(df_tmp_wide)[parse.(Int, names(df_tmp_wide))|>sortperm]
  # Select columns based on sorted column names  
  select!(df_tmp_wide, col_names)
  # Make data frame wider and convert to Matrix
  Matrix(df_tmp_wide)[M:-1:1, :] # flip y-axis
end

# Pre-allocate 3d array of type Float64
mat_all = zeros(Float64, M, N, size(dates_sort, 1))#52)

# Fill the 3d array with the matrices and add noise
for i in axes(mat_all, 3)
  mat_all[:, :, i] = convert.(Float64, all_mats[i])
end

# Create dates for 2023 and 2024 based on week
dates_23 = map(x -> Date(date -> week(date) == x, 2023, 01, 01), 1:52)
dates_24 = map(x -> Date(date -> week(date) == x, 2024, 01, 01), 1:52)
dates = vcat(dates_23, dates_24)


dates_ym = map(x -> string(x)[1:4] * "-" * string(week(x)), dates)

# ---------------------------------------------------------#
#                    SOP statistics                        #
# ---------------------------------------------------------#
d1_d2_vec = Iterators.product(1:1, 1:1) |> collect

# # Compute the critical limits for the EWMA chart
# d1_d2_crit_ewma = zeros(5, 5)
# d1_d2_shewart = zeros(5, 5)
# for d1d2 in d1_d2_vec 
#   d1 = d1d2[1]
#   d2 = d1d2[2]
#   cl = cl_sop(
#     ICSP(41, 25, Normal(0, 1)), 0.1, 370, 0.01, d1, d2, 10^3;
#     chart_choice=3, jmin=4, jmax=7, verbose=true
#   )
#   d1_d2_crit_ewma[d1, d2] = cl
# end
# # Verify critical limits
# for d1d2 in d1_d2_vec 
#   d1 = d1d2[1]
#   d2 = d1d2[2]
#   cl = cl_sop(
#     ICSP(41, 25, Normal(0, 1)), 1, 370, 0.05, d1, d2, 10^3;
#     chart_choice=3, jmin=4, jmax=7, verbose=true
#   )
#   d1_d2_shewart[d1, d2] = cl
# end


lam = [0.1, 1]
d1_d2_crit_ewma = [0.01009]
d1_d2_shewart = [0.04867]

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
      xlabel="Year-Week",
      title=fig_title[i],
      titlefont=:regular,
      xaxisposition=:bottom,
      yreversed=false,
      width=350,
      height=250,
      xticks=(1:19:104, dates_ym[1:19:104])
    )

    if lam[i] == 0.1
      cl = d1_d2_crit_ewma[d1, d2] #cl_sop_crit[i]
    elseif lam[i] == 1
      cl = d1_d2_shewart[d1, d2]
    end    
    lines!(ax, ooc_vec, color=:black, label="Typical run")
    lines!(ax, mean_vec, color=:blue, label="Mean of runs")
    hlines!([-cl, cl], color=:"red", label="Control limits")

    # Add legend
    if i == 2
      Legend(fig[2, 1:2], ax, labelsize=14, framecolor=:white, orientation=:horizontal)
    end

  end

  # Save figure
  resize_to_layout!(fig)
  save("Figure5_$(d1)_$(d2).pdf", fig)

end

##save("Figure5_4321.pdf", fig)
#save("Figure5_update.pdf", fig)
#


# -----------------------------------------------------------------------------
#                           Makie heatmaps
# -----------------------------------------------------------------------------
#Random.seed!(123)
v = [1, 39]
ic_mats = log.(cat((all_mats[[1, 2, 53]])..., dims=3))
oc_mats = log.(cat(all_mats[[33, 39, 92]]..., dims=3))
maps = cat(ic_mats, oc_mats, dims=3)

title_string = ["2023 - 01", "2023 - 02", "2024 - 01", "2023 - 33", "2023 - 39", "2024 - 40"]
maps[maps.==-Inf] .= 0

extremas = map(extrema, maps)
global_min = minimum(t -> first(t), extremas)
global_max = maximum(t -> last(t), extremas)

# These limits have to be shared by the maps and the colorbar
clims = (global_min, global_max)
cm = CairoMakie.cgrad([:white, :firebrick1, :darkred])

borders_x = collect(0:25)
borders_y = collect(40:-1:0)
## plots
let
  fig = Figure()
  k = 1
  for i in 1:2
    for j in 1:3
      ax = Axis(
        fig[i, j],
        title="$(title_string[k])",
        aspect=CairoMakie.AxisAspect(1),
        xaxisposition=:top,
        xticks=0:5:(N-1),
        yticks=0:10:(M-1),)
      ax.yreversed = true
      heatmap!(
        fig[i, j],
        borders_x,
        borders_y,
        transpose(maps[M:-1:1, :, k]);
        colorrange=clims,
        colormap=cm,
      )
      k += 1
    end
  end
  cb = Colorbar(fig[:, n_cols+1]; limits=clims, colormap=cm)
  resize_to_layout!(fig)
  fig
  save("ukraine_heatmap_update.pdf", fig)
end


# ---------------------------------------------------------#
#   BP statistics with w = 3 and 5, and nthreads = 10      #
# ---------------------------------------------------------#

# M = 41
# N = 26
# w = 5
# reps = 1_000
# lam = 1
# L0 = 370
# sp_dgp = ICSP(M, N, Normal(0, 1))
# crit_init = map(i -> stat_sop_bp(randn(M, N, 370), lam, w) |> last, 1:1_000) |> x -> quantile(x, 0.99)
# cl = cl_sop_bp(
#     sp_dgp, lam, L0, crit_init, w, reps; jmin=4, jmax=7, verbose=true
# )

lam = [0.1, 1]
cl_bp = [
  0.000362 0.00765; # w = 3, lam = 0.1, 1
  0.000786 0.01619  # w = 5, lam = 0.1, 1
]

fig_title = [
  "(a) BP-EWMA chart (w = 3, λ = 0.1)" "(b) BP-EWMA chart (w = 3, λ = 1)";
  "(c) BP-EWMA chart (w = 5, λ = 0.1)" "(d) BP-EWMA chart (w = 5, λ = 1)"
]


w = [3, 5]
lam = [0.1, 1]
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
      xlabel="Year-Week",
      title=fig_title[i, j],
      titlefont=:regular,
      xaxisposition=:bottom,
      yreversed=false,
      width=350,
      height=250,
      xticks=(1:19:104, dates_ym[1:19:104])
    )
    lines!(ax, ooc_vec, color=:black, label="Typical run")
    lines!(ax, mean_vec, color=:blue, label="Mean of runs")

    hlines!([cl_bp[i, j]], color=:"red", label="Control limits")

    # Add legend
    if i == 2 && j == 2
      Legend(fig[3, 1:2], ax, labelsize=14, framecolor=:white, orientation=:horizontal)
    end

  end
end

resize_to_layout!(fig)
fig


# save("Figure5_BP_4321.pdf", fig)
save("Figure5_BP_Ukraine.pdf", fig)
