

#using Pkg;
using DataFrames
using DataFramesMeta
using OrdinalPatterns
using Dates
using CSV
using CategoricalArrays
using Makie
using CairoMakie
using Random

# Load the data
cd(@__DIR__)
ukraine_fires = CSV.read("ukraine_war_fires_12_04_24.csv", DataFrame)
ukraine_regions = CSV.read("ukr_regions.csv", DataFrame, decimal=',')

# Rename column names
rename!(ukraine_fires, :LATITUDE => :latitude, :LONGITUDE => :longitude)

# Select specific columns
select!(ukraine_fires, :date, :year, :latitude, :longitude, r"fire")

# Add 'week_year' column 
@chain ukraine_fires begin
  @rtransform!(:week = string(week(:date)))
  @rtransform!(:week = ifelse(length(:week) == 1, string("0", :week), :week))
  @rtransform!(:week_year = string(:year, "_", :week))
  select!(:date, :week_year, All())
end

chart_choice = 3
M = 41    # grid size for M (latitude, row of final count matrix)
N = 26    # grid size for N (longitude, column of final count matrix)
N_charts = Int(1e3) # Number of charts
lam = 0.1

# Prepare data and compute number of fires in each bin
df_prepared = @chain ukraine_fires begin
  # Subset data frame based on long-and latitude for east ukraine  
  @subset(:latitude .>= 45.75651,
    :latitude .<= 50.43185,
    :longitude .>= 31.50439,
    :longitude .<= 40.15952)
  # Make bins for longitude and latitude        
  @transform(:lat_bin = cut(:latitude, collect(range(45.75651, 50.43185, length=M + 1)), labels=1:M))
  @transform(:long_bin = cut(:longitude, collect(range(31.50439, 40.15952, length=N + 1)), labels=1:N))
  @transform(:lat_bin = :lat_bin.refs)
  @transform(:long_bin = :long_bin.refs)
  # Compute sum of fires in each bin
  groupby([:week_year, :lat_bin, :long_bin])
  combine(:war_fire => sum => :sum_fire)
  # Add 'year' and 'week' columns as integers
  @rtransform(:year = parse(Int, :week_year[1:4]))
  @rtransform(:week = parse(Int, :week_year[6:7]))
  # Subset 
  @rsubset!(:year == 2023)
end

# Count number of fires in eastern Ukraine
@chain ukraine_fires begin
  # Subset data frame based on long-and latitude for east ukraine  
  @subset(:latitude .>= 45.75651,
    :latitude .<= 50.43185,
    :longitude .>= 31.50439,
    :longitude .<= 40.15952)
  @rsubset!(:year == 2023)
end

# Use 52 weeks
dates_sort = 1:52

# Create grid-matrices for all possible lat and long bins. Fill fires with zero
add_df = DataFrame(lat_bin=UInt32[], long_bin=UInt32[], sum_fire=Int64[])
foreach(x -> push!(add_df, x), Iterators.product(1:M, 1:N, 0))
tuple_add_df = Tuple.(eachrow(add_df[:, 1:2]))

# Create matrices for each week based on the prepared data frame
all_mats = map(dates_sort) do i
  # Subset data frame based on week
  df_tmp = df_prepared[df_prepared.week.==i, [:lat_bin, :long_bin, :sum_fire]]
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
mat_all = zeros(Float64, M, N, 52)

# Fill the 3d array with the matrices and add noise
for i in axes(mat_all, 3)
  mat_all[:, :, i] = convert.(Float64, all_mats[i])
end


# Makie figure to save
lam = [0.1, 1]
cl = [0.01009, 0.04867]
fig = Figure()
fig_title = ["EWMA chart (λ = 0.1)", "Shewhart chart (λ = 1)"]
for i in 1:2
  # Compute the statistic 1000 times
  Random.seed!(123)
  results_all = map(x -> stat_sop(mat_all, lam[i], 1, 1; chart_choice=3, add_noise=true)', 1:1000)
  # Convert to matrix
  mapooc_stat_sops = vcat(results_all...)

  # Compute mean vector
  mean_vec = vec(mean(mapooc_stat_sops, dims=1))

  # Get L1 distance
  dist_vec = map(x -> sum(abs.(mean_vec - x)), eachrow(mapooc_stat_sops))
  ooc_vec = mapooc_stat_sops[sortperm(dist_vec)[1], :]

  # Compute the mean for each week and plot
  mean_vec = mean(vcat(results_all...), dims=1)[:]

  fontsize_theme = Theme(fontsize=14)
  set_theme!(fontsize_theme)

  ax = Axis(
    fig[1, i],
    ylabel="τ̃",
    xlabel="Week",
    title=fig_title[i],
    xaxisposition=:bottom,
    yreversed=false,
    width=350,
    height=250
  )
  lines!(ax, 1:length(ooc_vec), ooc_vec, color=:black, label="Typical run")
  lines!(ax, mean_vec, color=:blue, label="Mean of runs")
  hlines!([-cl[i], cl[i]], color=:"red", label="Control limits")

  # Add legend
  if i == 2
    Legend(fig[2, 1:2], ax, labelsize=14, framecolor = :white, orientation=:horizontal)
  end

end

resize_to_layout!(fig)
fig


# -----------------------------------------------------------------------------
#                           Makie heatmaps
# -----------------------------------------------------------------------------
#Random.seed!(123)
v = [1, 39]
ic_mats = log.(cat((all_mats[1:3])..., dims = 3))
oc_mats = log.(cat(all_mats[[31, 33, 39]]..., dims = 3))
maps = cat(ic_mats, oc_mats, dims = 3)

title_string = "Week " .* ["1", "2", "3", "31", "33", "39"]
maps[maps .== -Inf] .= 0

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
        aspect = CairoMakie.AxisAspect(1),
        xaxisposition = :top,
        xticks = 0:5:(N-1),
        yticks = 0:10:(M-1),

      )      
      ax.yreversed=true
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
  #save("ukraine_spatial_plot_combined_layout.pdf", fig) 
end
