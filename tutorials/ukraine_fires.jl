

using Pkg;
cd(@__DIR__)
Pkg.activate(".");
using DataFrames
using DataFramesMeta
using OrdinalPatterns
using Dates
using CSV
using CategoricalArrays
using Plots

# Load the data
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
      @subset(:latitude  .>= 45.75651,
              :latitude  .<= 50.43185,
              :longitude .>= 31.50439,
              :longitude .<= 40.15952)
    # Make bins for longitude and latitude        
      @transform(:lat_bin  = cut(:latitude,  collect(range(45.75651, 50.43185, length=M+1)), labels = 1:M))
      @transform(:long_bin = cut(:longitude, collect(range(31.50439, 40.15952, length=N+1)), labels = 1:N))      
      @transform(:lat_bin  = :lat_bin.refs)
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
      @subset(:latitude  .>= 45.75651,
              :latitude  .<= 50.43185,
              :longitude .>= 31.50439,
              :longitude .<= 40.15952)
      @rsubset!(:year == 2023)  
  end

# Use 52 weeks
dates_sort = 1:52 

# Create grid-matrices for all possible lat and long bins. Fill fires with zero
  add_df = DataFrame(lat_bin = UInt32[], long_bin = UInt32[], sum_fire = Int64[]) 
  foreach(x -> push!(add_df, x), Iterators.product(1:M, 1:N, 0))
  tuple_add_df = Tuple.(eachrow(add_df[:, 1:2]))

# Create matrices for each week based on the prepared data frame
all_mats = map(dates_sort) do i  
  # Subset data frame based on week
    df_tmp = df_prepared[df_prepared.week .== i, [:lat_bin, :long_bin, :sum_fire]] 
  # Convert data frames to tuples to find which rows from 'add_df' to keep 
    tuple_df_tmp = Tuple.(eachrow(df_tmp[:, 1:2]))
  # Find rows from 'add_df' to keep
    index = map(x -> x ∉ tuple_df_tmp, tuple_add_df) 
  # Add rows from 'add_df' to 'df_tmp' that are not already in 'df_tmp'
    df_tmp = vcat(df_tmp, add_df[index, :])
  # Make data frame wider, sort the rows based on 'lat_bin' and remove column 'lat_bin' for conversion  
    df_tmp_wide = sort(unstack(df_tmp, :lat_bin, :long_bin, :sum_fire), :lat_bin)[:, 2:end]
  # Get column names and sort them ascendingly  
    col_names = names(df_tmp_wide)[parse.(Int, names(df_tmp_wide)) |> sortperm] 
  # Select columns based on sorted column names  
    select!(df_tmp_wide, col_names)    
  # Make data frame wider and convert to Matrix
    Matrix(df_tmp_wide)[M:-1:1, :] # flip y-axis
end  

# Pre-allocate 3d array of type Float64
empty = zeros(Float64, M, N, 52)

# Fill the 3d array with the matrices and add noise
for i in axes(empty, 3)
  empty[:, :, i] = convert.(Float64, all_mats[i]) 
end

# Compute the statistic 1000 times
results_all = map(x -> stat_sop(empty, true, 1, 3)', 1:1000)

# Compute the mean for each week and plot
mean(vcat(results_all...), dims = 1)[:] |> plot
