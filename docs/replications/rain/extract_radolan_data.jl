# The data for the matrices is extracted from the RADOLAN data and stored in a 3D array.It is extracted using the following code:
# https://opendata.dwd.de/climate_environment/CDC/help/landing_pages/doi_landingpage_RADKLIM_RW_V2017.002-de.html
# download raw data from
# https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/reproc/2017_002/asc/2007/RW2017.002_200708_asc.tar.gz
# extract asc files from "RW_2017.002_20070803_0050" to "RW_2017.002_20070809_2350" into folder

using RCall
using JLD2

R"""
wd <- "/home/pwitten/clones/OrdinalPatterns.jl/docs/replications/rain/raw_data/"
setwd(wd)
fn <- list.files(pattern = "^[RW]")
"""

# use parSapply to read in files in parallel
# not working with RCall
R"""
#nc <- 20
#cl <- parallel::makeCluster(nc)
#parallel::clusterEvalQ(cl, {library(dplyr); library(tidyr)})
#parallel::clusterExport(cl, c("wd", "fn"), envir=environment())

#res <- parallel::parLapplyLB(cl, 1:length(fn), function(i) {
#  df <- read.csv(paste0(wd, fn[i]), skip = 6, header = FALSE, sep = " ")
#  df2 <- data.frame(df)
#  colnames(df2) <- 1:ncol(df2)
#  rownames(df2) <- 1:nrow(df2)
#  df2[df2 == -9999] <- NA
#  df2$Y <- rownames(df2)
#  ## reverse order of rows for plotting
#  df2$Y <- rev(df2$Y)
#  df2$X <- rev(df2$X)
#  df2 |>
#  pivot_longer(-Y, names_to = "X", values_to = "Z") |>
#  mutate(X = as.numeric(X), Y = as.numeric(Y), Z = as.numeric(Z)) |>
#  filter(Y%in%c(489:515), X%in%c(656:667)) |>
#  pivot_wider(names_from=X,values_from=Z) |>
#  select(-Y)
#})
#parallel::stopCluster(cl)
#arr <- array(unlist(res), c(27, 12, length(fn)))
"""

# or single core version

R"""
library(dplyr)
library(tidyr)

res <- lapply(1:length(fn), function(i) {
  df <- read.csv(paste0(wd, fn[i]), skip = 6, header = FALSE, sep = " ")
  df2 <- data.frame(df)
  colnames(df2) <- 1:ncol(df2)
  rownames(df2) <- 1:nrow(df2)
  df2[df2 == -9999] <- NA
  df2$Y <- rownames(df2)
  ## reverse order of rows for plotting
  df2$Y <- rev(df2$Y)
  df2$X <- rev(df2$X)
  df2 |>
  pivot_longer(-Y, names_to = "X", values_to = "Z") |>
  mutate(X = as.numeric(X), Y = as.numeric(Y), Z = as.numeric(Z)) |>
  filter(Y%in%c(489:515), X%in%c(656:667)) |>
  pivot_wider(names_from=X,values_from=Z) |>
  select(-Y)
})
 
arr <- array(unlist(res), c(27, 12, length(fn)))
"""

@rget arr

cd(@__DIR__)

jldsave("rainfall_data_2007_08_one_week.jld2", true; large_array=arr)