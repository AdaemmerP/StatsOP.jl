
# Packages to use
# Change to current directory
cd(@__DIR__)
using Pkg
Pkg.activate("../.")
using Random
using LinearAlgebra
using Distributed
using OrdinalPatterns
using StatsBase

# using StaticArrays
# using ComplexityMeasures
# using RCall
# using Distributions
# using StatsBase
# using Random

# include("op_dgp_structs.jl")
# include("op_dgp_functions.jl")
# include("op_help_functions.jl")
# include("op_stat_functions.jl")
# include("op_arl_functions.jl")
# include("op_acf_functions.jl")
# include("op_test_functions.jl")
# includet("op_dependence.jl")


# -------------------------------------------------#
# ---------- Replications for Weiss (2002) --------#
# -------------------------------------------------#

# Help function to replicate table 3 in Weiss (2022)
function f_table3(dist, seq_long, eps_long, dist_error, d, xbiv)
  
  OrdinalPatterns.init_dgp_op!(dist, seq_long, eps_long, dist_error, d, xbiv)

  test_1 = OrdinalPatterns.test_op(seq_long, 1; chart_choice=1)[3]
  test_2 = OrdinalPatterns.test_op(seq_long, 1; chart_choice=2)[3]
  test_3 = OrdinalPatterns.test_op(seq_long, 1; chart_choice=3)[3]
  test_4 = OrdinalPatterns.test_op(seq_long, 1; chart_choice=4)[3]
  test_5 = OrdinalPatterns.test_op(seq_long, 1; chart_choice=5)[3]
  test_6 = OrdinalPatterns.test_op(seq_long, 1; chart_choice=6)[3]
  test_7 = OrdinalPatterns.test_op(seq_long, 1; chart_choice=1, op_length=2)[3]

  return (test_1, test_2, test_3, test_4, test_5, test_6, test_7)

end

# Replicates results of Table III in Weiss (2022), p.32
N = 10_000
T = [100, 250, 500, 1000, 1500, 2500]
xbiv = Vector{Float64}(undef, 100)
dist = ICTS(Normal(0, 1))
results = Array{Any}(undef, 6, 8, 5)
results[:, 1, 1:5] .= [100, 250, 500, 1000, 1500, 2500]
eps_long = Float64[]

for d in 1:5
  for (i, n) in enumerate(T)
    seq_long = Vector{Float64}(undef, n) 
    results_all = map(x -> f_table3(dist, seq_long, eps_long, dist.dist, d, xbiv), 1:N)

    results[i, 2, d] = sum(getindex.(results_all, 1)) / N
    results[i, 3, d] = sum(getindex.(results_all, 2)) / N
    results[i, 4, d] = sum(getindex.(results_all, 3)) / N
    results[i, 5, d] = sum(getindex.(results_all, 4)) / N
    results[i, 6, d] = sum(getindex.(results_all, 5)) / N
    results[i, 7, d] = sum(getindex.(results_all, 6)) / N
    results[i, 8, d] = sum(getindex.(results_all, 7)) / N
  end
  println(d)
end
vcat(results[:, :, 1], results[:, :, 2], results[:, :, 3], results[:, :, 4], results[:, :, 5])


# Help function to replicate table IV in Weiss (2022)
function f_table4(dist, seq_long, eps_long, dist_error, d, xbiv)

  x_output = OrdinalPatterns.init_dgp_op!(dist, seq_long, eps_long, dist_error, d, xbiv)

  test_1 = test_op(x_output, 1; chart_choice=1)[3]
  test_2 = test_op(x_output, 1; chart_choice=2)[3]
  test_3 = test_op(x_output, 1; chart_choice=3)[3]
  test_4 = test_op(x_output, 1; chart_choice=4)[3]
  test_5 = test_op(x_output, 1; chart_choice=5)[3]
  test_6 = test_op(x_output, 1; chart_choice=6)[3]
  test_7 = abs(StatsBase.autocor(x_output, [1], demean=true)[1]) > 1.959964 * sqrt(1 / length(seq_long))


  return (test_1, test_2, test_3, test_4, test_5, test_6, test_7)

end

# Help function to replicate MA processes in table IV in Weiss (2022)
function f_table4_ma(dist, x_long, eps_long, dist_error, d, xbiv)

  x_output = OrdinalPatterns.init_dgp_op!(dist, x_long, eps_long, dist_error, d, xbiv)

  test_1 = test_op(x_output, 1; chart_choice=1)[3]
  test_2 = test_op(x_output, 1; chart_choice=2)[3]
  test_3 = test_op(x_output, 1; chart_choice=3)[3]
  test_4 = test_op(x_output, 1; chart_choice=4)[3]
  test_5 = test_op(x_output, 1; chart_choice=5)[3]
  test_6 = test_op(x_output, 1; chart_choice=6)[3]
  test_7 = abs(StatsBase.autocor(x_long, [d], demean=true)[1]) > 1.959964 * sqrt(1 / length(x_long))


  return (test_1, test_2, test_3, test_4, test_5, test_6, test_7)

end

# Replicates results for DGP 1 in Table IV in Weiss (2022), p.32
N = 10_000
dist = AR1(0.5, Normal(0, 1))
xbiv = Vector{Float64}(undef, 100)
T = [100, 250, 500, 1000, 1500, 2500]
results = Array{Any}(undef, length(T), 8)
results[:, 1] = T
d = 1

for (i, n) in enumerate(T)

  seq_long = Vector{Float64}(undef, n)
  results_all = map(x -> f_table4(dist, seq_long, eps_long, dist.dist, d, xbiv), 1:N)

  results[i, 2] = sum(getindex.(results_all, 1)) / N
  results[i, 3] = sum(getindex.(results_all, 2)) / N
  results[i, 4] = sum(getindex.(results_all, 3)) / N
  results[i, 5] = sum(getindex.(results_all, 4)) / N
  results[i, 6] = sum(getindex.(results_all, 5)) / N
  results[i, 7] = sum(getindex.(results_all, 6)) / N
  results[i, 8] = sum(getindex.(results_all, 7)) / N
end
results

# Replicates results for DGP 2 in Table IV in Weiss (2022), p.32
N = 10_000
dist = MA1(0.8, Normal(0, 1))
T = [100, 250, 500, 1000, 1500, 2500]
results = Array{Any}(undef, length(T), 8)
results[:, 1] = T
d = 1

for (i, n) in enumerate(T)

  x_long = Vector{Float64}(undef, n + 1)
  eps_long = Vector{Float64}(undef, n + 1)
  results_all = map(x -> f_table4_ma(dist, x_long, eps_long, dist.dist, d, xbiv), 1:N)

  results[i, 2] = sum(getindex.(results_all, 1)) / N
  results[i, 3] = sum(getindex.(results_all, 2)) / N
  results[i, 4] = sum(getindex.(results_all, 3)) / N
  results[i, 5] = sum(getindex.(results_all, 4)) / N
  results[i, 6] = sum(getindex.(results_all, 5)) / N
  results[i, 7] = sum(getindex.(results_all, 6)) / N
  results[i, 8] = sum(getindex.(results_all, 7)) / N
  println(i)
end
results

# Replicates results for DGP 3 with d = 1 in Table IV in Weiss (2022), p.32
N = 10_000
dist = MA2(0.0, 0.8, Normal(0, 1))
T = [100, 250, 500, 1000, 1500, 2500]
results = Array{Any}(undef, length(T), 8)
results[:, 1] = T
d = 1

for (i, n) in enumerate(T)

  x_long = Vector{Float64}(undef, n + 2)
  eps_long = Vector{Float64}(undef, n + 2)
  results_all = map(x -> f_table4_ma(dist, x_long, eps_long, dist.dist, d, xbiv), 1:N)

  results[i, 2] = sum(getindex.(results_all, 1)) / N
  results[i, 3] = sum(getindex.(results_all, 2)) / N
  results[i, 4] = sum(getindex.(results_all, 3)) / N
  results[i, 5] = sum(getindex.(results_all, 4)) / N
  results[i, 6] = sum(getindex.(results_all, 5)) / N
  results[i, 7] = sum(getindex.(results_all, 6)) / N
  results[i, 8] = sum(getindex.(results_all, 7)) / N
  println(i)
end
results

# Replicates results for DGP 3 with d=2 in Table IV in Weiss (2022), p.32
N = 10_000
dist = MA2(0.0, 0.8, Normal(0, 1))
T = [100, 250, 500, 1000, 1500, 2500]
results = Array{Any}(undef, length(T), 8)
results[:, 1] = T
d = 2

for (i, n) in enumerate(T)

  x_long = Vector{Float64}(undef, n*d + 2)
  eps_long = Vector{Float64}(undef, n*d + 2)
  results_all = map(x -> f_table4_ma(dist, x_long, eps_long, dist.dist, d, xbiv), 1:N)

  results[i, 2] = sum(getindex.(results_all, 1)) / N
  results[i, 3] = sum(getindex.(results_all, 2)) / N
  results[i, 4] = sum(getindex.(results_all, 3)) / N
  results[i, 5] = sum(getindex.(results_all, 4)) / N
  results[i, 6] = sum(getindex.(results_all, 5)) / N
  results[i, 7] = sum(getindex.(results_all, 6)) / N
  results[i, 8] = sum(getindex.(results_all, 7)) / N
  println(i)
end
results


# Replicates results for DGP 1 in Table IV in Weiss (2022), p.32
N = 10_000
dist = AR1(0.5, Normal(0, 1))
xbiv = Vector{Float64}(undef, 100)
T = [100, 250, 500, 1000, 1500, 2500]
results = Array{Any}(undef, length(T), 8)
results[:, 1] = T
for (i, n) in enumerate(T)
  seq_long = Vector{Float64}(undef, n)
  init_dgp_op!(dist, seq_long, dist.dist, 1, xbiv)
  results_all = map(x -> f_table4(dist, seq_long, dist.dist, d, xbiv), 1:N)

  results[i, 2] = sum(getindex.(results_all, 1)) / N
  results[i, 3] = sum(getindex.(results_all, 2)) / N
  results[i, 4] = sum(getindex.(results_all, 3)) / N
  results[i, 5] = sum(getindex.(results_all, 4)) / N
  results[i, 6] = sum(getindex.(results_all, 5)) / N
  results[i, 7] = sum(getindex.(results_all, 6)) / N
  results[i, 8] = sum(getindex.(results_all, 7)) / N
  println(i)
end
results

# Replicates results for DGP 4 in Table IV in Weiss (2022), p.32
dist = TEAR1(0.15, Exponential(1))
for (i, n) in enumerate(T)
  seq_long = Vector{Float64}(undef, n)
  init_dgp_op!(dist, seq_long, dist.dist, 1, xbiv)
  results_all = map(x -> f_table4(dist, seq_long, dist.dist, d, xbiv), 1:N)

  results[i, 2] = sum(getindex.(results_all, 1)) / N
  results[i, 3] = sum(getindex.(results_all, 2)) / N
  results[i, 4] = sum(getindex.(results_all, 3)) / N
  results[i, 5] = sum(getindex.(results_all, 4)) / N
  results[i, 6] = sum(getindex.(results_all, 5)) / N
  results[i, 7] = sum(getindex.(results_all, 6)) / N
  results[i, 8] = sum(getindex.(results_all, 7)) / N
  println(i)
end
results