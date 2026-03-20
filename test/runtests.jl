using Combinatorics
using StatsOP
using Test
using Random

my_tests = ["test_frequencies.jl",
  "test_op_test_functions.jl",
  "test_cl_functions.jl",
  "test_fill_p_hat.jl",
  "test_stat_sop_bp.jl"]

println("Running tests:")

anyerrors = false
fatalerrors = false
quiet = false

# Approach taken from: https://github.com/JuliaData/DataFrames.jl/blob/main/test/runtests.jl
for my_test in my_tests
  try
    include(my_test)
    println("\t\033[1m\033[32mPASSED\033[0m: $(my_test)")
  catch e
    global anyerrors = true
    println("$(my_test) failed")
    if fatalerrors
      rethrow(e)
    elseif !quiet
      showerror(stdout, e, backtrace())
      println()
    end
  end
end

if anyerrors
  throw("Tests failed")
end