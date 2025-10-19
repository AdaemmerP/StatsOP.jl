# Change to current directory and activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("../../../../.")

using Distributed
# set the number of workers for parallel computing
addprocs(10)

@everywhere include("../../../src/OrdinalPatterns.jl")
@everywhere using .OrdinalPatterns
@everywhere using JLD2
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

reps = 10^3 # Increase the number of replications to 10^6 for reproduction of the results in the paper
L0 = 370
M = 27
N = 12
sp_dgp = ICSTS(M, N, Normal(0, 1))
d1 = 1
d2 = 1
# Rain limits SOP
lam = 0.1
crit_init = map(i -> stat_sop(randn(M, N, 370), lam, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_sop_rain_01 = cl_sop(
    sp_dgp, lam, L0, crit_init, d1, d2, reps; jmin=4, jmax=7, verbose=true
)
arl_sop_rain_01 = arl_sop_ic(sp_dgp, lam, cl_sop_rain_01, d1, d2, reps; chart_choice=3)

lam = 1
crit_init = map(i -> stat_sop(randn(M, N, 370), lam, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_sop_rain_1 = cl_sop(
    sp_dgp, lam, L0, crit_init, d1, d2, reps; jmin=4, jmax=7, verbose=true
)
arl_sop_rain_1 = arl_sop_ic(sp_dgp, lam, cl_sop_rain_1, d1, d2, reps; chart_choice=3)

# Ukraine limits SOP
M = 41
N = 26
lam = 0.1
sp_dgp = ICSTS(M, N, Normal(0, 1))
crit_init = map(i -> stat_sop(randn(M, N, 370), lam, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_sop_ukr_01 = cl_sop(
    sp_dgp, lam, L0, crit_init, d1, d2, reps; jmin=4, jmax=7, verbose=true
)
arl_sop_ukr_01 = arl_sop_ic(sp_dgp, lam, cl_sop_ukr_01, d1, d2, reps; chart_choice=3)

lam = 1
crit_init = map(i -> stat_sop(randn(M, N, 370), lam, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_sop_ukr_1 = cl_sop(
    sp_dgp, lam, L0, crit_init, d1, d2, reps; jmin=4, jmax=7, verbose=true
)
arl_sop_ukr_1 = arl_sop_ic(sp_dgp, lam, cl_sop_ukr_1, d1, d2, reps; chart_choice=3)

# Rain limits BP-SOP
M = 27
N = 12
w = 3
lam = 0.1
sp_dgp = ICSTS(M, N, Normal(0, 1))
crit_init = map(i -> stat_sop_bp(randn(M, N, 370), lam, w) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_bp_rain_01 = cl_sop_bp(
    sp_dgp, lam, L0, crit_init, w, reps; jmin=4, jmax=8, verbose=true
)
arl_bp_rain_01 = arl_sop_bp_ic(sp_dgp, lam, cl_bp_rain_01, w, reps; chart_choice=3)

# Ukraine limits BP-SOP
M = 41
N = 26
w = 3
lam = 0.1
sp_dgp = ICSTS(M, N, Normal(0, 1))
crit_init = map(i -> stat_sop_bp(randn(M, N, 370), lam, w) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_bp_ukr_01 = cl_sop_bp(
    sp_dgp, lam, L0, crit_init, w, reps; jmin=4, jmax=8, verbose=true
)
arl_bp_ukr_01 = arl_sop_bp_ic(sp_dgp, lam, cl_bp_ukr_01, w, reps; chart_choice=3)

lam = 1
crit_init = map(i -> stat_sop_bp(randn(M, N, 370), lam, w) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_bp_ukr_1 = cl_sop_bp(
    sp_dgp, lam, L0, crit_init, w, reps; jmin=4, jmax=8, verbose=true
)
arl_bp_ukr_1 = arl_sop_bp_ic(sp_dgp, lam, cl_bp_ukr_1, w, reps; chart_choice=3)

M = 250
N = 250
sp_dgp = ICSTS(M, N, Normal(0, 1))
d1 = 1
d2 = 1
# Textile limits SOP
lam = 0.1
crit_init = map(i -> stat_sop(randn(M, N, 370), lam, d1, d2) |> last, 1:1_000) |> x -> quantile(x, 0.99)
cl_sop_textile_01 = cl_sop(
    sp_dgp, lam, L0, crit_init, d1, d2, reps; jmin=4, jmax=7, verbose=true
)
arl_sop_textile_01 = arl_sop_ic(sp_dgp, lam, cl_sop_textile_01, d1, d2, reps; chart_choice=3)

# compare the results
# rain:
cl_sop_rain_01
arl_sop_rain_01
cl_sop_rain_1
arl_sop_rain_1
cl_bp_rain_01
arl_bp_rain_01

# ukraine:
cl_sop_ukr_01
arl_sop_ukr_01
cl_sop_ukr_1
arl_sop_ukr_1
cl_bp_ukr_01
arl_bp_ukr_01
cl_bp_ukr_1
arl_bp_ukr_1

# textile:
cl_sop_textile_01
arl_sop_textile_01

# Remove
rmprocs(workers())