# Smoke tests for all cl_ functions (ITP-based root-finding algorithm).
#
# These check that each cl_ function runs end to end and returns a finite,
# strictly positive Float64. They are not accuracy tests, so they use the
# cheapest settings that still exercise both phases of the search (bracketing,
# then ITP refinement).
#
# Runtime is proportional to reps * L0 * (number of ARL evaluations). Three
# things keep it small, and all three need care if you change them:
#
# 1. `_CL_TEST_L0 = 50` — a low target ARL. Each replication simulates ~L0
#    steps, so this is the single largest cost factor.
#
# 2. `_CL_TEST_REPS` — few replications. Two constraints:
#      * `reps_final` and `reps_bracket` must stay EQUAL. The two phases
#        evaluate the same noisy ARL function; at low reps a bracket found
#        under one rep count need not be a bracket under another, and
#        `find_zero` then throws "not a bracketing interval".
#      * The floor scales with the thread count. The arl_ functions chunk work
#        as `div(reps, Threads.nthreads() * 4)` and assert `reps > n_chunks`,
#        so `reps` must exceed `4 * nthreads`.
#
# 3. `cl_init` / `bracket_step` — each `cl_init` sits one `bracket_step` BELOW
#    the root for this `L0`, chosen so that the ARL at BOTH ends of the
#    resulting bracket is several standard errors clear of `L0`. Bracketing
#    then finishes in two evaluations, which is fast, and — more importantly —
#    the test does not depend on any particular random draw. An endpoint whose
#    ARL sits close to `L0` can flip sign between the bracketing and the
#    refinement phase, which makes `find_zero` throw and the test fail
#    intermittently; wide margins are what prevent that.
#
# These tests deliberately do NOT pin a seed. The assertions are properties
# that hold for every draw (finite, positive, Float64), not golden values, so
# a fixed seed would only narrow the test to a single realisation — and it
# could not deliver reproducibility anyway: the arl_ functions are not
# reproducible once Julia runs with more than one thread, because the RNG
# streams depend on how chunks get scheduled across tasks. Note that `cl_*`
# still uses common random numbers internally either way: with `seed=nothing`
# it draws one seed (`seed = isnothing(seed) ? rand(Int) : seed`) and reuses it
# across every ARL evaluation of that call, which is what keeps the objective
# smooth for the root finder. The one place a seed is passed below is the test
# that checks the documented `seed` contract itself.
#
# If you change `L0`, a DGP, or a chart here, re-measure the root and re-tune
# the matching `cl_init` / `bracket_step`, or the bracket search will crawl
# (it gives up after 100 steps) and the test may become flaky.

# ── Shared parameters ────────────────────────────────────────────────────────
const _CL_TEST_LAM = 0.1
const _CL_TEST_L0 = 50
const _CL_TEST_REPS = max(40, 8 * Threads.nthreads())
const _CL_TEST_REPS_FINAL = _CL_TEST_REPS
const _CL_TEST_REPS_BRACKET = _CL_TEST_REPS
const _CL_TEST_SEED = 42
const _CL_TEST_VERBOSE = false

# ── cl_sop ───────────────────────────────────────────────────────────────────
@testset "cl_sop" begin
    dgp = ICSTS(11, 11, Normal(0, 1))
    cl = cl_sop(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.018, 1, 1;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.008,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_op I ────────────────────────────────────────────────────────────────────
# Shannon is lower-sided: ARL decreases as cl grows, so the search walks upward.
@testset "cl_op" begin
    dgp = ContinuousDGPIC(Normal(0, 1))
    cl = cl_op(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 1.57;
        chart_choice=Shannon(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.1,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_op II ──────────────────────────────────────────────────────────────────
@testset "cl_op" begin
    dgp = ContinuousDGPIC(Normal(0, 1))
    cl = cl_op(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.14;
        chart_choice=Persistence(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.06,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_sacf ──────────────────────────────────────────────────────────────────
@testset "cl_sacf" begin
    dgp = ICSTS(11, 11, Normal(0, 1))
    cl = cl_sacf(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.028, 1, 1;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.016,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_sacf_bp ───────────────────────────────────────────────────────────────
@testset "cl_sacf_bp" begin
    dgp = ICSTS(11, 11, Normal(0, 1))
    cl = cl_sacf_bp(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.0175, 3;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.01,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_sop_bp ────────────────────────────────────────────────────────────────
@testset "cl_sop_bp" begin
    dgp = ICSTS(27, 12, Normal(0, 1))
    cl = cl_sop_bp(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.0007, 3;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.0004,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_gop ───────────────────────────────────────────────────────────────────
@testset "cl_gop" begin
    dgp = DiscreteDGPIC(Poisson(5), false)
    cl = cl_gop(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.057;
        chart_choice=D_Chart(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.03,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_kappa ─────────────────────────────────────────────────────────────────
@testset "cl_kappa" begin
    dgp = DiscreteDGPIC(Poisson(5), false)
    cl = cl_kappa(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.106;
        chart_choice=KappaN1(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.09,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_acf ───────────────────────────────────────────────────────────────────
@testset "cl_acf" begin
    dgp = ContinuousDGPIC(Normal(0, 1))
    cl = cl_acf(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.253;
        acf_version=1,
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.15,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# # ── cl_sop_bootstrap ─────────────────────────────────────────────────────────
@testset "cl_sop_bootstrap" begin
    rng = MersenneTwister(1)
    data = randn(rng, 11, 11, 100)
    cl = cl_sop_bootstrap(
        data, _CL_TEST_LAM, _CL_TEST_L0, 0.0184, 1, 1;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.01,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_sop_bp_bootstrap ──────────────────────────────────────────────────────
@testset "cl_sop_bp_bootstrap" begin
    rng = MersenneTwister(1)
    data = randn(rng, 6, 6, 100)
    cl = cl_sop_bp_bootstrap(
        data, _CL_TEST_LAM, _CL_TEST_L0, 0.0104, 3;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        bracket_step=0.008
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── L0 may be a Float64 ──────────────────────────────────────────────────────
# Regression: every cl_ function caps the ARL runs at `arl_truncation_factor *
# L0` and hands that value to `arl_*(...; rl_max=…)`, which is typed `::Int`.
# With an integer `L0` the product happens to be an Int; with a Float64 `L0` it
# is a Float64 and the call threw a TypeError. Every test above passes an Int,
# so the whole family was broken for `L0 = 200.0` without any test noticing.
@testset "cl_ accepts a Float64 L0" begin
    common = (reps_final=_CL_TEST_REPS_FINAL, reps_bracket=_CL_TEST_REPS_BRACKET,
        verbose=_CL_TEST_VERBOSE)

    cl_int = cl_sop(ICSTS(11, 11, Normal(0, 1)), _CL_TEST_LAM, 50, 0.018, 1, 1;
        bracket_step=0.008, common...)
    cl_float = cl_sop(ICSTS(11, 11, Normal(0, 1)), _CL_TEST_LAM, 50.0, 0.018, 1, 1;
        bracket_step=0.008, common...)
    # Both must run; they are separate random draws, so only the type and the
    # sign are asserted, not equality.
    @test cl_int isa Float64 && cl_int > 0 && isfinite(cl_int)
    @test cl_float isa Float64 && cl_float > 0 && isfinite(cl_float)

    dgp_c = ContinuousDGPIC(Normal(0, 1))
    @test cl_op(dgp_c, _CL_TEST_LAM, 50.0, 1.57; chart_choice=Shannon(),
        bracket_step=0.1, common...) isa Float64
    @test cl_acf(dgp_c, _CL_TEST_LAM, 50.0, 0.253; acf_version=1,
        bracket_step=0.15, common...) isa Float64
    @test cl_gop(DiscreteDGPIC(Poisson(5), false), _CL_TEST_LAM, 50.0, 0.057;
        chart_choice=D_Chart(), bracket_step=0.03, common...) isa Float64
    @test cl_kappa(DiscreteDGPIC(Poisson(5), false), _CL_TEST_LAM, 50.0, 0.106;
        chart_choice=KappaN1(), bracket_step=0.09, common...) isa Float64
    @test cl_sacf(ICSTS(11, 11, Normal(0, 1)), _CL_TEST_LAM, 50.0, 0.028, 1, 1;
        bracket_step=0.016, common...) isa Float64
    @test cl_sacf_bp(ICSTS(11, 11, Normal(0, 1)), _CL_TEST_LAM, 50.0, 0.0175, 3;
        bracket_step=0.01, common...) isa Float64
    @test cl_sop_bp(ICSTS(27, 12, Normal(0, 1)), _CL_TEST_LAM, 50.0, 0.0007, 3;
        bracket_step=0.0004, common...) isa Float64
end

# ── Seed contract ────────────────────────────────────────────────────────────
# The cl_ docstrings promise that a fixed `seed` makes the result reproducible.
# That only holds single-threaded: with more threads the arl_ functions split
# work into `Threads.nthreads() * 4` chunks whose RNG streams depend on task
# scheduling, so the same seed gives different answers run to run. Assert the
# guarantee where it actually applies.
if Threads.nthreads() == 1
    @testset "cl_ seed is reproducible (single-threaded)" begin
        dgp = ContinuousDGPIC(Normal(0, 1))
        args = (dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.253)
        kwargs = (acf_version=1, reps_final=_CL_TEST_REPS_FINAL,
            reps_bracket=_CL_TEST_REPS_BRACKET, bracket_step=0.15,
            verbose=_CL_TEST_VERBOSE)
        @test cl_acf(args...; seed=_CL_TEST_SEED, kwargs...) ==
              cl_acf(args...; seed=_CL_TEST_SEED, kwargs...)
    end
end
