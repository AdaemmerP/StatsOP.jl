# Tests for all updated cl_ functions (ITP-based root-finding algorithm).
#
# Each test uses small reps and a low target ARL (L0=50) to keep runtime short.
# A fixed seed is used so results are reproducible. The test verifies that
# the returned cl is a finite positive Float64.

# ── Shared parameters ────────────────────────────────────────────────────────
const _CL_TEST_LAM = 0.1
const _CL_TEST_L0 = 370
const _CL_TEST_REPS_FINAL = 1000
const _CL_TEST_REPS_BRACKET = 1000
const _CL_TEST_SEED = 42
const _CL_TEST_VERBOSE = false

# ── cl_sop ───────────────────────────────────────────────────────────────────
@testset "cl_sop" begin
    dgp = ICSTS(11, 11, Normal(0, 1))
    cl = cl_sop(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.02, 1, 1;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end

# ── cl_op I ────────────────────────────────────────────────────────────────────
@testset "cl_op" begin
    dgp = ContinuousDGPIC(Normal(0, 1))
    cl = cl_op(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 1.5;
        chart_choice=Shannon(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
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
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.25;
        chart_choice=Persistence(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
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
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.02, 1, 1;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
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
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.003, 3;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
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
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.001, 3;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
        bracket_step=0.0001,
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
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.01;
        chart_choice=D_Chart(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
    println("cl_gop test passed with cl = ", cl)
end

# ── cl_kappa ─────────────────────────────────────────────────────────────────
@testset "cl_kappa" begin
    dgp = DiscreteDGPIC(Poisson(5), false)
    cl = cl_kappa(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.1;
        chart_choice=KappaN1(),
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
    println("cl_kappa test passed with cl = ", cl)
end

# ── cl_acf ───────────────────────────────────────────────────────────────────
@testset "cl_acf" begin
    dgp = ContinuousDGPIC(Normal(0, 1))
    cl = cl_acf(
        dgp, _CL_TEST_LAM, _CL_TEST_L0, 0.6;
        acf_version=1,
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
        verbose=_CL_TEST_VERBOSE
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
    println("cl_acf test passed with cl = ", cl)
end

# # ── cl_sop_bootstrap ─────────────────────────────────────────────────────────
@testset "cl_sop_bootstrap" begin
    rng = MersenneTwister(1)
    data = randn(rng, 11, 11, 100)
    cl = cl_sop_bootstrap(
        data, _CL_TEST_LAM, _CL_TEST_L0, 0.01, 1, 1;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED,
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
        data, _CL_TEST_LAM, _CL_TEST_L0, 0.02, 3;
        reps_final=_CL_TEST_REPS_FINAL,
        reps_bracket=_CL_TEST_REPS_BRACKET,
        seed=_CL_TEST_SEED
    )
    @test cl isa Float64
    @test cl > 0
    @test isfinite(cl)
end
