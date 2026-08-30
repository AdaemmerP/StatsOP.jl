# Dispatch smoke tests for the arl_ family.
#
# These do not check ARL accuracy — they check that every documented
# (DGP, chart, refinement) combination actually reaches its run-length worker
# and returns a finite result. Several of these combinations used to throw a
# MethodError or a TypeError deep inside a spawned task, because a worker was
# left behind when a signature changed and nothing called it.
#
# Runtime is kept small by `rl_max`: each replication is stopped after at most
# `_ARL_RL_MAX` steps, so a loose control limit cannot make the test hang.
# `reps` must exceed the `Threads.nthreads() * 4` chunk count that the arl_
# functions assert on (see the header of test_cl_functions.jl).

const _ARL_LAM = 0.1
const _ARL_REPS = max(40, 8 * Threads.nthreads())
const _ARL_RL_MAX = 25

# `false` is the classical SOP classification; the three RefinedType instances
# are the refined ones from Weiß and Kim (2025).
const _ARL_REFINEMENTS = (false, RotationType(), DirectionType(), DiagonalType())

const _ARL_SOP_CHARTS = (TauHat(), KappaHat(), TauTilde(), KappaTilde())

# One in-control and one out-of-control spatial DGP per run-length worker.
_arl_icsts() = ICSTS(11, 11, Normal(0, 1))
_arl_sar1() = SAR1((0.4, 0.3, 0.2, 0.1), 11, 11, Normal(0, 1), nothing, 20)
_arl_sar11() = SAR11((0.4, 0.3, 0.2), 11, 11, Normal(0, 1), nothing, 20)
_arl_sqma11() = SQMA11((0.5, 0.3, 0.2), (1, 1, 2), 11, 11, Normal(0, 1), nothing)

# An ARL is a mean run length: finite, and at least 1 because every replication
# takes at least one step.
function _arl_ok(res)
    arl, se = res
    return isfinite(arl) && arl >= 1.0 && isfinite(se) && se >= 0.0
end

# ── SOP: refinement is honoured everywhere ───────────────────────────────────
# Regression: the run-length workers sized `p_hat` with `refinement ? 6 : 3`,
# which throws `TypeError: non-boolean (RotationType) used in boolean context`
# even though the signatures accept `Union{Bool,RefinedType}`. Only the 2D
# `stat_sop` path handled a RefinedType, so the refined charts could not be
# used for any ARL or control-limit computation.
@testset "arl_sop_ic — every refinement" begin
    for rf in _ARL_REFINEMENTS
        res = arl_sop_ic(_arl_icsts(), _ARL_LAM, 0.02, 1, 1, _ARL_REPS;
            chart_choice=TauTilde(), refinement=rf, rl_max=_ARL_RL_MAX)
        @test _arl_ok(res)
    end
end

@testset "arl_sop_oc — every refinement" begin
    for rf in _ARL_REFINEMENTS
        res = arl_sop_oc(_arl_sar11(), _ARL_LAM, 0.02, 1, 1, _ARL_REPS;
            chart_choice=TauTilde(), refinement=rf, rl_max=_ARL_RL_MAX)
        @test _arl_ok(res)
    end
end

@testset "arl_sop_bp_ic — every refinement" begin
    for rf in _ARL_REFINEMENTS
        res = arl_sop_bp_ic(_arl_icsts(), _ARL_LAM, 0.001, 2, _ARL_REPS;
            chart_choice=TauTilde(), refinement=rf, rl_max=_ARL_RL_MAX)
        @test _arl_ok(res)
    end
end

@testset "compute_p_array — every refinement" begin
    rng = MersenneTwister(1)
    data = randn(rng, 11, 11, 5)
    for rf in _ARL_REFINEMENTS
        p_mat = compute_p_array(data, 1, 1; chart_choice=TauTilde(), refinement=rf)
        # 3 type frequencies for the classical classification, 6 for a refined one.
        @test size(p_mat) == (5, rf === false ? 3 : 6)
        @test all(isfinite, p_mat)
    end
end

# ── SOP: the out-of-control Box-Pierce ARL ───────────────────────────────────
# Regression: every `rl_sop_bp_oc` worker called `fill_p_hat!` with 6 arguments
# while all `fill_p_hat!` methods take 7 (`refinement` is the 3rd). The whole
# function therefore threw a MethodError for every DGP and every refinement —
# the only arl_ function with no working path at all.
@testset "arl_sop_bp_oc — runs for every DGP" begin
    for dgp in (_arl_sar1(), _arl_sar11(), _arl_sqma11())
        res = arl_sop_bp_oc(dgp, _ARL_LAM, 0.001, 2, _ARL_REPS;
            chart_choice=TauTilde(), rl_max=_ARL_RL_MAX)
        @test _arl_ok(res)
    end
end

@testset "arl_sop_bp_oc — every refinement" begin
    for rf in _ARL_REFINEMENTS
        res = arl_sop_bp_oc(_arl_sar11(), _ARL_LAM, 0.001, 2, _ARL_REPS;
            chart_choice=TauTilde(), refinement=rf, rl_max=_ARL_RL_MAX)
        @test _arl_ok(res)
    end
end

# ── SOP: the SAR1 out-of-control worker ──────────────────────────────────────
# Regression: `rl_sop_oc(::SAR1, …)` had the same stale 6-argument
# `fill_p_hat!` call. SAR1 is the only DGP that dispatches to it, so the tests
# above (which use SAR11) would not have caught it.
@testset "arl_sop_oc — SAR1 worker" begin
    for cc in _ARL_SOP_CHARTS
        res = arl_sop_oc(_arl_sar1(), _ARL_LAM, 0.02, 1, 1, _ARL_REPS;
            chart_choice=cc, rl_max=_ARL_RL_MAX)
        @test _arl_ok(res)
    end
end

# ── ACF: the method without an explicit acf_version ──────────────────────────
# Regression: `arl_acf_ic(lam, cl, dgp, reps)` spawned
# `rl_acf_ic(lam, cl, i, dgp, dgp.dist; …)`, a 5-argument worker that does not
# exist — only the 6-argument one taking `acf_version` is defined. As the
# docstring states, the short method uses version 1.
@testset "arl_acf_ic — with and without acf_version" begin
    dgp = ContinuousDGPIC(Normal(0, 1))
    res_default = arl_acf_ic(_ARL_LAM, 0.3, dgp, _ARL_REPS; rl_max=_ARL_RL_MAX)
    res_v1 = arl_acf_ic(_ARL_LAM, 0.3, dgp, _ARL_REPS, 1; rl_max=_ARL_RL_MAX)
    @test _arl_ok(res_default)
    @test _arl_ok(res_v1)
end

# ── create_index_sop rejects unsupported refinements ─────────────────────────
# It used to fall off the end of its if/elseif chain and return `nothing` for
# anything other than `false` or a RefinedType, which surfaced much later as a
# confusing MethodError inside a spawned task.
@testset "create_index_sop — argument checking" begin
    @test length(StatsOP.create_index_sop(refinement=false)) == 3
    for rf in (RotationType(), DirectionType(), DiagonalType())
        @test length(StatsOP.create_index_sop(refinement=rf)) == 6
    end
    @test_throws ArgumentError StatsOP.create_index_sop(refinement=nothing)
    @test_throws ArgumentError StatsOP.create_index_sop(refinement=true)
    @test_throws ArgumentError StatsOP.create_index_sop(refinement=1)
end
