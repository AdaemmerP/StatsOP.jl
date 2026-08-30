# Tests for the ordinal-pattern dependence and changepoint functions in
# src/op/op_dependence.jl (Schnurr and Dehling, 2017).
#
# Assertions come in the two forms used throughout this suite (see test_helpers.jl):
#
#   * Exact identities that hold for EVERY draw. The dependence coefficient is
#     built from pattern frequencies, so it is exactly +1 when the two series
#     carry the same pattern in every window, exactly -1 when one is the
#     sign-flipped other, and completely unchanged by any strictly increasing
#     transformation of either series — an ordinal pattern only sees the ranks
#     inside its window, not the values.
#
#   * Aggregate rejection rates over many seeds for `changepoint_op`, whose
#     reject/no-reject decision on a single series is itself random.
#
# Both functions cost about 0.05 ms per call at n = 300, so the aggregate checks
# can afford enough replications to be insensitive to any particular seed.

const _DEP_N = 300
const _DEP_SIZE_REPS = 2_000   # H0 checks; MC standard error ≈ 0.004
const _DEP_POWER_REPS = 200

# Two independent standard normal series.
_indep_pair(seed) = (r = MersenneTwister(seed); (randn(r, _DEP_N), randn(r, _DEP_N)))

# Dependent, but with the SAME dependence throughout: the null hypothesis of
# `changepoint_op` is "the dependence does not change", not "there is none".
function _constant_dependence_pair(seed)
    r = MersenneTwister(seed)
    x = randn(r, _DEP_N)
    return (x, x .+ 0.5 .* randn(r, _DEP_N))
end

# `y` is independent of `x` up to index `at` and equal to it afterwards, i.e. a
# genuine break in the dependence structure at `at`.
function _break_pair(seed, at)
    r = MersenneTwister(seed)
    x = randn(r, _DEP_N)
    z = randn(r, _DEP_N)
    return (x, vcat(z[1:at], x[(at+1):end]))
end

# ── count_uv_op ──────────────────────────────────────────────────────────────
@testset "count_uv_op — every window is counted exactly once" begin
    x = randn(MersenneTwister(1), 200)
    rel, counts = count_uv_op(x)
    n_pat = 200 - 2  # length(ts) - (m - 1) * d, with m = 3 and d = 1

    @test length(counts) == factorial(3)
    @test sum(counts) == n_pat
    # The relative frequencies come back wrapped in a one-element vector.
    @test length(rel) == 1
    @test rel[1] ≈ counts ./ n_pat
    @test sum(rel[1]) ≈ 1.0

    # A larger delay widens each window, so fewer of them fit in the series.
    @test sum(count_uv_op(x; d=2)[2]) == 200 - 2 * 2
    @test length(count_uv_op(x; m=4)[2]) == factorial(4)
end

@testset "count_uv_op — a monotone series yields a single pattern" begin
    # Every window of an increasing series has sortperm [1,2,3], which is Lehmer
    # index 1; a decreasing one gives [3,2,1], the last of the 3! = 6 patterns.
    inc = collect(1.0:20.0)
    @test count_uv_op(inc)[2] == [18, 0, 0, 0, 0, 0]
    @test count_uv_op(reverse(inc))[2] == [0, 0, 0, 0, 0, 18]
end

# ── count_mv_op ──────────────────────────────────────────────────────────────
@testset "count_mv_op — consistent with count_uv_op" begin
    x, y = _indep_pair(1)
    count_x, count_y, count_yrev, count_eq, _, seq_x, seq_y = count_mv_op(x, y)
    n_pat = _DEP_N - 2

    # The per-series counts must be exactly what the univariate function reports.
    @test count_x == count_uv_op(x)[2]
    @test count_y == count_uv_op(y)[2]
    @test sum(count_x) == n_pat
    @test sum(count_y) == n_pat
    @test sum(count_yrev) == n_pat

    @test length(seq_x) == n_pat
    @test length(seq_y) == n_pat
    @test all(1 .<= seq_x .<= factorial(3))
    @test seq_x != seq_y  # the two series are independent

    # A series matches itself in every single window.
    count_eq_self = count_mv_op(x, x)[4]
    @test sum(count_eq_self) == n_pat
    @test count_mv_op(x, x)[6] == count_mv_op(x, x)[7]
    @test count_eq != count_eq_self
end

@testset "count_mv_op — argument checking" begin
    x = randn(MersenneTwister(1), 50)
    @test_throws AssertionError count_mv_op(x, randn(40))
    @test_throws AssertionError count_mv_op(x, x; m=1)
    @test_throws AssertionError count_mv_op(x, x; m=5)
end

# ── dependence_op ────────────────────────────────────────────────────────────
@testset "dependence_op — exact at both extremes" begin
    x, _ = _indep_pair(2)
    # Identical patterns everywhere, and perfectly reversed patterns everywhere.
    @test dependence_op(x, x)[1] == 1.0
    @test dependence_op(x, -x)[1] == -1.0
end

@testset "dependence_op — invariant under increasing transformations" begin
    # Ordinal patterns depend only on the ranks within a window, so a strictly
    # increasing transformation of either series cannot change the coefficient.
    # A dependent pair is used so the invariant value is far from zero (≈ 0.49)
    # and the check is not trivially satisfied.
    x, y = _constant_dependence_pair(3)
    base = dependence_op(x, y)[1]
    @test base > 0.4

    @test dependence_op(3 .* x .+ 7, y)[1] == base
    @test dependence_op(x, exp.(y))[1] == base
    @test dependence_op(exp.(x), 2 .* y .- 1)[1] == base
end

@testset "dependence_op — bounded, and centred at zero under independence" begin
    vals = [dependence_op(_indep_pair(s)...)[1] for s in 1:200]
    @test all(-1.0 .<= vals .<= 1.0)
    @test abs(mean(vals)) < 0.02        # centred on zero ...
    @test maximum(abs, vals) < 0.25     # ... with no wild single draw
end

@testset "dependence_op — pattern lengths 2, 3 and 4" begin
    x, y = _indep_pair(4)
    for m in 2:4
        @test dependence_op(x, x; m=m)[1] == 1.0
        @test dependence_op(x, -x; m=m)[1] == -1.0
        @test -1.0 <= dependence_op(x, y; m=m)[1] <= 1.0
    end
    @test dependence_op(x, x; d=2)[1] == 1.0
    @test_throws AssertionError dependence_op(x, randn(_DEP_N - 1))
end

# ── changepoint_op ───────────────────────────────────────────────────────────
@testset "changepoint_op — return structure" begin
    x, y = _indep_pair(5)
    Tnmax, changepoint, p_value, conf_iv = changepoint_op(x, y)
    n_pat = _DEP_N - 2

    @test Tnmax >= 0.0
    @test changepoint isa Int
    @test 1 <= changepoint <= n_pat
    @test 0.0 <= p_value <= 1.0
    @test conf_iv == (-1, 1) .* quantile(Kolmogorov(), 0.95)

    # `conf_level` only widens the reported interval; it must not touch the
    # statistic, the estimated breakpoint, or the p-value.
    r99 = changepoint_op(x, y; conf_level=0.99)
    @test r99[1] == Tnmax
    @test r99[2] == changepoint
    @test r99[3] == p_value
    @test r99[4][2] > conf_iv[2]
end

@testset "changepoint_op — unweighted variant runs" begin
    x, y = _break_pair(1, 150)
    Tnmax, changepoint, p_value, _ = changepoint_op(x, y; weight=false)
    @test Tnmax >= 0.0
    @test changepoint isa Int
    @test 0.0 <= p_value <= 1.0
end

@testset "changepoint_op — does not over-reject when dependence is constant" begin
    # H0 is "the dependence does not CHANGE", so it must hold both when the two
    # series are independent throughout and when they are dependent throughout.
    # The procedure is conservative in both cases (≈ 3% at a nominal 5%), so the
    # binding assertion is the upper one. The power testset below is what rules
    # out the degenerate alternative of a test that simply never rejects.
    size_indep = rejection_rate(
        s -> changepoint_op(_indep_pair(s)...)[3] < 0.05, _DEP_SIZE_REPS
    )
    size_dep = rejection_rate(
        s -> changepoint_op(_constant_dependence_pair(s)...)[3] < 0.05, _DEP_SIZE_REPS
    )
    @test 0.005 < size_indep < SIZE_UPPER
    @test 0.005 < size_dep < SIZE_UPPER
end

@testset "changepoint_op — detects a genuine break and locates it" begin
    at = 150  # halfway through the series
    power = rejection_rate(
        s -> changepoint_op(_break_pair(s, at)...)[3] < 0.05, _DEP_POWER_REPS
    )
    @test power > 0.9

    errors = [abs(changepoint_op(_break_pair(s, at)...)[2] - at) for s in 1:_DEP_POWER_REPS]
    @test median(errors) <= 5
    @test quantile(errors, 0.9) <= 25
end
