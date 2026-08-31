using Distributions

const _BP_CHARTS = (Shannon(), ShannonExtropy(), DistanceToWhiteNoise(), Persistence(),
  UpDownBalance(), RotationalAsymmetry(), UpDownScaling())

# ── test_op_bp: null distributions (deterministic) ───────────────────────────
# The strongest available internal check: wherever a closed-form null exists, feeding the
# tabulated/analytic critical value into that null must return exactly the significance
# level it was constructed for. This ties `_op_bp_null` to `crit_val_op_bp` and would
# catch any wrong scaling factor. No random data involved.
@testset "test_op_bp — null distribution matches crit_val_op_bp" begin
  cases = [
    (Shannon(), 2, 1), (Shannon(), 2, 3), (DistanceToWhiteNoise(), 2, 2),
    (UpDownBalance(), 3, 1), (UpDownBalance(), 3, 4),
    (Shannon(), 3, 1), (ShannonExtropy(), 3, 1), (DistanceToWhiteNoise(), 3, 1),
    (Persistence(), 3, 1), (RotationalAsymmetry(), 3, 1), (UpDownScaling(), 3, 1),
  ]
  for (chart, m, w) in cases
    cv = crit_val_op_bp(; chart_choice=chart, w=w, m=m, alpha=0.05)
    dist, scale = StatsOrdinalPatterns._op_bp_null(chart, m, w)
    @test 1.0 - cdf(dist, scale * cv) ≈ 0.05 atol = 1e-4
  end
end

@testset "test_op_bp — no closed-form null for correlated delays (w > 1)" begin
  # For m = 3 the individual statistics are correlated across delays, so no closed-form
  # null exists beyond w = 1 — except UpDownBalance, which is independent across delays.
  x = randn(MersenneTwister(1), 300)
  for chart in (Shannon(), ShannonExtropy(), DistanceToWhiteNoise(),
                Persistence(), RotationalAsymmetry(), UpDownScaling())
    @test StatsOrdinalPatterns._op_bp_null(chart, 3, 2) === nothing
    @test isnan(test_op_bp(x, 2; chart_choice=chart).asymp_pval)   # holds for any data
  end
  @test StatsOrdinalPatterns._op_bp_null(UpDownBalance(), 3, 2) !== nothing
  @test !isnan(test_op_bp(x, 2; chart_choice=UpDownBalance()).asymp_pval)
end

# ── test_op_bp: structural invariants (hold for every draw) ──────────────────
@testset "test_op_bp — fields agree with stat_op_bp / crit_val_op_bp" begin
  x = randn(MersenneTwister(2026), 500)
  res = test_op_bp(x, 3; chart_choice=Shannon())

  @test res isa OPBPTestResult
  @test res.stat ≈ stat_op_bp(x; chart_choice=Shannon(), m=3, w=3)
  @test res.asymp_crit ≈ crit_val_op_bp(; chart_choice=Shannon(), w=3, m=3, alpha=0.05)
  # The BP statistic is upper-tailed for every chart.
  @test res.asymp_reject == (res.stat > res.asymp_crit)
  @test res.stat >= 0.0
end

@testset "test_op_bp — p-value respects alpha where closed form exists" begin
  # UpDownBalance has a closed-form null for every w, so any alpha is allowed and the
  # reject decision must agree with the p-value. Both branches are exercised: the
  # threshold AR(1) rejects, the iid series does not. This identity holds for any draw.
  for data in (tar1_series(2000, 2026), randn(MersenneTwister(2026), 500))
    for alpha in (0.01, 0.05, 0.10)
      res = test_op_bp(data, 3; chart_choice=UpDownBalance(), alpha=alpha)
      @test !isnan(res.asymp_pval)
      @test res.asymp_reject == (res.asymp_pval < alpha)
    end
  end
end

@testset "test_op_bp — ljung_box weights are applied" begin
  x = randn(MersenneTwister(3), 400)
  bp = test_op_bp(x, 3; chart_choice=Persistence(), ljung_box=false).stat
  bl = test_op_bp(x, 3; chart_choice=Persistence(), ljung_box=true).stat
  @test bp ≈ stat_op_bp(x; chart_choice=Persistence(), m=3, w=3, ljung_box=false)
  @test bl ≈ stat_op_bp(x; chart_choice=Persistence(), m=3, w=3, ljung_box=true)
  @test !(bp ≈ bl)   # the BP and BL weights differ systematically
end

# ── test_op_bp: calibration and power (aggregate, RNG-robust) ────────────────
@testset "test_op_bp — empirical size is close to nominal alpha" begin
  # Aggregate replacement for "one iid draw must not reject", whose failure probability
  # would equal alpha itself. Bounds are ~4 Monte Carlo standard errors wide.
  for chart in _BP_CHARTS
    size_hat = rejection_rate(
      s -> test_op_bp(randn(MersenneTwister(s), 500), 3; chart_choice=chart).asymp_reject,
      SIZE_REPS
    )
    @test SIZE_LOWER < size_hat < SIZE_UPPER
  end
end

@testset "test_op_bp — power against dependent alternatives" begin
  # Charts sensitive to general serial dependence: power verified to be ≥ 0.99.
  for chart in (Shannon(), ShannonExtropy(), DistanceToWhiteNoise(), Persistence())
    power = rejection_rate(
      s -> test_op_bp(ar1_series(500, 0.7, s), 3; chart_choice=chart).asymp_reject, 100
    )
    @test power > 0.9
  end

  # UpDownBalance measures up/down asymmetry: it has essentially no power against a
  # time-reversible Gaussian AR(1) (rejection rate stays at the nominal level), but
  # detects the asymmetric threshold AR(1).
  power_ar = rejection_rate(
    s -> test_op_bp(ar1_series(500, 0.7, s), 3; chart_choice=UpDownBalance()).asymp_reject, 200
  )
  @test power_ar < 0.15

  power_tar = rejection_rate(
    s -> test_op_bp(tar1_series(2000, s), 3; chart_choice=UpDownBalance()).asymp_reject, 100
  )
  @test power_tar > 0.9
end

# ── test_op_bp: error handling (deterministic) ───────────────────────────────
# crit_val_op_bp silently ignores `alpha` for tabulated charts and returns `nothing` for
# unsupported combinations; test_op_bp must turn both into explicit errors.
@testset "test_op_bp — errors instead of silently wrong critical values" begin
  x = randn(MersenneTwister(1), 300)

  # alpha != 0.05 is not available for tabulated charts
  @test_throws ArgumentError test_op_bp(x, 3; chart_choice=Shannon(), alpha=0.01)
  @test_throws ArgumentError test_op_bp(x, 3; chart_choice=Persistence(), alpha=0.10)

  # w outside the tabulated range 1:5
  @test_throws ArgumentError test_op_bp(x, 6; chart_choice=Shannon())

  # combinations without any tabulated/analytic critical value
  @test_throws ArgumentError test_op_bp(x, 2; chart_choice=UpDownBalance(), m=2)
  @test_throws ArgumentError test_op_bp(x, 2; chart_choice=Persistence(), m=2)

  # alpha is free for the closed-form cases
  @test test_op_bp(x, 3; chart_choice=UpDownBalance(), alpha=0.01) isa OPBPTestResult
  @test test_op_bp(x, 3; chart_choice=Shannon(), m=2, alpha=0.01) isa OPBPTestResult
end

# ── test_op_bp_bootstrap ─────────────────────────────────────────────────────
@testset "bootstrap_op_bp" begin
  x = randn(MersenneTwister(2026), 300)
  Random.seed!(1)
  boot = bootstrap_op_bp(x, 500, 2; chart_choice=Shannon())

  @test boot isa Vector{Float64}
  @test length(boot) == 500
  @test all(isfinite, boot)
  @test all(>=(0.0), boot)   # BP statistics are non-negative
end

@testset "test_op_bp_bootstrap — structural invariants" begin
  x = randn(MersenneTwister(2026), 300)
  Random.seed!(1)
  res = test_op_bp_bootstrap(x, 1000, 2; chart_choice=Shannon())

  @test res isa OPBPTestResultBoot
  @test res.stat ≈ stat_op_bp(x; chart_choice=Shannon(), m=3, w=2)
  @test 0.0 <= res.boot_pval <= 1.0
  @test res.boot_reject == (res.stat > res.boot_crit)
  @test res.boot_crit > 0
end

@testset "test_op_bp_bootstrap — rejects a strongly dependent series" begin
  # AR(1) with phi = 0.9: power verified to be 1.0 over several hundred seeds.
  Random.seed!(1)
  res = test_op_bp_bootstrap(ar1_series(500, 0.9, 1), 999, 2; chart_choice=Shannon())
  @test res.boot_reject
  @test res.boot_pval < 0.05
end

@testset "test_op_bp_bootstrap — works where the asymptotic test has no p-value" begin
  # Shannon with w = 3 has no closed-form null, but the bootstrap always returns one.
  x = randn(MersenneTwister(2026), 300)
  @test isnan(test_op_bp(x, 3; chart_choice=Shannon()).asymp_pval)

  Random.seed!(1)
  res = test_op_bp_bootstrap(x, 1000, 3; chart_choice=Shannon(), alpha=0.01)
  @test !isnan(res.boot_pval)
  @test 0.0 <= res.boot_pval <= 1.0
end

@testset "test_op_bp_bootstrap — bootstrap critical value tracks the tabulated one" begin
  # Independent cross-validation of the bootstrap calibration: where theory exists, the
  # resampling critical value must land close to the tabulated asymptotic one. The ratio
  # was measured to stay within [0.976, 1.058] across seeds, so rtol = 0.15 is safe.
  y = randn(MersenneTwister(5), 800)
  Random.seed!(5)
  bc = test_op_bp_bootstrap(y, 4000, 3; chart_choice=Shannon()).boot_crit
  ac = crit_val_op_bp(; chart_choice=Shannon(), w=3, m=3, alpha=0.05)
  @test isapprox(bc, ac; rtol=0.15)
end

@testset "test_op_bp_bootstrap — block bootstrap runs" begin
  Random.seed!(1)
  res = test_op_bp_bootstrap(ar1_series(300, 0.7, 1), 500, 2; chart_choice=Shannon(), block_size=10)
  @test res isa OPBPTestResultBoot
  @test isfinite(res.boot_crit)
  @test res.boot_crit > 0
end
