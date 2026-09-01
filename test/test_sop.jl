# Tests for the asymptotic test based on spatial ordinal patterns:
# `stat_sop`, `crit_val_sop` and `test_sop`.
#
# These follow the two forms described in test_helpers.jl: structural invariants that
# hold for every draw, and aggregate rejection rates compared against generous bounds.
# The size checks are the substantive correctness test here — a mistranscribed variance
# constant in `crit_val_sop` leaves every structural invariant intact but pushes the
# empirical rejection rate away from the nominal level.
#
# RUNTIME. For the entropy charts a single `test_sop` call costs ~150 ms, essentially all
# of it the p-value: the generalized chi-squared cdf is evaluated by Davies' method, a
# numerical integration, and unlike the critical value it depends on the data and so
# cannot be cached. Full `test_sop` calls are therefore made only where the p-value is
# what is being asserted, and kept to a few draws. The aggregate size and power checks —
# which need hundreds of replications — compare `stat_sop` against `crit_val_sop`
# directly instead: that is the same reject decision (asserted below), reuses the cached
# critical value, and costs ~0.2 ms a draw rather than ~150 ms.

const _SOP_TK_CHARTS = (TauHat(), KappaHat(), TauTilde(), KappaTilde())
const _SOP_ENT_CHARTS = (Shannon(), ShannonExtropy(), DistanceToWhiteNoise())
const _SOP_REFINEMENTS = (RotationType(), DirectionType(), DiagonalType())

_sop_image(seed, size_out=30) = randn(MersenneTwister(seed), size_out, size_out)

# ── stat_sop: the frequency vector is a distribution over SOP types ───────────
# The entropy charts need all three type frequencies, so for them `p_hat` is a complete
# distribution. Each tau/kappa statistic depends on only a subset of the types, and
# `stat_sop` fills just those entries of the length-3 buffer and leaves the rest at
# zero — so for those charts the invariant is not that `p_hat` sums to one, but that the
# entries it does fill agree with the full distribution computed for the same image.
@testset "stat_sop — type frequencies" begin
  data = _sop_image(1)
  full = stat_sop(data, 1, 1; chart_choice=Shannon())[2]
  @test length(full) == 3
  @test all(>=(0), full)
  @test sum(full) ≈ 1

  for cc in _SOP_ENT_CHARTS
    stat, p_hat = stat_sop(data, 1, 1; chart_choice=cc)
    @test isfinite(stat)
    @test p_hat ≈ full
  end

  for cc in _SOP_TK_CHARTS
    stat, p_hat = stat_sop(data, 1, 1; chart_choice=cc)
    @test isfinite(stat)
    @test length(p_hat) == 3
    # Every entry is either left at zero or equal to the corresponding full frequency:
    # catches a statistic that fills the wrong slot of the buffer.
    @test all(p_hat[i] == 0 || p_hat[i] ≈ full[i] for i in 1:3)
    @test any(!=(0), p_hat)
  end

  # The refined classifications split the three classical types into six.
  for rf in _SOP_REFINEMENTS
    stat, p_hat = stat_sop(data, 1, 1; chart_choice=Shannon(), refinement=rf)
    @test isfinite(stat)
    @test length(p_hat) == 6
    @test all(>=(0), p_hat)
    @test sum(p_hat) ≈ 1
  end
end

# ── crit_val_sop: behaviour in alpha and in the image size ───────────────────
@testset "crit_val_sop — monotone in alpha and in sample size" begin
  for cc in (_SOP_TK_CHARTS..., _SOP_ENT_CHARTS...)
    cv05 = crit_val_sop(30, 30, 1, 1; chart_choice=cc, alpha=0.05)
    cv01 = crit_val_sop(30, 30, 1, 1; chart_choice=cc, alpha=0.01)
    @test cv05 > 0
    # A smaller significance level must not be easier to reject at.
    @test cv01 > cv05
    # Every chart is consistent: the rejection region shrinks as the image grows.
    @test crit_val_sop(60, 60, 1, 1; chart_choice=cc, alpha=0.05) < cv05
  end
end

# The kappa statistics have the larger asymptotic variance (2/3 against 2/9), so at a
# common alpha and image size their critical value must be the larger one.
@testset "crit_val_sop — kappa has the wider critical region" begin
  @test crit_val_sop(30, 30, 1, 1; chart_choice=KappaHat()) >
        crit_val_sop(30, 30, 1, 1; chart_choice=TauHat())
  @test crit_val_sop(30, 30, 1, 1; chart_choice=KappaTilde()) >
        crit_val_sop(30, 30, 1, 1; chart_choice=TauTilde())
end

@testset "crit_val_sop — alpha is a keyword, not positional" begin
  # Guards the argument-order convention: a positional third argument is the row
  # delay `d1`, never the significance level.
  @test crit_val_sop(30, 30, 1, 1; chart_choice=TauHat()) ==
        crit_val_sop(30, 30, 1, 1; chart_choice=TauHat(), alpha=0.05)
end

# ── test_sop: the reported fields are mutually consistent ────────────────────
# `asymp_reject` must be exactly the comparison of the statistic against the critical
# value, in the direction belonging to the chart, and the p-value must cross `alpha` at
# the same point. A test whose p-value and reject decision can disagree is the failure
# mode these assertions exist for.
@testset "test_sop — fields are mutually consistent" begin
  for seed in 1:3
    data = _sop_image(seed)
    for cc in _SOP_TK_CHARTS
      res = test_sop(data, 1, 1; chart_choice=cc)
      @test res.chart === cc
      @test isfinite(res.stat)
      @test 0 <= res.asymp_pval <= 1
      @test res.asymp_reject == (abs(res.stat) > res.asymp_crit)   # two-sided
      @test res.asymp_reject == (res.asymp_pval < 0.05)
    end
    for cc in _SOP_ENT_CHARTS
      res = test_sop(data, 1, 1; chart_choice=cc)
      @test 0 <= res.asymp_pval <= 1
      @test res.asymp_reject == (res.stat > res.asymp_crit)        # upper-tail
      @test res.asymp_reject == (res.asymp_pval < 0.05)
    end
  end
end

@testset "test_sop — alpha is honoured" begin
  data = _sop_image(3)
  for cc in (_SOP_TK_CHARTS..., _SOP_ENT_CHARTS...)
    res05 = test_sop(data, 1, 1; chart_choice=cc, alpha=0.05)
    res01 = test_sop(data, 1, 1; chart_choice=cc, alpha=0.01)
    # Only the critical value depends on alpha; the statistic does not.
    @test res05.stat ≈ res01.stat
    @test res01.asymp_crit > res05.asymp_crit
    # The p-value is a property of the data, not of the level it is compared against.
    @test res05.asymp_pval ≈ res01.asymp_pval
    @test res01.asymp_reject == (res01.asymp_pval < 0.01)
  end
end

@testset "test_sop — matches crit_val_sop and stat_sop" begin
  data = _sop_image(4)
  for cc in _SOP_TK_CHARTS, rf in (false,)
    res = test_sop(data, 1, 1; chart_choice=cc, refinement=rf, alpha=0.02)
    @test res.asymp_crit ≈ crit_val_sop(size(data, 1), size(data, 2), 1, 1;
      chart_choice=cc, refinement=rf, alpha=0.02)
    @test res.stat ≈ stat_sop(data, 1, 1; chart_choice=cc, refinement=rf)[1]
  end
end

@testset "test_sop — refined classification for the entropy charts" begin
  data = _sop_image(5)
  # One chart across all three refinements, and all three charts on one refinement:
  # enough to reach every (chart, refinement) code path without paying for the full
  # 3x3 grid of ~150 ms p-values.
  for rf in _SOP_REFINEMENTS
    res = test_sop(data, 1, 1; chart_choice=Shannon(), refinement=rf)
    @test isfinite(res.stat)
    @test 0 <= res.asymp_pval <= 1
    @test res.asymp_reject == (res.stat > res.asymp_crit)
  end
  for cc in _SOP_ENT_CHARTS
    res = test_sop(data, 1, 1; chart_choice=cc, refinement=RotationType())
    @test isfinite(res.stat)
    @test res.asymp_reject == (res.stat > res.asymp_crit)
  end
end

# The refined classifications have asymptotic theory only for the entropy charts, so
# tau/kappa plus a RefinedType has to be reported as unsupported rather than reaching a
# worker that silently uses the classical critical value.
@testset "test_sop — tau/kappa reject a refined classification" begin
  data = _sop_image(6)
  for cc in _SOP_TK_CHARTS, rf in _SOP_REFINEMENTS
    @test_throws ArgumentError test_sop(data, 1, 1; chart_choice=cc, refinement=rf)
  end
end

# ── Size and power ───────────────────────────────────────────────────────────
# Under H₀ the image is spatial white noise and the rejection rate must sit near the
# nominal level; against a spatially dependent image the test must have power.
#
# `_reject_sop` reproduces the reject decision of `test_sop` without its p-value: the
# statistic against the cached critical value, two-sided for tau/kappa and upper-tail for
# the entropy charts after the same `rescale_sop` transformation `test_sop` applies. The
# testset below pins it to `test_sop` so the two cannot drift apart.
function _reject_sop(data, cc; d1=1, d2=1, alpha=0.05)
  crit = crit_val_sop(size(data, 1), size(data, 2), d1, d2; chart_choice=cc, alpha=alpha)
  raw = stat_sop(data, d1, d2; chart_choice=cc)
  if cc isa Union{TauHat,KappaHat,TauTilde,KappaTilde}
    return abs(raw[1]) > crit
  else
    return StatsOrdinalPatterns.rescale_sop(raw[1], length(raw[2]), cc) > crit
  end
end

@testset "_reject_sop — agrees with test_sop" begin
  for seed in 1:3, cc in (_SOP_TK_CHARTS..., _SOP_ENT_CHARTS...)
    data = _sop_image(seed, 25)
    @test _reject_sop(data, cc) == test_sop(data, 1, 1; chart_choice=cc).asymp_reject
  end
end

@testset "test_sop — empirical size is close to nominal alpha" begin
  for cc in (_SOP_TK_CHARTS..., _SOP_ENT_CHARTS...)
    rate = rejection_rate(SIZE_REPS) do s
      _reject_sop(_sop_image(s, 25), cc)
    end
    @test SIZE_LOWER <= rate <= SIZE_UPPER
  end
end

@testset "test_sop — power against a spatially dependent image" begin
  # 3x3 moving-average smoothing induces strong positive spatial dependence; every
  # chart should reject it essentially always.
  for cc in (_SOP_TK_CHARTS..., _SOP_ENT_CHARTS...)
    rate = rejection_rate(50) do s
      _reject_sop(smoothed_image(25, s), cc)
    end
    @test rate > 0.9
  end
end

# ── The bootstrap test agrees with the asymptotic one ────────────────────────
# The bootstrap critical value is computed on the same scale as the asymptotic one
# (see the note in test_sop_bootstrap), so on a clearly dependent image the two must
# reach the same conclusion.
@testset "test_sop / test_sop_bootstrap — same conclusion on a clear alternative" begin
  data = smoothed_image(25, 11)
  for cc in _SOP_TK_CHARTS
    asymp = test_sop(data, 1, 1; chart_choice=cc)
    boot = test_sop_bootstrap(data, 500, 1, 1; chart_choice=cc)
    @test asymp.stat ≈ boot.stat
    @test asymp.asymp_reject == boot.boot_reject == true
  end
end
