@testset "test_sop_bp_bootstrap — structural invariants" begin
  img = randn(MersenneTwister(7), 30, 30)
  Random.seed!(1)
  res = test_sop_bp_bootstrap(img, 1000, 2; chart_choice=TauTilde())

  @test res isa SOPBPTestResultBoot
  # The test uses the buffered stat_sop_bp! internally; it must agree with the
  # public stat_sop_bp.
  @test res.stat ≈ stat_sop_bp(img, 2; chart_choice=TauTilde())
  @test res.stat >= 0.0            # sum of squared chart statistics
  @test 0.0 <= res.boot_pval <= 1.0
  @test res.boot_reject == (res.stat > res.boot_crit)
  @test res.boot_crit > 0
end

@testset "test_sop_bp_bootstrap — spatially dependent image rejects" begin
  # Power against a 3x3 smoothed noise field was verified to be 1.0 over 200 seeds.
  sm = smoothed_image(32, 11)
  Random.seed!(1)
  res = test_sop_bp_bootstrap(sm, 1000, 2; chart_choice=TauTilde())

  @test res.boot_reject
  @test res.boot_pval < 0.05
end

@testset "test_sop_bp_bootstrap — chart choices and w" begin
  img = randn(MersenneTwister(7), 25, 25)
  for chart in (TauTilde(), KappaTilde(), TauHat(), KappaHat())
    Random.seed!(1)
    res = test_sop_bp_bootstrap(img, 300, 1; chart_choice=chart)
    @test res isa SOPBPTestResultBoot
    @test res.stat ≈ stat_sop_bp(img, 1; chart_choice=chart)
    @test isfinite(res.boot_crit)
  end

  # Larger w aggregates over more (d1, d2) combinations, and every term is a square,
  # so the statistic can only grow. Holds for any image.
  Random.seed!(1)
  r1 = test_sop_bp_bootstrap(img, 300, 1; chart_choice=TauTilde())
  Random.seed!(1)
  r3 = test_sop_bp_bootstrap(img, 300, 3; chart_choice=TauTilde())
  @test r3.stat >= r1.stat
end

@testset "test_sop_bp_bootstrap — alpha and block bootstrap" begin
  img = randn(MersenneTwister(7), 25, 25)

  Random.seed!(1)
  r05 = test_sop_bp_bootstrap(img, 1000, 2; chart_choice=TauTilde(), alpha=0.05)
  Random.seed!(1)
  r01 = test_sop_bp_bootstrap(img, 1000, 2; chart_choice=TauTilde(), alpha=0.01)
  # A smaller alpha gives a larger upper-tail critical value — quantile monotonicity,
  # so this holds for any bootstrap draw (both use the same seed and hence the same
  # bootstrap distribution).
  @test r01.boot_crit >= r05.boot_crit

  Random.seed!(1)
  rb = test_sop_bp_bootstrap(img, 300, 2; chart_choice=TauTilde(), block_size=5)
  @test rb isa SOPBPTestResultBoot
  @test isfinite(rb.boot_crit)
  @test rb.boot_crit > 0
end
