@testset "bootstrap_acf" begin
  rng  = MersenneTwister(2026)
  data = randn(rng, 500)

  Random.seed!(1)
  boot = bootstrap_acf(data, 500, 1; block_size=1)

  @test boot isa Vector{Float64}
  @test length(boot) == 500
  @test all(isfinite, boot)
end

@testset "test_acf_bootstrap on iid noise" begin
  rng  = MersenneTwister(2026)
  data = randn(rng, 500)

  Random.seed!(1)
  res = test_acf_bootstrap(data, 2000, 1)

  @test res isa ACFTestResultBoot
  @test res.stat ≈ stat_acf(data, 1)
  @test 0.0 <= res.boot_pval <= 1.0
  @test res.boot_reject == (abs(res.stat) > res.boot_crit)
  @test !res.boot_reject
end

@testset "test_acf_bootstrap on strongly dependent AR(1)" begin
  rng = MersenneTwister(2026)
  n = 500
  x = zeros(n)
  for t in 2:n
    x[t] = 0.9 * x[t-1] + randn(rng)
  end

  Random.seed!(1)
  res = test_acf_bootstrap(x, 2000, 1)

  @test res.boot_reject
  @test res.boot_pval < 0.05
end

@testset "test_acf_bootstrap with block bootstrap (block_size > 1)" begin
  rng = MersenneTwister(2026)
  n = 500
  x = zeros(n)
  for t in 2:n
    x[t] = 0.9 * x[t-1] + randn(rng)
  end

  Random.seed!(1)
  res = test_acf_bootstrap(x, 2000, 1; block_size=10)

  @test res isa ACFTestResultBoot
  @test isfinite(res.boot_crit)
  @test res.boot_crit > 0
end

@testset "test_acf_bootstrap tightens the asymptotic critical value for short series" begin
  # Matches the empirical-size check that motivated this function: for short series the
  # asymptotic N(0, 1/n) critical value is conservative (too wide), so the bootstrap
  # critical value should generally sit below it.
  rng  = MersenneTwister(7)
  data = randn(rng, 15)

  Random.seed!(1)
  res_asymp = test_acf(data, 1)
  res_boot  = test_acf_bootstrap(data, 5000, 1)

  @test res_boot.boot_crit < res_asymp.asymp_crit
end
