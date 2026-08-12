using Statistics

@testset "crit_val_acf" begin
  @test crit_val_acf(100, 0.05) ≈ quantile(Normal(0, 1), 0.975) / sqrt(100)
  @test crit_val_acf(500, 0.01) ≈ quantile(Normal(0, 1), 0.995) / sqrt(500)
end

@testset "stat_acf (classical)" begin
  rng  = MersenneTwister(1)
  data = randn(rng, 200)
  h    = 3

  # Matches the textbook definition (Box–Jenkins / R `acf()` normalization):
  # numerator and denominator both centered on the full-sample mean, denominator
  # summed over all n observations (not n - h).
  manual = sum((data[1:end-h] .- mean(data)) .* (data[1+h:end] .- mean(data))) /
           sum((data .- mean(data)) .^ 2)

  @test stat_acf(data, h) ≈ manual
end

@testset "test_acf on iid noise" begin
  rng  = MersenneTwister(2026)
  data = randn(rng, 500)
  res  = test_acf(data, 1)

  @test res isa ACFTestResult
  @test res.stat ≈ stat_acf(data, 1)
  @test 0.0 <= res.asymp_pval <= 1.0
  @test res.asymp_reject == (abs(res.stat) > res.asymp_crit)
  @test !res.asymp_reject
end

@testset "test_acf on strongly dependent AR(1)" begin
  rng = MersenneTwister(2026)
  n = 500
  x = zeros(n)
  for t in 2:n
    x[t] = 0.9 * x[t-1] + randn(rng)
  end
  res = test_acf(x, 1)

  @test res.asymp_reject
  @test res.asymp_pval < 0.05
end
