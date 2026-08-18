using Statistics
using Distributions

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
  # summed over all n observations (not n - h). Holds for any data.
  manual = sum((data[1:end-h] .- mean(data)) .* (data[1+h:end] .- mean(data))) /
           sum((data .- mean(data)) .^ 2)

  @test stat_acf(data, h) ≈ manual
end

@testset "test_acf — structural invariants" begin
  data = randn(MersenneTwister(2026), 500)
  res  = test_acf(data, 1)

  @test res isa ACFTestResult
  @test res.stat ≈ stat_acf(data, 1)
  @test 0.0 <= res.asymp_pval <= 1.0
  @test res.asymp_reject == (abs(res.stat) > res.asymp_crit)
  @test res.asymp_crit ≈ crit_val_acf(500, 0.05)
end

@testset "test_acf — empirical size is close to nominal alpha" begin
  # Aggregate replacement for "one iid draw must not reject", whose failure probability
  # would equal alpha itself.
  size_hat = rejection_rate(
    s -> test_acf(randn(MersenneTwister(s), 500), 1).asymp_reject, SIZE_REPS
  )
  @test SIZE_LOWER < size_hat < SIZE_UPPER
end

@testset "test_acf — power against a strongly dependent AR(1)" begin
  # phi = 0.9 at n = 500: power verified to be 1.0 over several hundred seeds.
  res = test_acf(ar1_series(500, 0.9, 2026), 1)
  @test res.asymp_reject
  @test res.asymp_pval < 0.05

  power = rejection_rate(s -> test_acf(ar1_series(500, 0.9, s), 1).asymp_reject, 100)
  @test power > 0.95
end
