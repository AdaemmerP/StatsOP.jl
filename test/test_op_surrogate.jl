using TimeseriesSurrogates

@testset "test_op_surrogate" begin
  rng = Xoshiro(1234)

  # Gaussian AR(1): linear process, so the RandomFourier null should hold
  n = 1_000
  alpha_ar = 0.5
  data_ar1 = zeros(n)
  for t in 2:n
    data_ar1[t] = alpha_ar * data_ar1[t-1] + randn(rng)
  end

  res_lin = test_op_surrogate(data_ar1, RandomFourier(), 200;
    chart_choice=DistanceToWhiteNoise(), rng=Xoshiro(42))
  @test res_lin isa OPTestResultSurrogate
  @test res_lin.n_surrogates == 200
  @test 0.0 <= res_lin.surr_pval <= 1.0
  @test !res_lin.surr_reject

  # Logistic map: nonlinear, so the RandomFourier null should be rejected
  data_logistic = zeros(n)
  data_logistic[1] = 0.4
  for t in 2:n
    data_logistic[t] = 4.0 * data_logistic[t-1] * (1.0 - data_logistic[t-1])
  end

  res_nonlin = test_op_surrogate(data_logistic, RandomFourier(), 200;
    chart_choice=DistanceToWhiteNoise(), rng=Xoshiro(42))
  @test res_nonlin.surr_reject
  @test res_nonlin.surr_pval < 0.05

  # Reject decision is consistent with the reported critical value
  @test res_nonlin.surr_reject ==
        StatsOP.reject(DistanceToWhiteNoise(), res_nonlin.stat, res_nonlin.surr_crit)

  # Reproducibility with a fixed rng
  res_rep = test_op_surrogate(data_ar1, RandomFourier(), 200;
    chart_choice=DistanceToWhiteNoise(), rng=Xoshiro(42))
  @test res_rep.surr_pval == res_lin.surr_pval
  @test res_rep.surr_crit == res_lin.surr_crit

  # Other surrogate methods and chart choices go through the same code path
  for method in (AAFT(), RandomShuffle())
    res = test_op_surrogate(data_ar1, method, 100;
      chart_choice=Shannon(), m=3, d=1, rng=Xoshiro(7))
    @test res isa OPTestResultSurrogate
    @test 0.0 <= res.surr_pval <= 1.0
  end
end
