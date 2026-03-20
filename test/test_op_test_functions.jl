@testset "qup3_value" begin
  @test StatsOP.qup3_op_value(0.01) ≈ 2.2672 atol = 1e-4
  @test StatsOP.qup3_op_value(0.05) ≈ 1.4842 atol = 1e-4
  @test StatsOP.qup3_op_value(0.10) ≈ 1.1626 atol = 1e-4
end
