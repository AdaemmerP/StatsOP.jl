
# Tests for stat_sop_bp (both overloads).
#
# Verifications:
#   1. refinement is a SOPClassification with default OrdinaryType() (not Nothing).
#   2. 2D overload: with w=1 there is exactly one (d1,d2)=(1,1) combination,
#      so bp_stat == stat_sop(data, 1, 1; chart_choice=X)[1]^2.
#   3. 3D EWMA overload: returns a vector of length size(data,3), all non-negative.

const _DATA_BP = Float64[9 1 2;
                         6 7 6;
                         9 4 9]

# -----------------------------------------------------------------------
# Signature verification: refinement is a SOPClassification (not Nothing)
# -----------------------------------------------------------------------
@testset "stat_sop_bp — refinement::SOPClassification" begin

    # Calling with the default (OrdinaryType()) must work without error
    @test_nowarn stat_sop_bp(_DATA_BP, 1; chart_choice=TauTilde())

    # Calling with refinement=OrdinaryType() explicitly must work
    @test_nowarn stat_sop_bp(_DATA_BP, 1; chart_choice=TauTilde(), refinement=OrdinaryType())

    # Calling with a RefinedType must work
    @test_nowarn stat_sop_bp(_DATA_BP, 1; chart_choice=TauTilde(), refinement=RotationType())

    # Passing nothing must now throw a TypeError (old signature accepted Nothing;
    # Julia raises TypeError at the keyword argument boundary, not MethodError)
    @test_throws TypeError stat_sop_bp(_DATA_BP, 1; chart_choice=TauTilde(), refinement=nothing)

end

# -----------------------------------------------------------------------
# 2D overload: bp_stat == stat_sop(data, 1, 1; chart_choice)[1]^2  for w=1
# -----------------------------------------------------------------------
@testset "stat_sop_bp 2D — TauTilde, w=1 equals stat_sop squared" begin

    bp   = stat_sop_bp(_DATA_BP, 1; chart_choice=TauTilde())
    stat, _ = stat_sop(_DATA_BP, 1, 1; chart_choice=TauTilde())

    @test bp ≈ stat^2
    @test bp ≈ (3 / 4 - 1 / 3)^2   # = (5/12)^2, derived from known SOP frequencies

end

@testset "stat_sop_bp 2D — Shannon, w=1 equals stat_sop squared" begin

    bp   = stat_sop_bp(_DATA_BP, 1; chart_choice=Shannon())
    stat, _ = stat_sop(_DATA_BP, 1, 1; chart_choice=Shannon())

    @test bp ≈ stat^2
    @test bp ≥ 0.0

end

@testset "stat_sop_bp 2D — TauTilde and Shannon give different bp_stat" begin

    bp_tt = stat_sop_bp(_DATA_BP, 1; chart_choice=TauTilde())
    bp_sh = stat_sop_bp(_DATA_BP, 1; chart_choice=Shannon())

    @test bp_tt ≉ bp_sh

end

@testset "stat_sop_bp 2D — w=2 is sum over four (d1,d2) combinations" begin

    # With w=2 the combinations are (1,1),(1,2),(2,1),(2,2).
    # Manually sum the squared per-combination statistics.
    bp_w2 = stat_sop_bp(_DATA_BP, 2; chart_choice=TauTilde())

    expected = 0.0
    for d1 in 1:2, d2 in 1:2
        s, _ = stat_sop(_DATA_BP, d1, d2; chart_choice=TauTilde())
        expected += s^2
    end

    @test bp_w2 ≈ expected

end

# -----------------------------------------------------------------------
# 3D EWMA overload: returns a vector of length size(data,3), all ≥ 0
# -----------------------------------------------------------------------
@testset "stat_sop_bp 3D — return shape and non-negativity" begin

    Random.seed!(42)
    data_3d = rand(10, 10, 20)
    lam = 0.2
    w   = 1

    bp_vec = stat_sop_bp(data_3d, lam, w; chart_choice=TauTilde())

    @test length(bp_vec) == size(data_3d, 3)
    @test all(bp_vec .≥ 0.0)

end

@testset "stat_sop_bp 3D — TauTilde and Shannon give different results" begin

    Random.seed!(42)
    data_3d = rand(10, 10, 10)
    lam = 0.2
    w   = 1

    bp_tt = stat_sop_bp(data_3d, lam, w; chart_choice=TauTilde())
    bp_sh = stat_sop_bp(data_3d, lam, w; chart_choice=Shannon())

    @test any(bp_tt .≉ bp_sh)

end
