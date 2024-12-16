
using OrdinalPatterns

@testset "run sop functions" begin


    m = 10
    n = 10
    lam = 0.1
    cl = 0.03049
    reps = 1000
    chart_choice = 1
    dist = Normal(0, 1)

    arl_sop(m, n, lam, cl, reps, chart_choice, dist)[1] 

end