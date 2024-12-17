
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

# Test delay functions
function test(M_rows, N_cols, d1, d2, sop)

    data = reshape(collect(1:M_rows*N_cols), M_rows, N_cols)'
    @show data

    m = M_rows - d1
    n = N_cols - d2
  
    # Loop through data to fill sop vector
    for j in 1:n
      for i in 1:m
  
        sop[1] = data[i,    j]
        sop[2] = data[i,    j+d2]
        sop[3] = data[i+d1, j]
        sop[4] = data[i+d1, j+d2]
 
         @show [
            sop[1] sop[2];
            sop[3] sop[4]
            ]

      end
    end
   
  end

  test(6, 6, 2, 2, zeros(4))
