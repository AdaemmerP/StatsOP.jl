

# Packages to use
using Pkg
Pkg.activate()
using BenchmarkTools
using JLD2

Pkg.activate("tutorials/.")
using Random
using LinearAlgebra
using Statistics
#using Combinatorics
using Distributions
using Distributed
using OrdinalPatterns
using Printf

addprocs(10)
@everywhere using OrdinalPatterns
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

# Load critical values
matcl_sop = load("/home/adaemmerp/Dropbox/Greifswald/Forschung/Mit_PWitten_CWeiss/OrdinalPatterns/tutorials/replication_awwt_paper/matcl_sop_norm_1e6.jld2")["large_array"]
matcl_sacf = load("/home/adaemmerp/Dropbox/Greifswald/Forschung/Mit_PWitten_CWeiss/OrdinalPatterns/tutorials/replication_awwt_paper/matcl_sacf_norm_1e6.jld2")["large_array"]

#-------------------------------------------------------------------------#
#                        Replication Table 1                              #
#-------------------------------------------------------------------------#
mn_vec = [(10, 10); (15, 15); (25, 25); (40, 25)]
lambdas = [0.05; 0.1; 0.25]
reps = 1_000
verbose = false
jmin = 4
jmax = 7
L0 = 370
dist = Normal(0, 1)

@time for mn in mn_vec

    m = mn[1]
    n = mn[2]

    for lam = lambdas
        for chart_choice in 1:4

            println("#--------------------------------------------------------#")
            println("m = ", m, " n = ", n, " λ = ", lam, " chart = ", chart_choice)
            println("#--------------------------------------------------------#")
            init_cl = abs(init_vals_sop(m, n, lam, chart_choice, Normal(0, 1), reps, 0.99)[2])
            final_cl_sop = cl_sop(m, n, lam, L0, reps, init_cl, jmin, jmax, verbose, chart_choice, dist)
            println("value sop = ", round(final_cl_sop; digits=6))
        end
        init_cl = abs(init_vals_sop(m, n, lam, 4, Normal(0, 1), reps, 0.99)[2])
        final_cl_sacf = cl_sacf(m, n, lam, L0, reps, init_cl, jmin, jmax, verbose, dist)
        println("value sacf = ", round(final_cl_sacf; digits=6))
    end
end

#-------------------------------------------------------------------------#
#                        Replication Table 2                              #
#-------------------------------------------------------------------------#
mn_vec = [(10, 10); (15, 15); (25, 25); (40, 25)]
dists = [TDist(2); PoiBin(0.2, 5); Weibull(1, 1.5); Exponential(1); Poisson(0.5);  Laplace(0, 1); SkewNormal(0, 1, 10); Uniform(0, 1); BinNorm(-9, 9, 1, 1); Bernoulli(0.5)]
reps = 10^5

@time for (mn_i, mn) in enumerate(mn_vec)

    m = mn[1]
    n = mn[2]
    mn_i += 1 # Add one, to exclude m = 1, n = 1   

    println()
    println("#"*"-"^90*"#")
    println("m ", "  n")  
    println(m, " ", n)  
    for dist in dists

        
        #println("dist = ", typeof(dist))        
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]
        rl_value = arl_sacf(m, n, 0.1, cl_sacf_val, reps, dist)
        #print(round(rl_value; digits=2), " ")
        @printf("%.2f ", rl_value[1])
        print("(")
        @printf("%.2f ", rl_value[2])
        print(")")
        print(" - ")

    end

end



# #-------------------------------------------------------------------------#
# #                        IC Functions                                     #
# #-------------------------------------------------------------------------#
# @time arl_sop(m, n, 0.1, cl_3, 1000, 3, Normal(0, 1))
# @time arl_sop(m, n, 0.1, cl_3, 1000, 3, Poisson(5))

# @time arl_sacf(m, n, 0.1, cl_acf, 100000, Normal(0, 1))
# @time arl_sop(m, n, 0.1, cl_3, 5000, 3, Poisson(5))

#-------------------------------------------------------------------------#
#              Replication Table A.1                                      #
#-------------------------------------------------------------------------#
alpha_params = ((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.2, 0.2, 0.5), (0.4, 0.3, 0.1))
charts = 1:4
reps = Int(1e3)
prerun = 100
lam = 0.1


# Compute Table A.1
dist_error = Normal(0, 1)
dist_ao = nothing
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1
        dgp_spatial = SAR11(alpha, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Table A.2                                        #
#-------------------------------------------------------------------------#
dist_error = Poisson(5)
dist_ao = nothing
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1
        dgp_spatial = SINAR11(alpha, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Table A.3-1                                      #
#-------------------------------------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialC(0.1, 10)
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1
        dgp_spatial = SAR11(alpha, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Table A.3-2                                      #
#-------------------------------------------------------------------------#
dist_error = Normal(0, 1)
dist_ao = BinomialCvec(0.1, [-10; 10])
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1
        dgp_spatial = SAR11(alpha, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Tables A.4                                       #
#-------------------------------------------------------------------------#
dist_error = Poisson(5)
dist_ao = PoiBin(0.1, 25)
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1
        dgp_spatial = SINAR11(alpha, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Tables A.5                                       #
#-------------------------------------------------------------------------#
dist_error = ZIP(5, 0.9)
dist_ao = nothing
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1
        dgp_spatial = SINAR11(alpha, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Table A.6                                        #
#-------------------------------------------------------------------------#
eps_params = ((2, 2, 2), (2, 1, 2), (1, 1, 2), (2, 1, 1))

dist_error = Normal(0, 1)
dist_ao = nothing
for eps_param in eps_params
    println("#--------------------------------------------------------#")
    println("eps_param = ", eps_param)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1

        dgp_spatial = SQMA11((0.8, 0.8, 0.8), eps_param, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Table A.7                                        #
#-------------------------------------------------------------------------#
eps_params = ((2, 2, 2), (2, 1, 2), (1, 1, 2), (2, 1, 1))

dist_error = Poisson(5)
dist_ao = nothing
for eps_param in eps_params
    println("#--------------------------------------------------------#")
    println("eps_param = ", eps_param)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1

        dgp_spatial = SQINMA11((0.8, 0.8, 0.8), eps_param, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Table A.8                                        #
#-------------------------------------------------------------------------#
margin = 20
alpha_params = ((0.1, 0.1, 0.1, 0.1), (0.05, 0.05, 0.15, 0.15), (0.05, 0.15, 0.05, 0.15))

dist_error = Normal(0, 1)
dist_ao = nothing
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1

        dgp_spatial = SAR1(alpha, m, n, margin)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end


#-------------------------------------------------------------------------#
#              Replicate Table A.9                                        #
#-------------------------------------------------------------------------#
margin = 20
alpha_params = ((0.1, 0.1, 0.1, 0.1), (0.05, 0.05, 0.15, 0.15), (0.05, 0.15, 0.05, 0.15))

dist_error = Normal(0, 1)
dist_ao = BinomialCvec(0.1, [-5; 5])
for alpha in alpha_params
    println("#--------------------------------------------------------#")
    println("alpha = ", alpha)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1

        dgp_spatial = SAR1(alpha, m, n, margin)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end

#-------------------------------------------------------------------------#
#              Replicate Table A.10 for m = n = 10                         #
#-------------------------------------------------------------------------#
b_params = (0.8, 0.8, 0.8, 0.8)
eps_params = ((2, 2, 2, 2), (2, 1, 2, 1), (2, 2, 1, 1))

dist_error = Normal(0, 1)
dist_ao = nothing
for eps_param in eps_params
    println("#--------------------------------------------------------#")
    println("eps_param = ", eps_param)
    for (mn_i, mn) in enumerate(mn_vec)

        # Create DGP
        m = mn[1]
        n = mn[2]
        mn_i += 1 # Add one, to exclude m = 1, n = 1

        dgp_spatial = BSQMA11((0.8, 0.8, 0.8, 0.8), eps_param, m, n, prerun)

        # Compute ARLs
        cl_sop_1 = matcl_sop[mn_i, mn_i, 2, 1]
        cl_sop_2 = matcl_sop[mn_i, mn_i, 2, 2]
        cl_sop_3 = matcl_sop[mn_i, mn_i, 2, 3]
        cl_sop_4 = matcl_sop[mn_i, mn_i, 2, 4]
        cl_sacf_val = matcl_sacf[mn_i, mn_i, 2]

        arl_sop_1 = arl_sop(lam, cl_sop_1, reps, 1, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_2 = arl_sop(lam, cl_sop_2, reps, 2, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_3 = arl_sop(lam, cl_sop_3, reps, 3, dgp_spatial, dist_error, dist_ao)[1]
        arl_sop_4 = arl_sop(lam, cl_sop_4, reps, 4, dgp_spatial, dist_error, dist_ao)[1]
        arl_sacf_1 = arl_sacf(lam, cl_sacf_val, reps, dgp_spatial, dist_error, dist_ao)[1]

        print(round(arl_sop_1; digits=2), " ; ")
        print(round(arl_sop_2; digits=2), " ; ")
        print(round(arl_sop_3; digits=2), " ; ")
        print(round(arl_sop_4; digits=2), " ; ")
        println(round(arl_sacf_1; digits=2))
    end
end
