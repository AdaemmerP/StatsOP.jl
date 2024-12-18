

# function arl_sop(lam, cl, spatial_dgp::SpatialDGP, reps=10_000; chart_choice=3, d1_vec::Vector{Int}, d2_vec::Vector{Int})

#   # Compute m and n
#   dist_error = spatial_dgp.dist
#   dist_ao = spatial_dgp.dist_ao

#   # Compute lookup array to finde SOPs
#   lookup_array_sop = compute_lookup_array()

#   # Check whether to use threading or multi processing --> only one process threading, else distributed
#   if nprocs() == 1

#     # Make chunks for separate tasks        
#     chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect
#     # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
#     par_results = map(chunks) do i

#       Threads.@spawn rl_sop(lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice, d1_vec, d2_vec)

#     end

#   elseif nprocs() > 1 # Multi Processing

#     chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
#     par_results = pmap(chunks) do i

#       rl_sop(lam, cl, lookup_array_sop, i, spatial_dgp, dist_error, dist_ao, chart_choice, d1_vec, d2_vec)

#     end

#   end
#   # Collect results from tasks
#   rls = fetch.(par_results)
#   rlvec = Iterators.flatten(rls) |> collect
#   return (mean(rlvec), std(rlvec) / sqrt(reps))
# end


# function rl_sop(lam, cl, lookup_array_sop, p_reps, spatial_dgp, dist_error::UnivariateDistribution, dist_ao::Union{Nothing,UnivariateDistribution}, chart_choice, d1_vec::Vector{Int}, d2_vec::Vector{Int})

#   # find maximum values of d1 and d2 for construction of matrices
#   d1_max = maximum(d1_vec)
#   d2_max = maximum(d2_vec)

#   # Compute all possible combinations of d1 and d2
#   d1_d2_combinations = Iterators.product(d1_vec, d2_vec)

#   # pre-allocate
#   sop_freq = zeros(Int, 24)
#   win = zeros(Int, 4)
#   data = zeros(m + d1_max, n + d2_max)
#   p_ewma = zeros(3)
#   p_hat = zeros(3)
#   rls = zeros(Int, length(p_reps))
#   sop = zeros(4)

#   # pre-allocate indexes to compute sum of frequencies
#   s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
#   s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
#   s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

#   # pre-allocate mat, mat_ao and mat_ma
#   # mat:    matrix for the final values of the spatial DGP
#   # mat_ao: matrix for additive outlier 
#   # mat_ma: matrix for moving averages
#   # vec_ar: vector for SAR(1) model
#   # vec_ar2: vector for in-place multiplication for SAR(1) model
#   if spatial_dgp isa SAR1
#     mat = build_sar1_matrix(spatial_dgp) # will be done only once
#     mat_ao = zeros((m + d1_max + 2 * spatial_dgp.margin), (n + d2_max + 2 * spatial_dgp.margin))
#     vec_ar = zeros((m + d1_max + 2 * spatial_dgp.margin) * (n + d2_max + 2 * spatial_dgp.margin))
#     vec_ar2 = similar(vec_ar)
#   elseif spatial_dgp isa BSQMA11
#     mat = zeros(m + spatial_dgp.prerun + d1_max, n + spatial_dgp.prerun + 1)
#     mat_ma = zeros(m + spatial_dgp.prerun + d1_max + 1, n + spatial_dgp.prerun + d2_max + 1) # add one extra row and column for "forward looking" BSQMA11
#     mat_ao = similar(mat)
#   else
#     mat = zeros(m + spatial_dgp.prerun + d1_max, n + spatial_dgp.prerun + d2_max)
#     mat_ma = similar(mat)
#     mat_ao = similar(mat)
#   end

#   for r in 1:length(p_reps)

#     fill!(p_ewma, 1.0 / 3.0)
#     stat = chart_stat_sop(p_ewma, chart_choice)
#     bp_stat = 0.0

#     # Re-initialize matrix 
#     if spatial_dgp isa SAR1
#       # do nothing, 'mat' will not be overwritten for SAR1
#     else
#       fill!(mat, 0)
#       init_mat!(spatial_dgp, dist_error, spatial_dgp.dgp_params, mat)
#     end

#     rl = 0

#     while bp_stat < cl # BP-statistic can only be positive
#       rl += 1

#       # Fill matrix with dgp 
#       if spatial_dgp isa SAR1
#         data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, vec_ar, vec_ar2)
#       else
#         data .= fill_mat_dgp_sop!(spatial_dgp, dist_error, dist_ao, mat, mat_ao, mat_ma)
#       end

#       # Check whether to add noise to count data
#       if dist_error isa DiscreteUnivariateDistribution
#         for j in 1:size(data, 2)
#           for i in 1:size(data, 1)
#             data[i, j] = data[i, j] + rand()
#           end
#         end
#       end

#       # -------------------------------------------------------------------------------#
#       # ----------------     Loop for BP-Statistik                     ----------------#
#       # -------------------------------------------------------------------------------#
#       for d1d2_tup in d1_d2_combinations

#         d1_tmp = d1d2_tup[1]
#         d2_tmp = d1d2_tup[2]
#         m_tmp = spatial_dgp.M_rows - d1_tmp
#         n_tmp = spatial_dgp.N_cols - d2_tmp

#         # Compute sum of frequencies for each pattern group
#         sop_frequencies!(m_tmp, n_tmp, d1_tmp, d2_tmp, lookup_array_sop, data, sop, win, sop_freq)

#         # Fill 'p_hat' with sop-frequencies and compute relative frequencies
#         fill_p_hat!(p_hat, chart_choice, sop_freq, m_tmp, n_tmp, s_1, s_2, s_3)

#         # Apply EWMA to p-vectors
#         @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

#         # Compute test statistic for one d1-d2 combination
#         stat_tmp = chart_stat_sop(p_ewma, chart_choice)
#         # Compute BP-statistic
#         bp_stat += stat_tmp^2

#         # Reset win, sop_freq and p_hat
#         fill!(win, 0)
#         fill!(sop_freq, 0)
#         fill!(p_hat, 0)
#       end
#       # -------------------------------------------------------------------------------#
#       # -------------------------------------------------------------------------------#
#     end

#     rls[r] = rl
#   end
#   return rls
# end


# function cl_sop(lam, L0, sop_dgp::ICSP, cl_init, reps=10_000; chart_choice=3, jmin=4, jmax=6, verbose=false, d1_vec::Vector{Int}, d2_vec::Vector{Int})


#   for j in jmin:jmax
#     for dh in 1:40
#       cl_init = cl_init + (-1)^j * dh / 10^j
#       L1 = arl_sop(lam, cl_init, sop_dgp, reps; chart_choice, d1_vec=d1_vec, d2_vec=d2_vec)[1]
#       if verbose
#         println("cl = ", cl_init, "\t", "ARL = ", L1)
#       end
#       if (j % 2 == 1 && L1 < L0) || (j % 2 == 0 && L1 > L0)
#         break
#       end
#     end
#     cl_init = cl_init
#   end

#   if L1 < L0
#     cl_init = cl_init + 1 / 10^jmax
#   end
#   return cl_init
# end


# function stat_sop(data::Union{SubArray,Matrix{T}}; chart_choice=3, d1::Vector{Int}, d2::Vector{Int}) where {T<:Real}

#   # Print message
#   println("Computing BP-statistic for SOP chart ", chart_choice)

#   # Compute 4 dimensional cube to lookup sops
#   lookup_array_sop = compute_lookup_array()
#   p_hat = zeros(3)
#   sop = zeros(4)
#   sop_freq = zeros(Int, 24) # factorial(4)
#   win = zeros(Int, 4)
#   bp_stat = 0.0

#   # Compute all combinations of d1 and d2
#   d1_d2_combinations = Iterators.product(d1, d2)

#   # Pre-allocate indexes to compute sum of frequencies
#   s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
#   s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
#   s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

#   for d1d2_tup in d1_d2_combinations

#     d1_tmp = d1d2_tup[1]
#     d2_tmp = d1d2_tup[2]
#     m_tmp = spatial_dgp.M_rows - d1_tmp
#     n_tmp = spatial_dgp.N_cols - d2_tmp

#     # Compute sum of frequencies for each pattern group
#     sop_frequencies!(m_tmp, n_tmp, d1_tmp, d2_tmp, lookup_array_sop, data, sop, win, sop_freq)

#     # Fill 'p_hat' with sop-frequencies and compute relative frequencies
#     fill_p_hat!(p_hat, chart_choice, sop_freq, s_1, s_2, s_3)

#     # Compute test statistic
#     stat_tmp = chart_stat_sop(p_hat, chart_choice)
#     bp_stat += stat_tmp^2

#   end

#   return stat
# end

# function stat_sop(lam, data::Array{T,3}; chart_choice, add_noise, d1_vec::Vector{Int}, d2_vec::Vector{Int}) where {T<:Real}

#   println("Computing BP-statistic for SOP chart ", chart_choice)

#   # Compute 4 dimensional cube to lookup sops
#   lookup_array_sop = compute_lookup_array()
#   p_hat = zeros(3)
#   sop = zeros(4)
#   p_ewma = repeat([1.0 / 3.0], 3)  

#   # Pre-allocate for BP-computations
#   d1_d2_combinations = Iterators.product(d1_vec, d2_vec)
#   bp_stats_all = zeros(size(data, 3))

#   sop_freq = zeros(Int, 24) # factorial(4)
#   win = zeros(Int, 4)
#   M_rows = size(data, 1) 
#   N_cols = size(data, 2)

#   # Pre-allocate indexes to compute sum of frequencies
#   s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
#   s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
#   s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

#   for i = axes(data, 3)

#     if add_noise
#       data_tmp = data[:, :, i] + rand(m + 1, n + 1)
#     else
#       data_tmp = data[:, :, i]
#     end

#     # -------------------------------------------------------------------------------#
#     # ----------------     Loop for BP-Statistik                     ----------------#
#     # -------------------------------------------------------------------------------#
#     for d1d2_tup in d1_d2_combinations

#       d1_tmp = d1d2_tup[1]
#       d2_tmp = d1d2_tup[2]
#       m_tmp = M_rows - d1_tmp
#       n_tmp = N_cols - d2_tmp

#       # Compute frequencies of sops    
#       sop_frequencies!(m_tmp, n_tmp, d1_tmp, d2_tmp, lookup_array_sop, data_tmp, sop, win, sop_freq)

#       # Fill 'p_hat' with sop-frequencies and compute relative frequencies
#       fill_p_hat!(p_hat, chart_choice, sop_freq, m_tmp, n_tmp, s_1, s_2, s_3)

#       # Apply EWMA to p-vectors
#       @. p_ewma = (1 - lam) .* p_ewma .+ lam * p_hat

#       # Apply EWMA to p-vectors
#       stat_tmp = chart_stat_sop(p_ewma, chart_choice)

#       # Save temporary test statistic
#       bp_stats += stat_tmp^2 

#       # Reset win, sop_freq and p_hat
#       fill!(win, 0)
#       fill!(sop_freq, 0)
#       fill!(p_hat, 0)
#     end
#     bp_stats_all[i] = bp_stats

#   end

#   return stats_all

# end


test(a::Int = 1, b::Int=2) = a + b
test(a_vec::Vector{Int}=[1,1], b_vec::Vector{Int}=[1,1]) = a_vec + b_vec

foo(a::Vector{Int}, b::Vector{Int}) = a+b
foo(1, 1)