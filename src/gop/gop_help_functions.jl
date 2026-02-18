export competerank!,
    compute_lookup_array_gop,
    fill_p0!,
    abort_criterium_gop

# Competition ranking ("1224" ranking) 
# Code is based on StatsBase.jl
function competerank!(
    rks::AbstractArray, # vector for final ranks
    x::AbstractArray, # data vector
    ix::AbstractArray # vector for indices for sortperm!
)

    # Check input
    @assert length(rks) == length(x) "Rank vector and data vector must have the same length."
    @assert length(ix) == length(x) "Index vector and data vector must have the same length."

    n = length(x)
    sortperm!(ix, x)

    p1 = ix[1]
    v = x[p1]
    rks[p1] = k = 1

    for i in 2:n
        pi = ix[i]
        xi = x[pi]
        if xi != v
            v = xi
            k = i
        end
        rks[pi] = k
    end

    # return rks
end

#--- Lookup function for GOPs
function compute_lookup_array_gop()

    # Construct all possible ordinal patterns, Equation (2), page 574, 
    # and Equation (4), page 575 Weiss and Schnurr (2024)
    ranks = [
        1 2 3; # 1
        1 3 2; # 2
        2 1 3; # 3
        2 3 1; # 4
        3 1 2; # 5
        3 2 1; # 6
        1 1 1; # 7
        1 1 3; # 8 In paper: 1 1 2
        1 3 1; # 9 In paper: 1 2 1;
        1 2 2; # 10
        3 1 1; # 11 In paper: 2 1 1;
        2 1 2; # 12
        2 2 1  # 13
    ]

    # Construct multi-dimensional lookup array     
    lookup_array = zeros(Int, 3, 3, 3)

    for i in axes(ranks, 1)
        @views j, k, l = ranks[i, :]
        lookup_array[j, k, l] = i
    end

    return lookup_array

end

"""
    find_effective_support(dist::Distribution; p_low=1e-6, p_high=1-1e-6, max_extra=100)

Calculates a practical integer range [lb, ub] for distributions with infinite support 
(like Skellam, Poisson, or NegativeBinomial) where the probability mass outside 
this range is less than `p_low` and `1 - p_high`.
"""
function find_effective_support(
    dist::DiscreteUnivariateDistribution; p_low=1e-10, p_high=1 - 1e-10, max_extra=100
)
    μ = mean(dist)
    σ = std(dist)

    h_min = isfinite(minimum(dist)) ? Int(minimum(dist)) : typemin(Int)
    h_max = isfinite(maximum(dist)) ? Int(maximum(dist)) : typemax(Int)

    lb = max(h_min, Int(floor(μ - 6σ)))
    ub = min(h_max, Int(ceil(μ + 6σ)))

    # Refine Lower Bound with a safety counter
    count = 0
    while lb > h_min && cdf(dist, lb) > p_low && count < max_extra
        lb -= 1
        count += 1
    end

    # Refine Upper Bound with a safety counter
    count = 0
    while ub < h_max && cdf(dist, ub) < p_high && count < max_extra
        ub += 1
        count += 1
    end

    return (lb, ub)
end

function get_bounds(d::DiscreteUnivariateDistribution)::Tuple{Int,Int}
    if isfinite(minimum(d)) && isfinite(maximum(d))
        return Int(minimum(d)), Int(maximum(d))
    else
        return find_effective_support(d)
    end
end

function fill_p0!(p0, dist_null)

    # 1. Determine support boundaries
    sup_lb, sup_ub = get_bounds(dist_null)

    # 2. Pre-calculate PDF and CDF values
    # Include (sup_lb - 1) to ensure cdf_dict[x-1] is defined
    search_range = (sup_lb-1):sup_ub
    pdf_dict = Dict(x => pdf(dist_null, x) for x in search_range)
    cdf_dict = Dict(x => cdf(dist_null, x) for x in search_range)

    # Initialize p0
    p0 .= 0.0

    # p(1,1,1) = E[p(X)^2]
    for x in sup_lb:sup_ub
        p0[7] += pdf_dict[x]^3
    end

    # p(1,2,2) = p(2,1,2) = p(2,2,1) = E[f(X-1) * p(X)]
    val_122 = 0.0
    for x in sup_lb:sup_ub
        val_122 += cdf_dict[x-1] * pdf_dict[x]^2
    end
    p0[[10, 12, 13]] .= val_122

    # p(1,1,2) = p(1,2,1) = p(2,1,1) = E[p(X) * (1 - f(X))]
    val_112 = 0.0
    for x in sup_lb:sup_ub
        val_112 += pdf_dict[x]^2 * (1 - cdf_dict[x])
    end
    p0[[8, 9, 11]] .= val_112

    # p(1,2,3) = ... = E[f(X-1) * p(X) * (1 - f(X))]
    val_123 = 0.0
    for x in sup_lb:sup_ub
        val_123 += cdf_dict[x-1] * pdf_dict[x] * (1 - cdf_dict[x])
    end
    p0[1:6] .= val_123

    return p0
end

# # Function to fill p0 vector for GOPs
# function fill_p0!(p0, dist_null)
#     # in-control GOP-distribution depends on the specified in-control model for (Xt) 
#     # see Weiss and Schnurr (2023), page 577 proposition 2.3 for details how to fill the vector p0
#     # compute upper limit for the approximation
#     # only two distributions implemented so far
#     if dist_null isa Poisson
#         q = quantile(dist_null, 1 - (1 * 10^-12))
#     elseif dist_null isa Binomial
#         q = dist_null.n
#     elseif println("Distribution not impleneted.")
#     end

#     # Create dictionary with outcome (key) and probability (value) pairs
#     outcome = -1:q
#     pdf_dict = Dict(
#         zip(outcome, pdf(dist_null, outcome))
#     )
#     cdf_dict = Dict(
#         zip(outcome, cdf(dist_null, outcome))
#     )

#     # p(1,1,1)
#     for x in 0:q
#         p0[7] += pdf_dict[x]^3
#     end

#     # p(1,2,2)=p(2,1,2)=p(2,2,1)
#     val_tmp = 0.0
#     for x in 1:q
#         val_tmp += cdf_dict[x-1] * pdf_dict[x]^2
#     end
#     p0[[10, 12, 13]] .= val_tmp

#     # p(1,1,2)=p(1,2,1)=p(2,1,1)
#     val_tmp = 0.0
#     for x in 0:q
#         val_tmp += pdf_dict[x]^2 * (1 - cdf_dict[x])
#     end
#     p0[[8, 9, 11]] .= val_tmp

#     # p(1,2,3)=p(3,2,1)=p(3,1,2)=p(2,3,1)=p(1,3,2)=p(2,1,3)
#     val_tmp = 0.0
#     for x in 1:q
#         val_tmp += cdf_dict[x-1] * pdf_dict[x] * (1 - cdf_dict[x])
#     end
#     p0[1:6] .= val_tmp
# end

# --- Function to select abort criterium --- #
# see Equation (18) and Equation (20), page 7 in the paper
function abort_criterium_gop(stat, cl, ::Union{D_Chart,G_Chart,Persistence})

    # D-chart: Equation (18), page 7 in the paper      
    return stat > cl

end



