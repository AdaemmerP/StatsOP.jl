export competerank!,
    compute_lookup_array_gop,
    fill_p0!,
    abort_criterium_gop

# Competition ranking ("1224" ranking) 
# Code is based on StatsBase.jl
function competerank!(
    rks::AbstractArray, # vector with final ranks
    x::AbstractArray, # data vector
    ix::AbstractArray # vector with indices for sortperm!
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

    return rks
end

#--- Lookup function for GOPs
function compute_lookup_array_gop()

    # Construct all possible ordinal patterns, Equation (2), page 574, 
    # and Equation (4), page 575 Weiss and Schnurr (2024)
    ranks = [
        1 2 3;
        1 3 2;
        2 1 3;
        2 3 1;
        3 1 2;
        3 2 1;
        1 1 1;
        1 1 3; # In paper: 1 1 2
        1 3 1; # In paper: 1 2 1;
        1 2 2;
        3 1 1; # In paper: 2 1 1;
        2 1 2;
        2 2 1
    ]

    # Construct multi-dimensional lookup array     
    lookup_array = zeros(Int, 3, 3, 3)

    for i in axes(ranks, 1)
        @views j, k, l = ranks[i, :]
        lookup_array[j, k, l] = i
    end

    return lookup_array

end


# Function to fill p0 vector for GOPs
function fill_p0!(p0, gop_dgp_dist)
    # in-control GOP-distribution depends on the specified in-control model for (Xt) 
    # see Weiss and Schnurr (2023), page 577 proposition 2.3 for details how to fill the vector p0
    # compute upper limit for the approximation
    # only two distributions implemented so far
    if gop_dgp_dist isa Poisson
        q = quantile(gop_dgp_dist, 1 - (1 * 10^-12))
    elseif gop_dgp_dist isa Binomial
        q = gop_dgp_dist.n
    elseif println("Distribution not impleneted.")
    end

    # p(1,1,1)
    for x in 0:q
        p0[7] += pdf(gop_dgp_dist, x)^3
    end

    # p(1,2,2)=p(2,1,2)=p(2,2,1)
    val_tmp = 0.0
    for x in 0:q
        val_tmp += cdf(gop_dgp_dist, max(0, x - 1)) * pdf(gop_dgp_dist, x)^2
    end
    p0[[10, 12, 13]] .= val_tmp

    # p(1,1,2)=p(1,2,1)=p(2,1,1)
    val_tmp = 0.0
    for x in 0:q
        val_tmp += pdf(gop_dgp_dist, x)^2 * (1 - cdf(gop_dgp_dist, x))
    end
    p0[[8, 9, 11]] .= val_tmp

    # p(1,2,3)=p(3,2,1)=p(3,1,2)=p(2,3,1)=p(1,3,2)=p(2,1,3)
    val_tmp = 0.0
    for x in 1:q
        val_tmp += cdf(gop_dgp_dist, max(0, x - 1)) * pdf(gop_dgp_dist, x) * (1 - cdf(gop_dgp_dist, x))
    end
    p0[1:6] .= val_tmp
end



# --- Function to select abort criterium --- #
# see Equation (18) and Equation (20), page 7 in the paper
function abort_criterium_gop(stat, cl, ::Union{D_Chart,G_Chart,Persistence})

    # D-chart: Equation (18), page 7 in the paper      
    return stat > cl

end



