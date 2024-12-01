# -------------------------------------------------#
# --------------- In-control methods---------------#
# -------------------------------------------------#
# Method to initialize in-control when d is Int
function init_dgp_op!(dgp::IC, x_long, eps_long, dist_error, d::Int, xbiv)
    rand!(dist_error, x_long)
    return @views x_long[1:d:end]
end

# Method to update in-control when d is Int
function update_dgp_op!(dgp::IC, x_long, eps_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(dist_error)
    return @views x_long[1:d:end]
end

# Method to initialize in-control when d is Vector{Int}
function init_dgp_op!(dgp::IC, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
    rand!(dist_error, x_long)
    return @views x_long[d]
end

# Method to update in-control when d is Vector{Int}
function update_dgp_op!(dgp::IC, x_long, eps_long, dist_error, d::Vector{Int})
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(dist_error)
    return @views x_long[d]
end

# -------------------------------------------------#
# ---------------  AR(1) methods    ---------------#
# -------------------------------------------------#

# Initialize AR(1) when d is Int 
function init_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Int, xbiv)
    x = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
    x_long[1] = dgp.α * x + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * x_long[i-1] + rand(dist_error)
    end
    return @views x_long[1:d:end]
end

# Update AR(1) when d is Int 
function update_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * x_long[2] + rand(dist_error)
    return @views x_long[1:d:end]
end

# Initialize AR(1) when d is Vector{Int}
function init_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
    x = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
    x_long[1] = dgp.α * x + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * x_long[i-1] + rand(dist_error)
    end
    return @views x_long[d]
end

# Update AR(1) when d is Vector{Int}
function update_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Vector{Int})
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * x_long[end-1] + rand(dist_error)
    return @views x_long[d]
end


# Initialize AR(1) for CED when d is Int 
function init_dgp_op_ced!(dgp::AR1, x_long, d::Int)
    rand!(Normal(0, sqrt(1 / (1 - dgp.α^2))), x_long)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
    return @views x_long[1:d:end]
end

# Update AR(1) for CED when d is Int
function update_dgp_op_ced!(dgp::AR1, x_long, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
    return @views x_long[1:d:end]
end

# -------------------------------------------------#
# ---------------  MA(1) methods    ---------------#
# -------------------------------------------------#
# Initialize MA(1) when d is Int 
function init_dgp_op!(dgp::MA1, x_long, eps_long, dist_error, d::Int, xbiv)
    rand!(dist_error, eps_long)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * eps_long[i-1]^2 + eps_long[i]
    end
    return @views x_long[2:d:end]
end

# Update MA(1) when d is Int
function update_dgp_op!(dgp::MA1, x_long, eps_long, dist_error, d::Int)
    # Update eps sequence
    for i in 1:(lastindex(eps_long)-1)
        eps_long[i] = eps_long[i+1]
    end
    eps_long[end] = rand(dist_error)
    # Update x sequence
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * eps_long[i-1]^2 + eps_long[i]
    end
    return @views x_long[2:d:end]
end

# # Initialize MA(1) when d is Vector{Int}
# function init_dgp_op!(dgp::MA1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
#   rand!(dist_error, eps_long)
#   for i in 2:lastindex(x_long)
#       x_long[i] = dgp.α * eps_long[i-1]^2 + eps_long[i]
#   end
#   return @views x_long[d]  
# end

# # Update MA(1) when d is Vector{Int}
# function update_dgp_op!(dgp::MA1, x_long, eps_long, dist_error, d::Vector{Int})
#   # Update eps sequence
#   for i in 1:(lastindex(eps_long) - 1)
#       eps_long[i] = eps_long[i+1]
#   end
#   eps_long[end] = rand(dist_error)
#   # Update x sequence
#   for i in 2:lastindex(x_long)
#       x_long[i] = dgp.α * eps_long[i-1]^2 + eps_long[i]     
#   end
#   return @views x_long[d]
# end


# -------------------------------------------------#
# ---------------  MA(2) methods    ---------------#
# -------------------------------------------------#
# Initialize MA(2) when d is Int
function init_dgp_op!(dgp::MA2, x_long, eps_long, dist_error, d::Int, xbiv)
    rand!(dist_error, eps_long)
    for i in 3:lastindex(x_long)
        x_long[i] = dgp.α₁ * eps_long[i-1]^2 + dgp.α₂ * eps_long[i-2]^2 + eps_long[i]
    end
    return @views x_long[3:d:end]
end

# Update MA(2) when d is Int
function update_dgp_op!(dgp::MA2, x_long, eps_long, dist_error, d::Int)
    # Update eps sequence
    for i in 1:(lastindex(eps_long)-1)
        eps_long[i] = eps_long[i+1]
    end
    eps_long[end] = rand(dist_error)

    # Update x sequence
    for i in 3:lastindex(eps_long)
        x_long[i] = dgp.α₁ * eps_long[i-1]^2 + dgp.α₂ * eps_long[i-2]^2 + eps_long[i]
    end

    return @views x_long[3:d:end]
end

# # Initialize MA(2) when d is Vector{Int}
# function init_dgp_op!(dgp::MA2, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
#     rand!(dist_error, eps_long)
#     for i in 3:lastindex(x_long)
#         x_long[i] = dgp.α₁ * eps_long[i-1]^2 + dgp.α₂ * eps_long[i-2]^2 + eps_long[i]
#     end
#     return @views x_long[d]
# end

# # Update MA(2) when d is Vector{Int}
# function update_dgp_op!(dgp::MA2, x_long, eps_long, dist_error, d::Vector{Int})
#     # Update eps sequence
#     for i in 1:(lastindex(eps_long) - 1)
#         eps_long[i] = eps_long[i+1]
#     end
#     eps_long[end] = rand(dist_error)

#     # Update x sequence
#     for i in 3:lastindex(eps_long)
#         x_long[i] = dgp.α₁ * eps_long[i-1]^2 + dgp.α₂ * eps_long[i-2]^2 + eps_long[i]
#     end
#     return @views x_long[d]
# end

# -------------------------------------------------#
# ---------------  TEAR(1) methods  ---------------#
# -------------------------------------------------#

# Method to initialize TEAR(1) when d is Int 
function init_dgp_op!(dgp::TEAR1, x_long, eps_long, dist_error, d::Int, xbiv)
    x = rand(dist_error)
    x_long[1] = rand(Bernoulli(dgp.α)) * x + (1 - dgp.α) * rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = rand(Bernoulli(dgp.α)) * x_long[i-1] + (1 - dgp.α) * rand(dist_error)
    end
    return @views x_long[1:d:end]
end

# Method to update TEAR(1) 
function update_dgp_op!(dgp::TEAR1, x_long, eps_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(Bernoulli(dgp.α)) * x_long[end-1] + (1 - dgp.α) * rand(dist_error)
    return @views x_long[1:d:end]
end

# Method to initialize TEAR(1) when d is Vector{Int}
function init_dgp_op!(dgp::TEAR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
    x = rand(dist_error)
    x_long[1] = rand(Bernoulli(dgp.α)) * x + (1 - dgp.α) * rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = rand(Bernoulli(dgp.α)) * x_long[i-1] + (1 - dgp.α) * rand(dist_error)
    end
    return @views x_long[d]
end

# Method to update TEAR(1) when d is Vector{Int}
function update_dgp_op!(dgp::TEAR1, x_long, eps_long, dist_error, d::Vector{Int})
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(Bernoulli(dgp.α)) * x_long[end-1] + (1 - dgp.α) * rand(dist_error)
    return @views x_long[d]
end

# Initialize TEAR(1) for CED when d is Int
function init_dgp_op_ced!(dgp::TEAR1, x_long, d::Int)
    rand!(Exponential(1), x_long)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(Exponential(1))
    return @views x_long[1:d:end]
end

# Update TEAR(1) for CED
function update_dgp_op_ced!(dgp::TEAR1, x_long, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(Exponential(1))
    return @views x_long[1:d:end]
end

# -------------------------------------------------#
# ---------------  AAR(1) methods   ---------------#
# -------------------------------------------------#

# Method to initialize AAR(1) when d is Int 
function init_dgp_op!(dgp::AAR1, x_long, eps_long, dist_error, d::Int, xbiv)
    # Burn-in prerun 
    xbiv[1] = rand(dist_error)
    for i in 2:lastindex(xbiv)
        xbiv[i] = dgp.α * abs(xbiv[i-1]) + rand(dist_error)
    end
    # Initialize sequence
    x_long[1] = dgp.α * abs(xbiv[end]) + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * abs(x_long[i-1]) + rand(dist_error)
    end
    return @views x_long[1:d:end]
end

# Method to update AAR(1) when d is Int
function update_dgp_op!(dgp::AAR1, x_long, eps_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * abs(x_long[end-1]) + rand(dist_error)
    return @views x_long[1:d:end]
end

# Method to initialize AAR(1) when d is Vector{Int}
function init_dgp_op!(dgp::AAR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
    # Burn-in prerun 
    xbiv[1] = rand(dist_error)
    for i in 2:lastindex(xbiv)
        xbiv[i] = dgp.α * abs(xbiv[i-1]) + rand(dist_error)
    end
    # Initialize sequence
    x_long[1] = dgp.α * abs(xbiv[end]) + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * abs(x_long[i-1]) + rand(dist_error)
    end
    return @views x_long[d]
end

# Method to update AAR(1) when d is Vector{Int}
function update_dgp_op!(dgp::AAR1, x_long, eps_long, dist_error, d::Vector{Int})
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * abs(x_long[2]) + rand(dist_error)
    return @views x_long[d]
end

# -------------------------------------------------#
# ---------------  QAR(1) methods   ---------------#
# -------------------------------------------------#

# Method to initialize QAR(1) when d is Int 
function init_dgp_op!(dgp::QAR1, x_long, eps_long, dist_error, d::Int, xbiv)
    # Burn-in prerun
    xbiv[1] = rand(dist_error)
    for i in 2:lastindex(xbiv)
        xbiv[i] = dgp.α * xbiv[i-1]^2 + rand(dist_error)
    end
    # Initialize sequence
    x_long[1] = dgp.α * xbiv[end]^2 + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * x_long[i-1]^2 + rand(dist_error)
    end
    return @views x_long[1:d:end]
end

# Method to update QAR(1) when d is Int 
function update_dgp_op!(dgp::QAR1, x_long, eps_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * x_long[end-1]^2 + rand(dist_error)
    return @views x_long[1:d:end]
end

# Method to initialize QAR(1) when d is Vector{Int}
function init_dgp_op!(dgp::QAR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
    # Burn-in prerun
    xbiv[1] = rand(dist_error)
    for i in 2:lastindex(xbiv)
        xbiv[i] = dgp.α * xbiv[i-1]^2 + rand(dist_error)
    end
    # Initialize sequence
    x_long[1] = dgp.α * xbiv[end]^2 + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * x_long[i-1]^2 + rand(dist_error)
    end
    return @views x_long[d]
end

# Method to update QAR(1) when d is Vector{Int}
function update_dgp_op!(dgp::QAR1, x_long, eps_long, dist_error, d::Vector{Int})
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * x_long[end-1]^2 + rand(dist_error)
    return @views x_long[d]
end