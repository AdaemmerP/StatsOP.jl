# -------------------------------------------------#
# --------------- In-control methods---------------#
# -------------------------------------------------#
# Method to initialize in-control for Continous Distribution
function init_dgp_op!(::ContinuousDGPIC, x_long, dist_error::ContinuousUnivariateDistribution, d::Int)
    rand!(dist_error, x_long)
    return @views x_long[1:d:end]
end

# Method to update in-control for Continous Distribution
function update_dgp_op!(::ContinuousDGPIC, x_long, dist_error::ContinuousUnivariateDistribution, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = rand(dist_error)
    return @views x_long[1:d:end]
end

# Method to initialize in-control for Discrete Distribution
function init_dgp_op!(dgp::DiscreteDGPIC, x_long, dist_error::DiscreteUnivariateDistribution, d::Int)
    rand!(dist_error, x_long)
    # add noise ?
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end
    return @views x_long[1:d:end]
end

# Method to update in-control for Discrete Distribution
function update_dgp_op!(dgp::DiscreteDGPIC, x_long, dist_error::DiscreteUnivariateDistribution, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    # add noise?
    if dgp.add_noise
        x_long[end] = rand(dist_error) + rand()
    else
        x_long[end] = rand(dist_error)
    end
    return @views x_long[1:d:end]
end

# -------------------------------------------------#
# ---------------  AR(1) methods    ---------------#
# -------------------------------------------------#
# Initialize AR(1)
function init_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Int, xbiv)
    x = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
    x_long[1] = dgp.α * x + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * x_long[i-1] + rand(dist_error)
    end
    return @views x_long[1:d:end]
end

# Update AR(1)
function update_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * x_long[end-1] + rand(dist_error)
    return @views x_long[1:d:end]
end

# Initialize AR(1)
function init_dgp_op!(dgp::AR1, x_long, dist_error, d::Int)
    x = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
    x_long[1] = dgp.α * x + rand(dist_error)
    for i in 2:lastindex(x_long)
        x_long[i] = dgp.α * x_long[i-1] + rand(dist_error)
    end
    return @views x_long[1:d:end]
end

# Update AR(1)
function update_dgp_op!(dgp::AR1, x_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x_long[end] = dgp.α * x_long[end-1] + rand(dist_error)
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


# -------------------------------------------------#
# ---------------  INAR(1) methods  ---------------#
# -------------------------------------------------#

# Method to initialize INAR(1) when d is Int 
function init_dgp_op!(dgp::INAR1, x_long, dist_error::Poisson, d::Int)
    x_long[1] = rand(Poisson(dist_error.λ / (1 - dgp.α)))
    for i in 2:lastindex(x_long)
        x_long[i] = rand(Binomial(x_long[i-1], dgp.α)) + rand(dist_error)
    end

    # add noise ?
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end
    return @views x_long[1:d:end]
end

# Method to update INAR(1) 
function update_dgp_op!(dgp::INAR1, x_long, dist_error::Poisson, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x = floor(x_long[end-1])

    # add noise?
    if dgp.add_noise
        x_long[end] = rand(Binomial(x, dgp.α)) + rand(dist_error) + rand()
    else
        x_long[end] = rand(Binomial(x, dgp.α)) + rand(dist_error)
    end

    return @views x_long[1:d:end]
end

# -------------------------------------------------#
# ---------------  SINAR(1) methods  ---------------#
# -------------------------------------------------#
# Method to initialize SINAR(1)
function init_dgp_op!(dgp::SINAR1, x_long, dist::DiscreteUnivariateDistribution, d::Int)
    α = dgp.α
    abs_α = abs(α)
    sign_α = sign(α)

    # 1. Burn-in Phase   
    # Start heuristically near steady state
    # Convert to Float64 to avoid type issues when adding noise 
    μ_ε = mean(dist) |> abs
    xₜ₋₁ = rand(Poisson(μ_ε / (1 - min(0.99, abs_α)))) |> Float64

    for _ in 1:dgp.burn_in

        # Signed binomial thinning operator: α ⊙ xₜ₋₁
        # -> sgn(α) * sgn(xₜ₋₁) * (|α| ∘ |xₜ₋₁|)
        operator_result = sign_α * sign(xₜ₋₁) * rand(Binomial(Int(abs(xₜ₋₁)), abs_α))

        # Compute observed value.
        εₜ = rand(dist)
        xₜ = operator_result + εₜ
        xₜ₋₁ = xₜ
    end

    # 2. Fill vector and start with last burn-in value
    x_long[1] = xₜ₋₁
    for t in 2:lastindex(x_long)

        xₜ₋₁ = x_long[t-1]

        # Compute observed value.
        εₜ = rand(dist)
        operator_result = sign(α) * sign(xₜ₋₁) * rand(Binomial(Int(abs(xₜ₋₁)), abs_α))
        x_long[t] = operator_result + εₜ
    end

    # 3. Add noise
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end

    return @views x_long[1:d:end]
end

# Method to update SINAR(1) 
function update_dgp_op!(dgp::SINAR1, x_long, dist::DiscreteUnivariateDistribution, d::Int)

    # Extract parameters for easier access
    α = dgp.α

    # Shift window left
    for t in 1:(lastindex(x_long)-1)
        x_long[t] = x_long[t+1]
    end

    # Get previous state xₜ₋₁
    xₜ₋₁ = floor(x_long[end-1])
    εₜ = rand(dist)

    # Apply signed operator: α ⊙ xₜ₋₁
    # -> sgn(α) * sgn(xₜ₋₁) * (|α| ∘ |xₜ₋₁|)
    operator_result = sign(α) * sign(xₜ₋₁) * rand(Binomial(abs(xₜ₋₁), abs(α)))
    xₜ = operator_result + εₜ
    x_long[end] = dgp.add_noise ? xₜ + rand() : xₜ

    return @views x_long[1:d:end]


end

# -------------------------------------------------#
# ---------------  Tobit-INAR(1) methods  ---------#
# -------------------------------------------------#
# Method to initialize with Burn-in phase
function init_dgp_op!(dgp::TINAR1, x_long::Vector{Float64}, dist::DiscreteUnivariateDistribution, d::Int)

    # Extract parameters for easier access
    α = dgp.α
    abs_α = abs(α)
    sign_α = sign(α)
    lower_bound = dgp.L

    # 1. Burn-in Phase    
    # Start heuristically near steady state
    # Convert to Float64 to avoid type issues when adding noise 
    μ_ε = mean(dist) |> abs
    xₜ₋₁ = rand(Poisson(μ_ε / (1 - min(0.99, abs_α)))) |> Float64

    for _ in 1:dgp.burn_in
        εₜ = rand(dist)

        # Signed binomial thinning operator: α ⊙ xₜ₋₁
        # -> sgn(α) * sgn(xₜ₋₁) * (|α| ∘ |xₜ₋₁|)
        operator_result = sign_α * sign(xₜ₋₁) * rand(Binomial(Int(abs(xₜ₋₁)), abs_α))

        # Get observed value with Tobit censoring
        yₜ = operator_result + εₜ
        xₜ = max(lower_bound, yₜ) |> Float64 # convert to Float64 for noise addition
        xₜ₋₁ = xₜ
    end

    # 2. Fill vector and start with last burn-in value
    x_long[1] = xₜ₋₁
    for t in 2:lastindex(x_long)
        εₜ = rand(dist)
        xₜ₋₁ = x_long[t-1]

        # Get observed value with Tobit censoring
        operator_result = sign(α) * sign(xₜ₋₁) * rand(Binomial(Int(abs(xₜ₋₁)), abs_α))
        yₜ = operator_result + εₜ
        x_long[t] = max(lower_bound, yₜ) |> Float64
    end

    # 3. Add noise
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end

    return @views x_long[1:d:end]
end

# Method to update the process
function update_dgp_op!(dgp::TINAR1, x_long::Vector{Float64}, dist::DiscreteUnivariateDistribution, d::Int)

    # Extract parameters for easier access
    α = dgp.α
    abs_α = abs(α)
    sign_α = sign(α)
    lower_bound = dgp.L

    # Shift window left
    for t in 1:(lastindex(x_long)-1)
        x_long[t] = x_long[t+1]
    end

    # Get previous state xₜ₋₁
    xₜ₋₁ = floor(x_long[end-1])
    εₜ = rand(dist)

    # Apply signed operator: α ⊙ xₜ₋₁
    # -> sgn(α) * sgn(xₜ₋₁) * (|α| ∘ |xₜ₋₁|)
    operator_result = sign_α * sign(xₜ₋₁) * rand(Binomial(abs(xₜ₋₁), abs_α))

    yₜ = operator_result + εₜ
    x_new = max(lower_bound, yₜ) |> Float64

    x_long[end] = dgp.add_noise ? x_new + rand() : x_new

    return @views x_long[1:d:end]
end


# -------------------------------------------------#
# ---------------  BAR(1) methods  ----------------#
# -------------------------------------------------#

# Method to initialize BAR(1) when d is Int 
function init_dgp_op!(dgp::BAR1, x_long, dist_error::Nothing, d::Int)
    x_long[1] = rand(Binomial(dgp.n, dgp.parpi))
    for i in 2:lastindex(x_long)
        x = x_long[i-1]
        x_long[i] = rand(Binomial(x, dgp.α)) + rand(Binomial(dgp.n - x, dgp.β))
    end
    # add noise ?
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end
    return @views x_long[1:d:end]
end

# Method to update BAR(1) when d is Int 
function update_dgp_op!(dgp::BAR1, x_long, dist_error::Nothing, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    x = floor(x_long[end-1])
    # add noise?
    if dgp.add_noise
        x_long[end] = rand(Binomial(x, dgp.α)) + rand(Binomial(dgp.n - x, dgp.β)) + rand()
    else
        x_long[end] = rand(Binomial(x, dgp.α)) + rand(Binomial(dgp.n - x, dgp.β))
    end
    return @views x_long[1:d:end]
end



# -------------------------------------------------#
# ---------------  DAR(1) methods  ----------------#
# -------------------------------------------------#

# Method to initialize DAR(1) when d is Int 
function init_dgp_op!(dgp::DAR1, x_long, dist_error, d::Int)

    # First draw
    x_long[1] = rand(dist_error)

    # Fill
    for i in 2:lastindex(x_long)
        if rand(Bernoulli(dgp.α))
            x_long[i] = x_long[i-1]
        else
            x_long[i] = rand(dist_error)
        end
    end

    # add noise ? 
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end

    return @views x_long[1:d:end]
end

# Method to update DAR(1) when d is Int
function update_dgp_op!(dgp::DAR1, x_long, dist_error, d::Int)
    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end
    # add noise ?
    if dgp.add_noise
        if rand(Bernoulli(dgp.α))
            x_long[end] = floor(x_long[end-1]) + rand()
        else
            x_long[end] = rand(dist_error) + rand()
        end
    else
        if rand(Bernoulli(dgp.α))
            x_long[end] = floor(x_long[end-1])
        else
            x_long[end] = rand(dist_error)
        end
    end
    return @views x_long[1:d:end]
end

# -------------------------------------------------#
# ---------------  WDAR(1) methods ----------------#
# -------------------------------------------------#
function init_dgp_op!(dgp::WDAR1, x_long, dist_error::Categorical, d::Int)
    # 1. Initialize first value
    x_long[1] = rand(dist_error)

    # 2. Fill the rest of the path
    for i in 2:lastindex(x_long)
        if rand(Bernoulli(dgp.α))
            prev_idx = Int(floor(x_long[i-1])) # conversion to Int is necessary for indexing
            x_long[i] = rand(dgp.W_samplers[prev_idx])
        else
            x_long[i] = rand(dist_error)
        end
    end

    # Add noise?
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end

    return @views x_long[1:d:end]
end

# Update
function update_dgp_op!(dgp::WDAR1, x_long, dist_error::Categorical, d::Int)

    for i in 1:(lastindex(x_long)-1)
        x_long[i] = x_long[i+1]
    end

    new_val = if rand(Bernoulli(dgp.α))
        x_prev = Int(floor(x_long[end-1])) # conversion to Int is necessary for indexing
        rand(dgp.W_samplers[x_prev])
    else
        rand(dist_error)
    end

    x_long[end] = dgp.add_noise ? new_val + rand() : new_val

    return @views x_long[1:d:end]
end


# --------------------------------------------------------------------#
# ----  Methods to initializize and update vectors for CED -----------#
# --------------------------------------------------------------------#

# Methods when using Conditional Expected Delay approach
function init_dgp_op_ced!(
    dgp::ContinuousDGP, x_long::Vector{Float64}, pool::Vector{Float64}, d::Int
)
    # Fill 'x_long' with randomly drawn values from the pool
    for i in eachindex(x_long)
        x_long[i] = rand(pool)
    end

    return @views x_long[1:d:end]
end



function update_dgp_op_ced!(
    dgp::ContinuousDGP, x_long::Vector{Float64}, pool::Vector{Float64}, d::Int
)
    # 1. Shift window left
    for t in 1:(lastindex(x_long)-1)
        x_long[t] = x_long[t+1]
    end

    # 2. Insert value (with noise option)
    x_long[end] = rand(pool)

    return @views x_long[1:d:end]
end


# Methods when using Conditional Expected Delay approach
function init_dgp_op_ced!(
    dgp::DiscreteDGP, x_long::Vector{Float64}, pool::Vector{Float64}, d::Int
)
    # Fill 'x_long' with randomly drawn values from the pool
    for i in eachindex(x_long)
        x_long[i] = rand(pool)
    end

    # Optional: add noise
    if dgp.add_noise
        for i in eachindex(x_long)
            x_long[i] += rand()
        end
    end

    return @views x_long[1:d:end]
end



function update_dgp_op_ced!(
    dgp::DiscreteDGP, x_long::Vector{Float64}, pool::Vector{Float64}, d::Int
)
    # 1. Shift window left
    for t in 1:(lastindex(x_long)-1)
        x_long[t] = x_long[t+1]
    end

    # 2. Draw a new independent value from the pool
    x_new = rand(pool)

    # 3. Insert value (with noise option)
    x_long[end] = dgp.add_noise ? x_new + rand() : x_new

    return @views x_long[1:d:end]
end


# -------------------------------------------------#
# -------  Methods for d as Vector{Int} -----------#
# -------------------------------------------------#
# # Method to initialize in-control when d is Vector{Int}
# function init_dgp_op!(dgp::ICTS, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
#     rand!(dist_error, x_long)
#     return @views x_long[d]
# end

# # Method to update in-control when d is Vector{Int}
# function update_dgp_op!(dgp::ICTS, x_long, eps_long, dist_error, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = rand(dist_error)
#     return @views x_long[d]
# end

# -------------------------------------------------#
# ---------------  AR(1) methods    ---------------#
# -------------------------------------------------#
# # Initialize AR(1) when d is Vector{Int}
# function init_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
#     x = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
#     x_long[1] = dgp.α * x + rand(dist_error)
#     for i in 2:lastindex(x_long)
#         x_long[i] = dgp.α * x_long[i-1] + rand(dist_error)
#     end
#     return @views x_long[d]
# end

# # Update AR(1) when d is Vector{Int}
# function update_dgp_op!(dgp::AR1, x_long, eps_long, dist_error, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = dgp.α * x_long[end-1] + rand(dist_error)
#     return @views x_long[d]
# end

# -------------------------------------------------#
# ---------------  MA(1) methods    ---------------#
# -------------------------------------------------#
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
# # Method to initialize TEAR(1) when d is Vector{Int}
# function init_dgp_op!(dgp::TEAR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
#     x = rand(dist_error)
#     x_long[1] = rand(Bernoulli(dgp.α)) * x + (1 - dgp.α) * rand(dist_error)
#     for i in 2:lastindex(x_long)
#         x_long[i] = rand(Bernoulli(dgp.α)) * x_long[i-1] + (1 - dgp.α) * rand(dist_error)
#     end
#     return @views x_long[d]
# end

# # Method to update TEAR(1) when d is Vector{Int}
# function update_dgp_op!(dgp::TEAR1, x_long, eps_long, dist_error, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = rand(Bernoulli(dgp.α)) * x_long[end-1] + (1 - dgp.α) * rand(dist_error)
#     return @views x_long[d]
# end

# -------------------------------------------------#
# ---------------  AAR(1) methods   ---------------#
# -------------------------------------------------#
# # Method to initialize AAR(1) when d is Vector{Int}
# function init_dgp_op!(dgp::AAR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
#     # Burn-in prerun 
#     xbiv[1] = rand(dist_error)
#     for i in 2:lastindex(xbiv)
#         xbiv[i] = dgp.α * abs(xbiv[i-1]) + rand(dist_error)
#     end
#     # Initialize sequence
#     x_long[1] = dgp.α * abs(xbiv[end]) + rand(dist_error)
#     for i in 2:lastindex(x_long)
#         x_long[i] = dgp.α * abs(x_long[i-1]) + rand(dist_error)
#     end
#     return @views x_long[d]
# end

# # Method to update AAR(1) when d is Vector{Int}
# function update_dgp_op!(dgp::AAR1, x_long, eps_long, dist_error, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = dgp.α * abs(x_long[2]) + rand(dist_error)
#     return @views x_long[d]
# end


# # Method to initialize QAR(1) when d is Vector{Int}
# function init_dgp_op!(dgp::QAR1, x_long, eps_long, dist_error, d::Vector{Int}, xbiv)
#     # Burn-in prerun
#     xbiv[1] = rand(dist_error)
#     for i in 2:lastindex(xbiv)
#         xbiv[i] = dgp.α * xbiv[i-1]^2 + rand(dist_error)
#     end
#     # Initialize sequence
#     x_long[1] = dgp.α * xbiv[end]^2 + rand(dist_error)
#     for i in 2:lastindex(x_long)
#         x_long[i] = dgp.α * x_long[i-1]^2 + rand(dist_error)
#     end
#     return @views x_long[d]
# end

# # Method to update QAR(1) when d is Vector{Int}
# function update_dgp_op!(dgp::QAR1, x_long, eps_long, dist_error, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = dgp.α * x_long[end-1]^2 + rand(dist_error)
#     return @views x_long[d]
# end


# # Method to initialize INAR(1) when d is Vector{Int}
# function init_dgp_op!(dgp::INAR1, x_long, dist_error::Poisson, d::Vector{Int})
#     x_long[1] = rand(Poisson(dist_error.λ / (1 - dgp.α)))
#     for i in 2:lastindex(x_long)
#         x_long[i] = rand(Binomial(x_long[i-1], dgp.α)) + rand(dist_error)
#     end
#     # add noise
#     for i in axes(x_long, 1)
#         x_long[i] += rand()
#     end
#     return @views x_long[d]
# end

# # Method to update INAR(1) when d is Vector{Int}
# function update_dgp_op!(dgp::INAR1, x_long, dist_error::Poisson, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x = floor(x_long[end-1])
#     x_long[end] = rand(Binomial(x, dgp.α)) + rand(dist_error) + rand()
#     return @views x_long[d]
# end


# # Method to initialize BAR(1) when d is Vector{Int} 
# function init_dgp_op!(dgp::BAR1, x_long, dist_error::Nothing, d::Vector{Int})
#     x_long[1] = rand(Binomial(dgp.n, dgp.parpi))
#     for i in 2:lastindex(x_long)
#         x = x_long[i-1]
#         x_long[i] = rand(Binomial(x, dgp.α)) + rand(Binomial(dgp.n - x, dgp.β))
#     end
#     # add noise
#     for i in axes(x_long, 1)
#         x_long[i] += rand()
#     end
#     return @views x_long[d]
# end

# # Method to update BAR(1) when d is Vector{Int}
# function update_dgp_op!(dgp::BAR1, x_long, dist_error::Nothing, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x = floor(x_long[end-1])
#     x_long[end] = rand(Binomial(x, dgp.α)) + rand(Binomial(dgp.n - x, dgp.β)) + rand()
#     return @views x_long[d]
# end


# # Method to initialize DAR(1) when d is Vector{Int}
# function init_dgp_op!(dgp::DAR1, x_long, dist_error, d::Vector{Int})
#     x_long[1] = rand(dist_error)
#     for i in 2:lastindex(x_long)
#         if rand(Binomial(1, dgp.α)) == 0
#             x_long[i] = rand(dist_error)
#         else
#             x_long[i] = x_long[i-1]
#         end
#     end
#     # add noise
#     for i in axes(x_long, 1)
#         x_long[i] += rand()
#     end
#     return @views x_long[d]
# end

# # Method to update DAR(1) when d is Vector{Int}
# function update_dgp_op!(dgp::DAR1, x_long, dist_error, d::Vector{Int})
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     if rand(Binomial(1, dgp.α)) == 0
#         x_long[end] = rand(dist_error) + rand()
#     else
#         x_long[end] = x_long[end-1]
#     end
#     return @views x_long[d]
# end

# # Initialize AR(1) for CED when d is Int 
# function init_dgp_op_ced!(dgp::AR1, x_long, d::Int)
#     rand!(Normal(0, sqrt(1 / (1 - dgp.α^2))), x_long)
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
#     return @views x_long[1:d:end]
# end

# # Update AR(1) for CED when d is Int
# function update_dgp_op_ced!(dgp::AR1, x_long, d::Int)
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = rand(Normal(0, sqrt(1 / (1 - dgp.α^2))))
#     return @views x_long[1:d:end]
# end



# # Initialize TEAR(1) for CED when d is Int
# function init_dgp_op_ced!(dgp::TEAR1, x_long, d::Int)
#     rand!(Exponential(1), x_long)
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = rand(Exponential(1))
#     return @views x_long[1:d:end]
# end

# # Update TEAR(1) for CED
# function update_dgp_op_ced!(dgp::TEAR1, x_long, d::Int)
#     for i in 1:(lastindex(x_long)-1)
#         x_long[i] = x_long[i+1]
#     end
#     x_long[end] = rand(Exponential(1))
#     return @views x_long[1:d:end]
# end