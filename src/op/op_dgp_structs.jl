export DiscreteDGPIC,
    ContinuousDGPIC,
    AR1,
    TEAR1,
    MA1,
    MA2,
    QAR1,
    INAR1,
    BAR1,
    DAR1,
    WDAR1,
    TobitINAR1


# Make abstract type for continuous and discrete DGPs
abstract type ContinuousDGP end
abstract type DiscreteDGP end

"""
    ContinuousDGPIC(dist::UnivariateContinuousDistribution)
A struct to define a continuous in-control (IC) process. The struct contains one field, namely `dist::UnivariateContinuousDistribution`, which is the distribution of the in-control process.
```julia
ic = ContinuousDGPIC(Normal(0, 1))
```
"""
struct ContinuousDGPIC
    dist::ContinuousUnivariateDistribution
end


"""
    DiscreteDGPIC(dist::UnivariateDistribution)

A struct to define a discrete in-control (IC) process. The struct contains one field, namely `dist::UnivariateDistribution`, which is the distribution of the in-control process.
    
```julia
ic = DiscreteDGPIC(Poisson(5))
```    
"""
struct DiscreteDGPIC
    dist::DiscreteUnivariateDistribution
    add_noise::Bool
end

"""
    AR1(α::Float64, dist::UnivariateDistribution)

A struct to define an AR(1) process:    

`` \\qquad X_t = α  \\cdot X_{t-1} + \\epsilon_t.``

`dist` specifies the distribution of ``\\epsilon`` using `Distributions.jl`. 

```julia
ar1 = AR1(0.5, Normal(0, 1))
```
"""
struct AR1 <: ContinuousDGP
    α::Float64
    dist::UnivariateDistribution
end

"""
    MA1(α::Float64, dist::UnivariateDistribution)

A struct to define an MA(1) process:        

`` \\qquad X_t = α  \\cdot \\epsilon_{t-1} + \\epsilon_t.``

`dist` specifies the distribution of ``\\epsilon`` using `Distributions.jl`. 

```julia
ma1 = MA1(0.5, Normal(0, 1))
```
"""
struct MA1 <: ContinuousDGP
    α::Float64
    dist::UnivariateDistribution
end

"""
    MA2(α₁::Float64, α₂::Float64, dist::UnivariateDistribution)

A struct to define an MA(2) process:    

`` \\qquad X_t = α₁  \\cdot \\epsilon_{t-1} + α₂  \\cdot \\epsilon_{t-2} + \\epsilon_t.``    


`dist` specifies the distribution of ``\\epsilon`` using `Distributions.jl`. 

```julia
ma2 = MA2(0.5, 0.3, Normal(0, 1))
```
"""
struct MA2 <: ContinuousDGP
    α₁::Float64
    α₂::Float64
    dist::UnivariateDistribution
end

"""
    TEAR1(α::Float64, dist::UnivariateDistribution)

A struct to define a TEAR(1) process:
 
`` \\qquad X_t = B_t^{(\\alpha)} \\cdot X_{t-1}+(1-\\alpha) \\cdot \\epsilon_t.``    

`dist` specifies the distribution of ``\\epsilon`` using `Distributions.jl`. 

```julia
tear1 = TEAR1(0.5, Normal(0, 1))
```
"""
struct TEAR1 <: ContinuousDGP
    α::Float64
    dist::UnivariateDistribution
end

"""

    AAR1(α::Float64, dist::UnivariateDistribution)

A struct to define a AAR(1) (absolute AR) process:

`` \\qquad X_t=\\alpha \\cdot\\left|X_{t-1}\\right|+\\epsilon_t.``

`dist` specifies the distribution of ``\\epsilon`` using `Distributions.jl`. 

```julia
aar1 = AAR1(0.5, Normal(0, 1))
```
"""
struct AAR1 <: ContinuousDGP
    α::Float64
    dist::UnivariateDistribution
end

"""

    QAR1(α::Float64, dist::UnivariateDistribution)

A struct to define a QAR(1) (quadratic AR) process:

`` \\qquad X_t=\\alpha \\cdot X_{t-1}^2+\\epsilon_t.``

`dist` specifies the distribution of ``\\epsilon`` using `Distributions.jl`.

```julia
qar1 = QAR1(0.5, Normal(0, 1))
```
"""
struct QAR1 <: ContinuousDGP
    α::Float64
    dist::UnivariateDistribution
end


"""
    INAR1(α, dist, add_noise)

First-Order **I**nteger **N**umerated **A**uto**R**egressive Process.

The INAR(1) model for a time series \$X_t\$ is defined by:
\$\$X_t = \\alpha \\circ X_{t-1} + \\epsilon_t\$\$
where:
* \$\\alpha \\circ X_{t-1}\$ is a **thinning operator** (e.g., binomial thinning).
* \$\\epsilon_t\$ is an independent sequence of random variables (the innovation).

# Fields
- `α::Float64`: The autoregressive parameter (thinning probability). Must be in \$(0, 1)\$.
- `dist::DiscreteUnivariateDistribution`: The distribution of the innovation term \$\\epsilon_t\$.
- `add_noise::Bool`: Flag indicating whether a small amount of uniform noise should be added to the process (usually for simulating continuous-like observations from a discrete process).
"""
struct INAR1 <: DiscreteDGP
    α::Float64
    dist::DiscreteUnivariateDistribution
    add_noise::Bool
end

"""
    SINAR1(α, dist, add_noise, burn_in)
First-Order **S**igned **I**nteger **N**umerated **A**uto**R**egressive Process

The SINAR(1) model is a signed version of the INAR(1) process, allowing for both positive and negative integer values. The process is defined by:

# Fields
- `α::Float64`: The autoregressive parameter (thinning probability). Can be in
\$(0, 1)\$ for positive dependence or \$( -1, 0)\$ for negative dependence.
- `dist::DiscreteUnivariateDistribution`: The distribution of the innovation term \$\\epsilon_t\$.
- `add_noise::Bool`: Flag indicating whether a small amount of uniform noise should be added to the process (usually for simulating continuous-like observations from a discrete process).
- `burn_in::Int`: The number of initial observations to discard to allow the process to reach its stationary distribution before collecting data for analysis.

"""
struct SINAR1 <: DiscreteDGP
    α::Float64
    dist::DiscreteUnivariateDistribution
    add_noise::Bool
    burn_in::Int
end

# -----------------------------------------------------------------------------
"""    
    TINAR1(α, dist_error, L, add_noise, burn_in)

A struct to define a Tobit-INAR(1) process, which is a censored version of the INAR(1) process. The observed values are censored at a specified lower bound (L).

    The process is defined by:

`` \\qquad X_t = \\max(0, \\alpha \\circ X_{t-1} + \\epsilon_t)``
where:
* \$\\alpha \\circ X_{t-1}\$ is a thinning operator (e.g., binomial thinning).
* \$\\epsilon_t\$ is an independent sequence of random variables (the innovation) with distribution specified by `dist_error`.
* The observed value is censored at a lower bound `L`, meaning that if the generated value is below `L`, it is recorded as `L`.
# Fields
- `α::Float64`: The autoregressive parameter (thinning probability). Must be in \$(0, 1)\$.
- `dist_error::DiscreteUnivariateDistribution`: The distribution of the innovation term \$\\epsilon_t\$.
- `L::Int`: The lower bound for censoring (the "Tobit" threshold).
- `add_noise::Bool`: Flag indicating whether a small amount of uniform noise should be added to the process (usually for simulating continuous-like observations from a discrete process).
- `burn_in::Int`: The number of initial observations to discard to allow the process to reach its stationary distribution before collecting data for analysis.
"""
struct TINAR1{T<:DiscreteUnivariateDistribution} <: DiscreteDGP
    α::Float64
    dist::T
    L::Int
    add_noise::Bool
    burn_in::Int
end

"""
    BAR1(n, ρ, μ, α, β, parpi, dist, add_noise)

**B**inomial **A**uto**R**egressive process of order 1.

The BAR(1) model is a two-state process (0 and 1) that can be extended to model counts up to `n`.
The process maintains a stationary mean `μ` through its construction.

# Fields
- `n::Int64`: The maximum count (the 'n' parameter of the underlying Binomial distribution).
- `ρ::Float64`: The persistence/correlation parameter of the process.
- `μ::Float64`: The stationary mean of the process.
- `α::Float64`: Calculated internal parameter related to \$\\rho\$ and \$\\mu\$.
- `β::Float64`: Calculated internal parameter related to \$\\rho\$ and \$\\mu\$.
- `parpi::Float64`: The probability \$\\pi = \\mu/n\$.
- `dist::Nothing`: Placeholder, as the innovation distribution is implicitly Binomial/Bernoulli via the structure.
- `add_noise::Bool`: Flag to add small noise.
"""
struct BAR1 <: DiscreteDGP
    n::Int64
    ρ::Float64
    μ::Float64
    α::Float64
    β::Float64
    parpi::Float64
    dist::Nothing
    add_noise::Bool
end

"""
    BAR1(n, rho, mu, add_noise)

Convenience constructor for a `BAR1` process.

Calculates the internal parameters `α` and `β` from the provided parameters `n`, `rho`, and `mu`.
"""
function BAR1(n, rho, mu, add_noise)
    parpi = mu / n
    beta = (1 - rho) * parpi
    alpha = beta + rho
    return BAR1(n, rho, mu, alpha, beta, parpi, nothing, add_noise)
end

# -----------------------------------------------------------------------------

"""
    DAR1(α, dist, add_noise)

**D**iscrete **A**uto**R**egressive process of order 1.

The DAR(1) model is a simple discrete-valued time series model defined by:
\$\$X_t = (1 - B_t) X_{t-1} + B_t \\epsilon_t\$\$
where:
* \$B_t\$ is an i.i.d. Bernoulli random variable with parameter \$\\alpha\$.
* \$\\epsilon_t\$ is an independent sequence of random variables (the innovation).

# Fields
- `α::Float64`: The autoregressive parameter (probability of selecting the previous value). Must be in \$(0, 1)\$.
- `dist::DiscreteUnivariateDistribution`: The distribution of the innovation term \$\\epsilon_t\$.
- `add_noise::Bool`: Flag to add small noise.
"""
struct DAR1 <: DiscreteDGP
    α::Float64
    dist::DiscreteUnivariateDistribution
    add_noise::Bool
end


"""
    WDAR1(α, W, dist, add_noise)
**W**eighted **D**iscrete **A**uto**R**egressive process of order 1.
The WDAR(1) model is a generalization of the DAR(1) process that allows for a weighted combination of multiple past values. The process is defined by:
\$\$X_t = \\alpha \\cdot \\sum_{j=1}^{m} W_{j} X_{t-j} + (1 - \\alpha) \\cdot \\epsilon_t\$\$
where:
* \$W_j\$ are the weights for the past values, which sum to 1.
* \$\\epsilon_t\$ is an independent sequence of random variables (the innovation).
# Fields
- `α::Float64`: The autoregressive parameter (weighting factor). Must be in \$(0, 1)\$.
- `W::Matrix{Float64}`: A matrix of weights for the past values. Each column corresponds to a different lag, and the weights in each column should sum to 1.
- `W_samplers::Vector{Distributions.AliasTable}`: A vector of samplers corresponding to the columns of `W`, used for efficient sampling from the weighted past values.
- `dist::Categorical`: The distribution of the innovation term \$\\epsilon_t\$. This is typically a categorical distribution that matches the support of the process.   
- `add_noise::Bool`: Flag to add small noise.
"""
struct WDAR1
    α::Float64
    W::Matrix{Float64}
    W_samplers::Vector{Distributions.AliasTable}
    dist::Categorical
    add_noise::Bool
end

# Constructor computes the samplers for each column of W
function WDAR1(α::Float64, W::Matrix{Float64}, dist::Categorical, add_noise::Bool)
    samps = [sampler(Categorical(W[:, j])) for j in 1:size(W, 2)]
    return WDAR1(α, W, samps, dist, add_noise)
end