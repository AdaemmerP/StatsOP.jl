"""
    IC(dist::UnivariateDistribution)

A struct to define the in-control (IC) process. The struct contains one field, namely `dist::UnivariateDistribution`, which is the distribution of the in-control process.   
    
```julia
ic = IC(Normal(0, 1))
```    
"""
struct ICTS
    dist::UnivariateDistribution
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
struct AR1
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
struct MA1
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
struct MA2
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
struct TEAR1
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
struct AAR1
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
struct QAR1
    α::Float64
    dist::UnivariateDistribution
end


"""
    INAR1(α, dist, add_noise)

First-Order **I**nteger **N**umerated **A**uto**R**egressive Process.

The INAR(1) model for a time series \$X\_t\$ is defined by:
\$\$X_t = \\alpha \\circ X\_{t-1} + \\epsilon\_t\$\$
where:
* \$\\alpha \\circ X\_{t-1}\$ is a **thinning operator** (e.g., binomial thinning).
* \$\\epsilon\_t\$ is an independent sequence of random variables (the innovation).

# Fields
- `α::Float64`: The autoregressive parameter (thinning probability). Must be in \$(0, 1)\$.
- `dist::DiscreteUnivariateDistribution`: The distribution of the innovation term \$\\epsilon\_t\$.
- `add_noise::Bool`: Flag indicating whether a small amount of uniform noise should be added to the process (usually for simulating continuous-like observations from a discrete process).
"""
struct INAR1
    α::Float64
    dist::DiscreteUnivariateDistribution
    add_noise::Bool
end

# -----------------------------------------------------------------------------

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
struct BAR1
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
\$\$X\_t = (1 - B\_t) X\_{t-1} + B\_t \\epsilon\_t\$\$
where:
* \$B\_t\$ is an i.i.d. Bernoulli random variable with parameter \$\\alpha\$.
* \$\\epsilon\_t\$ is an independent sequence of random variables (the innovation).

# Fields
- `α::Float64`: The autoregressive parameter (probability of selecting the previous value). Must be in \$(0, 1)\$.
- `dist::DiscreteUnivariateDistribution`: The distribution of the innovation term \$\\epsilon\_t\$.
- `add_noise::Bool`: Flag to add small noise.
"""
struct DAR1
    α::Float64
    dist::DiscreteUnivariateDistribution
    add_noise::Bool
end
