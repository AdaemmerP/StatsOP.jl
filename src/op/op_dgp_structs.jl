"""
    IC(dist::UnivariateDistribution)

A struct to define the in-control (IC) process. The struct contains one field, namely `dist::UnivariateDistribution`, which is the distribution of the in-control process.   
    
```julia
ic = IC(Normal(0, 1))
```    
"""
struct IC
    dist::UnivariateDistribution
end

"""
    AR1(α::Float64, dist::UnivariateDistribution)

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

