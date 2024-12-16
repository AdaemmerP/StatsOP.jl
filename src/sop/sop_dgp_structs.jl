"""
    ICSP(M_rows, N_cols, dist)

A struct to define an independent and identically distributed (IID) process for in-control
  
- `M_rows::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `N_cols::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1.
- `dist::UnivariateDistribution`: A distribution from the Distributions.jl package.
"""
struct ICSP
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
end


""" 
    SAR11(dgp_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a first-order spatial autoregressive (SAR(1, 1)) process:

`` \\qquad \\qquad Y_{t_1, t_2}=\\alpha_1 \\cdot Y_{t_1-1, t_2}+\\alpha_2 \\cdot Y_{t_1, t_2-1}+\\alpha_3 \\cdot Y_{t_1-1, t_2-1}+\\varepsilon_{t_1, t_2} ``  

- `dgp_params::Tuple(α₁::Float64, α₂::Float64, α₃::Float64)` The requirements to guarantee stationarity for the process are |α₁| < 1, |α₂| < 1, |α₁ + α₂| < 1 - α₃, and |α₁ - α₂| < 1 + α₃.
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `prerun::Int`: A value to initialize the DGP to achieve stationarity. These number of rows and columns will be discarded after the initialization. 
- `dist::Distribution`: A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Union{Nothing, Distribution}`: Nothing or a distribution for additive outliers. For additive outliers you can use any univariate distribution from the `Distributions.jl` package.
- `prerun::Int`: A value to initialize the DGP to guarantee stationarity. These number of rows and columns will be discarded for the final spatial matrix.
```julia
sar11 = SAR11((0.5, 0.3, 0.2), 11, 11, Normal(0, 1), nothing, 100)
```
"""
struct SAR11
  dgp_params::Tuple{Float64, Float64, Float64}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
  prerun::Int  
  SAR11(dgp_params, M_rows, N_cols, dist, dist_ao, prerun) =
    abs(dgp_params[1]) < 1 && 
    abs(dgp_params[2]) < 1 && 
    abs(dgp_params[1] + dgp_params[2] ) < 1 - dgp_params[3] &&
    abs(dgp_params[1] - dgp_params[2] ) < 1 + dgp_params[3]  ?
    new(dgp_params, M_rows, N_cols, dist, dist_ao, prerun) :
    @warn "
    Note that the parameters provided do not guarantee stationarity.
    The requirements to guarantee stationarity are |α₁| < 1, |α₂| < 1, |α₁ + α₂| < 1 - α₃, and |α₁ - α₂| < 1 + α₃.    
    "
end


""" 
    SINAR11(dgp_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a first-order integer spatial autoregressive (SINAR(1, 1)) model:


`` \\qquad X_{t_1, t_2}=\\alpha_1 \\circ X_{t_1-1, t_2}+\\alpha_2 \\circ X_{t_1, t_2-1}+\\alpha_3 \\circ X_{t_1-1, t_2-1}+\\varepsilon_{t_1, t_2}.`` 

- `dgp_params::Tuple(α₁::Float64, α₂::Float64, α₃::Float64)` Note that α₁, α₂, and α₃ ∈ [0, 1) and α₁ + α₂ + α₃ < 1 to guarantee stationarity.
- `m::Int` The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int` The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `prerun::Int` A value to initialize the DGP to guarantee stationarity. These number of rows and columns will be discarded for the final spatial matrix. 
- `dist::Distribution` A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Union{Nothing, Distribution}` Nothing or a distribution for additive outliers. For additive outliers you can use any univariate distribution from the `Distributions.jl` package.
- `prerun::Int` A value to initialize the DGP to achieve stationarity. These number of rows and columns will be discarded after the initialization.
```julia
sar11 = SINAR((0.5, 0.3, 0.2), 11, 11, 100)
```
"""
struct SINAR11
  dgp_params::Tuple{Float64, Float64, Float64}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
  prerun::Int
  SINAR11(dgp_params, m, n, dist, dist_ao, prerun) =
    0.00 <= dgp_params[1] < 1 && 
    0.00 <= dgp_params[2] < 1 && 
    0.00 <= dgp_params[3] < 1 &&  
    dgp_params[1] + dgp_params[2] + dgp_params[3] < 1 ?
    new(dgp_params, m, n, dist, dist_ao, prerun) :
    @warn "
    Note that the parameters provided do not guarantee stationarity.
    The requirements to guarantee stationarity are α₁, α₂, and α₃ ∈ [0, 1) and α₁ + α₂ + α₃ < 1.    
    "
end

""" 
    SQMA11(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a Spatial quadratic moving-average (SQMA(1, 1)):

`` \\qquad  Y_{t_1, t_2}=\\beta_1 \\cdot \\varepsilon_{t_1-1, t_2}^a+\\beta_2 \\cdot \\varepsilon_{t_1, t_2-1}^b+\\beta_3 \\cdot \\varepsilon_{t_1-1, t_2-1}^c+\\varepsilon_{t_1, t_2}``


- `dgp_params::Tuple(β₁::Float64, β₂::Float64, β₃::Float64)`: A tuple of the parameters of the DGP. The first element is the parameter β₁, the second element is the parameter β₂, and the third element is the parameter β₃. 
- `eps_params::Tuple(a::Int, b::Int, c::Int)`: A tuple of the parameters of the DGP, indicating which error terms shall be squared. Note that `eps_params` ∈ {1, 2}.
- `M_rows::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `N_cols::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `dist::Distribution` A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Nothing`.
- `prerun::Int` A value to initialize the DGP to achieve stationarity. These number of rows and columns will be discarded after the initialization.
```julia
sqma11 = SQMA11((0.5, 0.3, 0.2), (1, 1, 2), 10, 10, Normal(0,1), nothing, 1)
```
"""
struct SQMA11
  dgp_params::Tuple{Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
  prerun::Int
end


""" 
    SQINMA11(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a Spatial quadratic integer moving-average SQINMA(1, 1):    

``\\qquad X_{t_1, t_2}=\\beta_1 \\circ \\epsilon_{t_1-1, t_2}^a+\\beta_2 \\circ \\epsilon_{t_1, t_2-1}^b+\\beta_3 \\circ \\epsilon_{t_1-1, t_2-1}^c+\\epsilon_{t_1, t_2}.``


- `dgp_params::Tuple(β₁::Float64, β₂::Float64, β₃::Float64).` Note that β₁, β₂, and β₃ ∈ [0, 1].
- `eps_params::Tuple(a::Int, b::Int, c::Int).` A tuple of the parameters of the DGP, indicating which error terms shall be squared. Note that `eps_params` ∈ {1, 2}.
- `M_rows::Int.` The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `N_cols::Int.` The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `dist::Distribution` A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Nothing`.
- `prerun::Int` A value to initialize the DGP to achieve stationarity. These number of rows and columns will be discarded after the initialization.

```julia
sqinma11 = SQINMA11((0.5, 0.3, 0.2), (1, 1, 2), 10, 10, Normal(0, 1), nothing, 1)
```
"""
struct SQINMA11
  dgp_params::Tuple{Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Nothing
  prerun::Int
  SQINMA11(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao, prerun) = 
    0.00 <= dgp_params[1] < 1 && 
    0.00 <= dgp_params[2] < 1 && 
    0.00 <= dgp_params[3] < 1 ?
    new(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao, prerun) : 
    error("Note that β₁, β₂, and β₃ ∈ [0, 1] is required for binomial thinning.")
end

""" 
    SAR1(dgp_params, M_rows, N_cols, dist, dist_ao, margin)

A struct to define a first-order simultaneous autoregressive (SAR(1)) model:    

``\\qquad Y_{t_1, t_2}=a_1 \\cdot Y_{t_1-1, t_2}+a_2 \\cdot Y_{t_1, t_2-1}+a_3 \\cdot Y_{t_1, t_2+1}+a_4 \\cdot Y_{t_1+1, t_2}+\\varepsilon_{t_1, t_2}.``


- `dgp_params::Tuple{a₁::Float64, a₂::Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The first element is the parameter a₁, the second element is the parameter a₂, and the third element is the parameter a₃. a₄.
- `M_rows::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `N_cols::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `dist::Distribution`: A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Union{Nothing, Distribution}`: Nothing or a distribution for additive outliers. For additive outliers you can use any univariate distribution from the `Distributions.jl` package.
- `margin::Int`: The margin for the spatial matrix used for initialization.

```julia
sar1 = SAR1((0.5, 0.3, 0.2, 0.1), 10, 10, Normal(0, 1), nothing, 100) 
```
"""
struct SAR1
  dgp_params::Tuple{Float64, Float64, Float64, Float64}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
  margin::Int
end


""" 
    BSQMA11(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a bilateral spatial quadratic moving-average (BSQMA(1, 1)) process:

Y_{t_1, t_2}=b_1 \\cdot \\varepsilon_{t_1-1, t_2-1}^a+b_2 \\cdot \\varepsilon_{t_1+1, t_2-1}^b+b_3 \\cdot \\varepsilon_{t_1+1, t_2+1}^c+b_4 \\cdot \\varepsilon_{t_1-1, t_2+1}^d+\\varepsilon_{t_1, t_2},
    
- `dgp_params::Tuple{Float64, Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The parameters correspond to b₁, b₂, b₃ and b₄, respectively. 
- `eps_params::Tuple{Int, Int, Int, Int}`: A tuple of the parameters of the DGP, indicating which error terms shall be squared. Note that `eps_params` ∈ {1, 2}.
- `M_rows::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `N_cols::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `dist::Distribution`: A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Nothing`: Nothing.
- `prerun::Int`: A value to initialize the DGP. This value should be set to 1.

```julia
bsqma11 = BSQMA11((0.5, 0.3, 0.2, 0.1), (1, 1, 2, 2), 10, 10, Normal(0, 1), nothing, 1)
```
"""
struct BSQMA11
  dgp_params::Tuple{Float64, Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int, Int}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Nothing
  prerun::Int
end







