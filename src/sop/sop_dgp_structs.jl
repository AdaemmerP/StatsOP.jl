# Abstract type for all DGPS
abstract type SpatialDGP end

# ---------------------------------------------#
#   In-control spatial process (ICSP)          #
# ---------------------------------------------#
"""
    ICSP(M_rows, N_cols, dist)

A struct to define an independent and identically distributed (IID) process for in-control
  
- `M_rows::Int`: The number of rows for the image.
- `N_cols::Int`: The number of columns for image.
- `dist::UnivariateDistribution`: A distribution from the Distributions.jl package.
"""
struct ICSTS
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
end


# ---------------------------------------------#
#         Out-of-control spatial processes     #
# ---------------------------------------------#
""" 
    SAR11(dgp_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a first-order spatial autoregressive (SAR(1, 1)) process:

`` \\qquad \\qquad Y_{t_1, t_2}=\\alpha_1 \\cdot Y_{t_1-1, t_2}+\\alpha_2 \\cdot Y_{t_1, t_2-1}+\\alpha_3 \\cdot Y_{t_1-1, t_2-1}+\\varepsilon_{t_1, t_2} ``  

    Yₜ₁,ₜ₂ = α₁ ⋅ Yₜ₁₋₁,ₜ₂ + α₂ ⋅ Yₜ₁,ₜ₂₋₁ + α₃ ⋅ Yₜ₁₋₁,ₜ₂₋₁ + εₜ₁,ₜ₂

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
struct SAR11 <: SpatialDGP
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
    SAR22(dgp_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a first-order spatial autoregressive (SAR(1, 1)) process:

`` \\qquad \\qquad Y_{t_1, t_2} = 
              \\alpha_1 \\cdot Y_{t_1-1, t_2} 
            + \\alpha_2 \\cdot Y_{t_1,   t_2-1} 
            + \\alpha_3 \\cdot Y_{t_1-1, t_2-1} 
            + \\alpha_4 \\cdot Y_{t_1-2, t_2} 
            + \\alpha_5 \\cdot Y_{t_1,   t_2-2} 
            + \\alpha_6 \\cdot Y_{t_1-2, t_2-1} 
            + \\alpha_7 \\cdot Y_{t_1-1, t_2-2} 
            + \\alpha_8 \\cdot Y_{t_1-2, t_2-2} 
            + \\varepsilon_{t_1, t_2}. ``

            Yₜ₁ = α₁ ⋅ Yₜ₁₋₁ + 
                  α₂ ⋅ Yₜ₁,ₜ₂₋₁ + 
                  α₃ ⋅ Yₜ₁₋₁,ₜ₂₋₁ + 
                  α₄ ⋅ Yₜ₁₋₂,ₜ₂ + 
                  α₅ ⋅ Yₜ₁,ₜ₂₋₂ + 
                  α₆ ⋅ Yₜ₁₋₂,ₜ₂₋₁ + 
                  α₇ ⋅ Yₜ₁₋₁,ₜ₂₋₂ + 
                  α₈ ⋅ Yₜ₁₋₂,ₜ₂₋₂ + 
                  εₜ₁,ₜ₂  

"""
struct SAR22 <: SpatialDGP
  dgp_params::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
  prerun::Int  
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
struct SINAR11 <: SpatialDGP
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
```julia
sqma11 = SQMA11((0.5, 0.3, 0.2), (1, 1, 2), 10, 10, Normal(0,1), nothing, 1)
```
"""
struct SQMA11 <: SpatialDGP
  dgp_params::Tuple{Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
end


""" 
    SQMA22(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a Spatial quadratic moving-average (SQMA(1, 1)):

  ``
  \\quad \\quad Y_{t_1, t_2} = 
  \\beta_1 \\cdot \\varepsilon_{t_1-1, t_2}^a 
+ \\beta_2 \\cdot \\varepsilon_{t_1, t_2-1}^b 
+ \\beta_3 \\cdot \\varepsilon_{t_1-1, t_2-1}^c 
+ \\beta_4 \\cdot \\varepsilon_{t_1-2, t_2}^d 
+ \\beta_5 \\cdot \\varepsilon_{t_1, t_2-2}^e 
+ \\beta_6 \\cdot \\varepsilon_{t_1-2, t_2-1}^f 
+ \\beta_7 \\cdot \\varepsilon_{t_1-1, t_2-2}^g 
+ \\beta_8 \\cdot \\varepsilon_{t_1-2, t_2-2}^h 
+ \\varepsilon_{t_1, t_2}.
``

Yₜ₁,ₜ₂ = β₁ ⋅ εₜ₁₋₁,ₜ₂ᵃ + 
          β₂ ⋅ εₜ₁,ₜ₂₋₁ᵇ + 
          β₃ ⋅ εₜ₁₋₁,ₜ₂₋₁ᶜ + 
          β₄ ⋅ εₜ₁₋₂,ₜ₂ᵈ + 
          β₅ ⋅ εₜ₁,ₜ₂₋₂ᵉ + 
          β₆ ⋅ εₜ₁₋₂,ₜ₂₋₁ᶠ + 
          β₇ ⋅ εₜ₁₋₁,ₜ₂₋₂ᵍ + 
          β₈ ⋅ εₜ₁₋₂,ₜ₂₋₂ʰ + 
          εₜ₁,ₜ₂

"""
struct SQMA22 <: SpatialDGP
  dgp_params::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int, Int, Int, Int, Int, Int}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
end

""" 
    SQINMA11(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao, prerun)

A struct to define a Spatial quadratic integer moving-average SQINMA(1, 1):    

``\\qquad X_{t_1, t_2}=\\beta_1 \\circ \\epsilon_{t_1-1, t_2}^a+\\beta_2 \\circ \\epsilon_{t_1, t_2-1}^b+\\beta_3 \\circ \\epsilon_{t_1-1, t_2-1}^c+\\epsilon_{t_1, t_2}.``

Xₜ₁,ₜ₂ = β₁ ⋅ εₜ₁₋₁,ₜ₂ᵃ + 
         β₂ ⋅ εₜ₁,ₜ₂₋₁ᵇ + 
         β₃ ⋅ εₜ₁₋₁,ₜ₂₋₁ᶜ + 
         εₜ₁,ₜ₂


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
struct SQINMA11 <: SpatialDGP
  dgp_params::Tuple{Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Nothing
  SQINMA11(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao) = 
    0.00 <= dgp_params[1] < 1 && 
    0.00 <= dgp_params[2] < 1 && 
    0.00 <= dgp_params[3] < 1 ?
    new(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao) : 
    error("Note that β₁, β₂, and β₃ ∈ [0, 1] is required for binomial thinning.")
end

""" 
    SAR1(dgp_params, M_rows, N_cols, dist, dist_ao, margin)

A struct to define a first-order simultaneous autoregressive (SAR(1)) model:    

``\\qquad Y_{t_1, t_2}=a_1 \\cdot Y_{t_1-1, t_2}+a_2 \\cdot Y_{t_1, t_2-1}+a_3 \\cdot Y_{t_1, t_2+1}+a_4 \\cdot Y_{t_1+1, t_2}+\\varepsilon_{t_1, t_2}.``

Yₜ₁,ₜ₂ = a₁ ⋅ Yₜ₁₋₁,ₜ₂ + 
         a₂ ⋅ Yₜ₁,ₜ₂₋₁ + 
         a₃ ⋅ Yₜ₁,ₜ₂₊₁ + 
         a₄ ⋅ Yₜ₁₊₁,ₜ₂ + 
         εₜ₁,ₜ₂


- `dgp_params::Tuple{a₁::Float64, a₂::Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The first element is the parameter a₁, the second element is the parameter a₂, and the third element is the parameter a₃. a₄.
- `M_rows::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `N_cols::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `dist::Distribution`: A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Union{Nothing, Distribution}`: Nothing or a distribution for additive outliers. For additive outliers you can use any univariate distribution from the `Distributions.jl` package.
- `margin::Int`: The margin for the spatial matrix used for initialization.

```julia
sar1 = SAR1((0.5, 0.3, 0.2, 0.1), 10, 10, Normal(0, 1), nothing, 20) 
```
"""
struct SAR1  <: SpatialDGP
  dgp_params::Tuple{Float64, Float64, Float64, Float64}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Union{Nothing, UnivariateDistribution}
  margin::Int
end


""" 
    BSQMA11(dgp_params, eps_params, M_rows, N_cols, dist, dist_ao)

A struct to define a bilateral spatial quadratic moving-average (BSQMA(1, 1)) process:

Y_{t_1, t_2}=b_1 \\cdot \\varepsilon_{t_1-1, t_2-1}^a+b_2 \\cdot \\varepsilon_{t_1+1, t_2-1}^b+b_3 \\cdot \\varepsilon_{t_1+1, t_2+1}^c+b_4 \\cdot \\varepsilon_{t_1-1, t_2+1}^d+\\varepsilon_{t_1, t_2},

Yₜ₁,ₜ₂ = b₁ ⋅ εₜ₁₋₁,ₜ₂₋₁ᵃ + 
         b₂ ⋅ εₜ₁₊₁,ₜ₂₋₁ᵇ + 
         b₃ ⋅ εₜ₁₊₁,ₜ₂₊₁ᶜ + 
         b₄ ⋅ εₜ₁₋₁,ₜ₂₊₁ᵈ + 
         εₜ₁,ₜ₂
    
- `dgp_params::Tuple{Float64, Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The parameters correspond to b₁, b₂, b₃ and b₄, respectively. 
- `eps_params::Tuple{Int, Int, Int, Int}`: A tuple of the parameters of the DGP, indicating which error terms shall be squared. Note that `eps_params` ∈ {1, 2}.
- `M_rows::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `N_cols::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `dist::Distribution`: A distribution for \\varepsilon_{t_1, t_2}. You can use any univariate distribution from the `Distributions.jl` package.
- `dist_ao::Nothing`: Nothing.

```julia
bsqma11 = BSQMA11((0.5, 0.3, 0.2, 0.1), (1, 1, 2, 2), 10, 10, Normal(0, 1), nothing)
```
"""
struct BSQMA11 <: SpatialDGP
  dgp_params::Tuple{Float64, Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int, Int}
  M_rows::Int
  N_cols::Int
  dist::UnivariateDistribution
  dist_ao::Nothing
end







