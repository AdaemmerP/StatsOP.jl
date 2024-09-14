""" 
    SAR11(dgp_params, m, n, prerun)

First-order spatial autoregressive (SAR(1, 1)) model as defined by Weiß and Kim (2024) on page 6. The struct contains the following fields:

- `dgp_params::Tuple{Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The first element is the parameter α₁, the second element is the parameter α₂, and the third element is the parameter α₃. The requirements to guarantee stationarity for the process are |α₁| < 1, |α₂| < 1, |α₁ + α₂| < 1 - α₃, and |α₁ - α₂| < 1 + α₃.
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `prerun::Int`: A value to initialize the DGP to achieve stationarity. These number of rows and columns will be discarded after the initialization. 

```julia
sar11 = SAR11((0.5, 0.3, 0.2), 11, 11, 100)
```
"""
struct SAR11
  dgp_params::Tuple{Float64, Float64, Float64}
  m::Int
  n::Int
  prerun::Int  
  SAR11(dgp_params, m, n, prerun) = 
    abs(dgp_params[1]) < 1 && 
    abs(dgp_params[2]) < 1 && 
    abs(dgp_params[1] + dgp_params[2] ) < 1 - dgp_params[3] &&
    abs(dgp_params[1] - dgp_params[2] ) < 1 + dgp_params[3]  ?
    new(dgp_params, m, n, prerun) : 
    @warn "
    Note that the parameters provided do not guarantee stationarity.
    The requirements to guarantee stationarity are |α₁| < 1, |α₂| < 1, |α₁ + α₂| < 1 - α₃, and |α₁ - α₂| < 1 + α₃.    
    "
end


""" 
    SINAR11(dgp_params, m, n, prerun)

First-order integer spatial autoregressive (SINAR(1, 1)) model as defined by Weiß and Kim (2024) on page 6. The struct contains the following fields:

- `dgp_params::Tuple{Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The first element is the parameter α₁, the second element is the parameter α₂, and the third element is the parameter α₃. Note that α₁, α₂, and α₃ ∈ [0, 1) and α₁ + α₂ + α₃ < 1 to guarantee stationarity.
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `prerun::Int`: A value to initialize the DGP to guarantee stationarity. These number of rows and columns will be discarded for the final spatial matrix. 

```julia
sar11 = SINAR((0.5, 0.3, 0.2), 11, 11, 100)
```
"""
struct SINAR11
  dgp_params::Tuple{Float64, Float64, Float64}
  m::Int
  n::Int
  prerun::Int
  SINAR11(dgp_params, m, n, prerun) = 
    0.00 <= dgp_params[1] < 1 && 
    0.00 <= dgp_params[2] < 1 && 
    0.00 <= dgp_params[3] < 1 &&  
    dgp_params[1] + dgp_params[2] + dgp_params[3] < 1 ?
    new(dgp_params, m, n, prerun) : 
    @warn "
    Note that the parameters provided do not guarantee stationarity.
    The requirements to guarantee stationarity are α₁, α₂, and α₃ ∈ [0, 1) and α₁ + α₂ + α₃ < 1.    
    "
end

""" 
    SQMA11(dgp_params, eps_params, m, n, prerun)

Spatial quadratic moving-average (SQMA(1, 1)) model as defined by Weiß and Kim (2024) on page 7. The struct contains the following fields:

- `dgp_params::Tuple{Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The first element is the parameter β₁, the second element is the parameter β₂, and the third element is the parameter β₃. 
- `eps_params::Tuple{Int, Int, Int}`: A tuple of the parameters of the DGP, indicating which error terms shall be squared. Note that `eps_params` ∈ {1, 2}.
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `prerun::Int`: A value to initialize the DGP. This value should be set to 1.

```julia
sqma11 = SQMA11((0.5, 0.3, 0.2), (1, 1, 2), 11, 11, 1)
```
"""
struct SQMA11
  dgp_params::Tuple{Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int}
  m::Int
  n::Int
  prerun::Int  
end


""" 
    SQINMA11(dgp_params, eps_params, m, n, prerun)

Spatial quadratic integer moving-average SQINMA(1, 1) model as defined by Weiß and Kim (2024) on page 7. The struct contains the following fields:

- `dgp_params::Tuple{Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The first element is the parameter β₁, the second element is the parameter β₂, and the third element is the parameter β₃. Note that β₁, β₂, and β₃ ∈ [0, 1].
- `eps_params::Tuple{Int, Int, Int}`: A tuple of the parameters of the DGP, indicating which error terms shall be squared. Note that `eps_params` ∈ {1, 2}.
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `prerun::Int`: A value to initialize the DGP. This value should be set to 1.

```julia
sqinma11 = SQINMA11((0.5, 0.3, 0.2), (1, 1, 2), 11, 11, 1)
```
"""
struct SQINMA11
  dgp_params::Tuple{Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int}
  m::Int
  n::Int
  prerun::Int
  SQINMA11(dgp_params, eps_params, m, n, prerun) = 
    0.00 <= dgp_params[1] < 1 && 
    0.00 <= dgp_params[2] < 1 && 
    0.00 <= dgp_params[3] < 1 ?
    new(dgp_params, eps_params, m, n, prerun) : 
    error("Note that β₁, β₂, and β₃ ∈ [0, 1] is required for binomial thinning.")
end

""" 
    SAR1(dgp_params, eps_params, m, n, prerun)

First-order simultaneous autoregressive (SAR(1)) model as defined by Weiß and Kim (2024) on page 8. The struct contains the following fields:

- `dgp_params::Tuple{Float64, Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The first element is the parameter β₁, the second element is the parameter β₂, and the third element is the parameter β₃. Note that β₁, β₂, and β₃ ∈ [0, 1].
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `margin::Int`: The margin for the spatial matrix used for initialization.

```julia
sar1 = SAR1((0.5, 0.3, 0.2, 0.1), 11, 11, 1)
```
"""
struct SAR1
  dgp_params::Tuple{Float64, Float64, Float64, Float64}
  m::Int
  n::Int
  margin::Int  
end


""" 
    BSQMA11(dgp_params, eps_params, m, n, prerun)

Bilateral spatial quadratic moving-average (BSQMA(1, 1)) model as defined by Weiß and Kim (2024) on page 9. The struct contains the following fields:

- `dgp_params::Tuple{Float64, Float64, Float64, Float64}`: A tuple of the parameters of the DGP. The parameters correspond to b₁, b₂, b₃ and b₄, respectively. 
- `eps_params::Tuple{Int, Int, Int, Int}`: A tuple of the parameters of the DGP, indicating which error terms shall be squared. Note that `eps_params` ∈ {1, 2}.
- `m::Int`: The number of rows for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals m + 1.
- `n::Int`: The number of columns for the final "SOP" matrix. Note that the final spatial matrix ("picture") equals n + 1. 
- `prerun::Int`: A value to initialize the DGP. This value should be set to 1.

```julia
bsqma11 = BSQMA11((0.5, 0.3, 0.2, 0.1), (1, 1, 2, 2), 11, 11, 1)
```
"""
struct BSQMA11
  dgp_params::Tuple{Float64, Float64, Float64, Float64}
  eps_params::Tuple{Int, Int, Int, Int}
  m::Int
  n::Int
  prerun::Int
end










  # DGP 3
  # Yₜ₁,ₜ₂ = α_1 * Yₜ₁₋₁,ₜ₂ + α_2 * Yₜ₁,ₜ₂₋₁ + α_3 * Yₜ₁₋₁,ₜ₂₋₁ + εₜ₁,ₜ₂

  # DGP 4
  # Xₜ₁,ₜ₂ = α_1 ∘ Xₜ₁₋₁,ₜ₂ + α_2 ∘ Xₜ₁,ₜ₂₋₁ + α_3 ∘ Xₜ₁₋₁,ₜ₂₋₁ + εₜ₁,ₜ₂

  # DGP 5
  # Yₜ₁,ₜ₂ = α_1 * Yₜ₁₋₁,ₜ₂ + α_2 * Yₜ₁,ₜ₂₋₁ + α_3 * Yₜ₁₋₁,ₜ₂₋₁ + ε*ₜ₁,ₜ₂ + κₜ₁,ₜ₂ ⋅ ε**ₜ₁,ₜ₂

  # DGP 6
  # Xₜ₁,ₜ₂ = α_1 ∘ Xₜ₁₋₁,ₜ₂ + α_2 ∘ Xₜ₁,ₜ₂₋₁ + α_3 ∘ Xₜ₁₋₁,ₜ₂₋₁ + ε*ₜ₁,ₜ₂ + κₜ₁,ₜ₂ ⋅ ε**ₜ₁,ₜ₂

 # DGP 7
  # Yₜ₁,ₜ₂ = α_1 * Yₜ₁₋₁,ₜ₂ + α_2 * Yₜ₁,ₜ₂₋₁ + α_3 * Yₜ₁₋₁,ₜ₂₋₁ + εₜ₁,ₜ₂ + κₜ₁,ₜ₂ ⋅ ε**ₜ₁,ₜ₂

  # DGP 8

