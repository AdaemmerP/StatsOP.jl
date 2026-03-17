
"""
    BinomialC{T <: Real}  <: DiscreteUnivariateDistribution 

A struct to define a binomial distribution, which gets multiplied by a constant `c`. 

```julia-repl
dist = BinomialC(0.5, 2)
rand(dist)
````
"""
struct BinomialC{T<:Real} <: DiscreteUnivariateDistribution
  p::T
  c::T
end
BinomialC(p::Real, c::Real) = BinomialC(promote(p, c)...)


"""
    BinomialCvec{T <: Real}  <: DiscreteUnivariateDistribution 

A struct to define a binomial distribution, which gets multiplied by a value from vector `c_vec`. 

```julia-repl
dist = BinomialCvec(0.5, [-10; 10])
rand(dist)
````
"""
struct BinomialCvec{T<:Real} <: DiscreteUnivariateDistribution
  p::T
  c_vec::Vector{T}
end
BinomialCvec(p, c_vec) = BinomialCvec(p, convert.(typeof(p), c_vec))


# Create in-place methods for BinomialC when c is constant
function Random.rand!(d::BinomialC{T}, mat::Matrix{Float64}) where {T}
  rand!(Binomial(1, d.p), mat)
  lmul!(d.c, mat)
  return mat
end

# Create in-place methods for BinomialCvec when c is a vector
function Random.rand!(d::BinomialCvec{T}, mat::Matrix{Float64}) where {T}

  # fill matrix with binomial draws
  rand!(Binomial(1, d.p), mat)

  # multiply with c
  for j in 1:size(mat, 2)
    for i in 1:size(mat, 1)

      if rand() < 0.5
        mat[i, j] = mat[i, j] * d.c_vec[1]
      else
        mat[i, j] = mat[i, j] * d.c_vec[2]
      end

    end
  end

  return mat
end


"""
    PoiBin{T <: Real}  <: DiscreteUnivariateDistribution

A struct to define a Poisson-Binomial distribution. 
  
```julia-repl
dist = PoiBin(0.5, 5)
rand(dist)
```
"""
struct PoiBin{T<:Real} <: DiscreteUnivariateDistribution where {T}
  # Parameters for Binomial distribution  
  p::T
  # Constant for γ
  γ::T
end
PoiBin(p::Real, γ::Real) = PoiBin(promote(p, γ)...)

# Create method for single PoiBin draw
Base.rand(d::PoiBin) = rand(Binomial(1, d.p)) * rand(Poisson(d.γ))

# Create method for in-place multiple draws
function Random.rand!(d::PoiBin, mat::Matrix)# where T

  # fill matrix with binomial draws
  rand!(Binomial(1, d.p), mat)

  # multiply with Poisson draws
  for j in 1:size(mat, 2)
    for i in 1:size(mat, 1)
      mat[i, j] = mat[i, j] * rand(Poisson(d.γ))
    end
  end

  return mat
end

"""
    ZIP <: DiscreteUnivariateDistribution

A struct to define a zero-inflated Poisson distribution.
  
```julia-repl
dist = ZIP(5.0, 0.2)
rand(dist)
```
"""
struct ZIP{T<:Real} <: DiscreteUnivariateDistribution where {T}
  λ::T
  ω::T
end
ZIP(λ::Real, ω::Real) = ZIP(promote(λ, ω)...)

# add rand method for ZIP process 
Base.rand(d::ZIP) = rand(Binomial(1, 1 - d.ω)) * rand(Poisson(d.λ / (1 - d.ω)))
Statistics.mean(d::ZIP) = d.λ

# struct for BinNorm
struct BinNorm{T<:Real} <: ContinuousUnivariateDistribution where {T}
  μ₁::T
  μ₂::T
  σ₁::T
  σ₂::T
end
BinNorm(μ₁::Real, μ₂::Real, σ₁::Real, σ₂::Real) = BinNorm(promote(μ₁, μ₂, σ₁, σ₂)...)

# add rand method for BinNorm 
function Base.rand(d::BinNorm)

  if rand(Binomial(1, 0.5)) == 0
    return rand(Normal(d.μ₁, d.σ₁))
  else
    return rand(Normal(d.μ₂, d.σ₂))
  end

end

function Random.rand!(d::BinNorm, mat::Matrix{Float64})

  for j in 1:size(mat, 2)
    for i in 1:size(mat, 1)
      x = rand(d)
      mat[i, j] = x
    end
  end

  return mat

end
