# In-control struct
struct IC
    dist::UnivariateDistribution
end

# Out-of-control struct -> AR(1) 
struct AR1
    α::Float64
    dist::UnivariateDistribution
end

# Out-of-control struct -> MA(1)
struct MA1
    α::Float64
    dist::UnivariateDistribution
end

# Out-of-control struct -> MA(2)
struct MA2
  α₁::Float64
  α₂::Float64
  dist::UnivariateDistribution
end

# Out-of-control struct -> TEAR(1) 
struct TEAR1
    α::Float64
    dist::UnivariateDistribution
end

# Out-of-control struct -> AAR(1) 
struct AAR1
    α::Float64
    dist::UnivariateDistribution
end

# Out-of-control struct -> QAR(1) 
struct QAR1
    α::Float64
    dist::UnivariateDistribution
end

