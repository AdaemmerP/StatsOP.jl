export stat_sop_lehmer

function stat_sop_lehmer(
  data::Union{SubArray,Array{T,2}},
  d1::Int, d2::Int;
  chart_choice,
  refinement::Union{Nothing,RefinedType}=nothing,
  add_noise::Bool=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1)
) where {T<:Real}

  # Pre-allocate  
  if isnothing(refinement) #&& chart_choice in 1:4
    p_hat = zeros(3) # classical approach
  else
    p_hat = zeros(6) # refined approach
  end

  sop = zeros(4)
  win = zeros(Int, 4)
  sop_freq = zeros(Int, 24) # factorial(4)  
  used = zeros(Int, 4)

  # Compute m and n based on data
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  # Add noise?
  if add_noise
    data = data .+ rand(noise_dist, size(data, 1), size(data, 2))
  end

  # Compute frequencies of sops    
  sop_frequencies_lehmer!(
    m, n, d1, d2, data, sop, win, sop_freq, used
  )

  # Fill 'p_hat' with sop-frequencies and compute relative frequencies
  fill_p_hat_lehmer!(
    p_hat, chart_choice, refinement, sop_freq, m, n
  )

  # Compute test statistic
  stat = chart_stat_sop(p_hat, chart_choice)

  return stat
end